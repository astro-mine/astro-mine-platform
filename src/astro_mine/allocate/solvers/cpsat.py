"""``CpSatSolver`` — the CP-SAT search driver behind the ``Solver`` strategy (RM-P1-ALLOC-02).

The primary engine (allocate.md §4/§11): it lowers the solver-neutral
:class:`~astro_mine.allocate.AllocationIR` to CP-SAT via the pure
:func:`~astro_mine.allocate.model.compile.cpsat.lower_to_cpsat` compiler, drives the search
within a :class:`~astro_mine.allocate.SolveBudget`, and streams an :class:`Incumbent` per
improving solution (objective, best dual bound, derived gap) — the anytime contract's data.

Separation of concerns holds the line the charter draws: the *lowering* is pure and decoupled
(``model/compile/cpsat.py``); this module only *searches* and *decodes*. The realized objective
of every incumbent is recomputed from the **float IR objective terms** (never CP-SAT's scaled
integer objective), so it agrees exactly with :func:`~astro_mine.allocate.verify_feasible`, the
independent oracle the planner re-checks every emitted plan against — a solver bug can never
smuggle an infeasible or mis-scored plan past the shield (allocate.md §9).

**Budget → CP-SAT parameters:** ``seed`` → ``random_seed``; ``target_gap`` → ``relative_gap_limit``;
``deterministic`` → a fixed single search worker plus a **deterministic** time bound (not
wall-clock), so the seeded golden-plan gate is byte-reproducible regardless of machine speed
(RM-P1-ALLOC-07); a non-deterministic solve maps ``wall_clock_deadline_s`` → ``max_time_in_seconds``
and ``workers`` → ``num_search_workers``. ``hints`` seed CP-SAT ``AddHint`` warm starts, which
CP-SAT verifies and never trusts.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import replace
from typing import Any

from ortools.sat.python import cp_model

from astro_mine.allocate.api.model import SolveBudget
from astro_mine.allocate.enums import AllocationStatus, VariableKind, VariableSemantic
from astro_mine.allocate.model.compile.cpsat import (
    OBJECTIVE_SCALE,
    TIME_SCALE,
    CpSatModel,
    lower_to_cpsat,
)
from astro_mine.allocate.model.ir.model import AllocationIR
from astro_mine.allocate.solvers._common import Pair, build_plan, realized_objective
from astro_mine.allocate.solvers.base import Incumbent
from astro_mine.core.messages.enums import TaskKind

__all__ = ["CpSatSolver"]

#: Deterministic mode runs a single fixed search worker so the seeded golden plan is
#: byte-identical across machines (allocate.md §8; RM-P1-ALLOC-07).
_DETERMINISTIC_WORKERS: int = 1

#: The deterministic-time bound (machine-independent units) a deterministic solve falls back to
#: when the budget states no deadline — ample for the Phase-1 instance sizes, bounded for safety.
_DEFAULT_DETERMINISTIC_TIME: float = 60.0


def _map_terminal_status(status: Any) -> AllocationStatus:
    """Map a terminal CP-SAT status with no usable plan to an :class:`AllocationStatus`."""
    if status == cp_model.INFEASIBLE:
        return AllocationStatus.INFEASIBLE
    return AllocationStatus.UNKNOWN


class CpSatSolver:
    """A :class:`~astro_mine.allocate.solvers.base.Solver` backed by OR-Tools CP-SAT.

    Constructed with the presentation context a plan decode needs — the Core ``TaskKind`` per task
    (so a decoded :class:`~astro_mine.allocate.ScheduledTask` carries its real kind) and the
    per-``(task, asset)`` durations (so a scheduled task's ``end_s`` reflects its operation length,
    defaulting to a zero-length point when unknown). The strategy itself stays IR-only.
    """

    def __init__(
        self,
        *,
        task_kinds: Mapping[str, TaskKind],
        durations: Mapping[Pair, float] | None = None,
        time_scale: int = TIME_SCALE,
        objective_scale: int = OBJECTIVE_SCALE,
    ) -> None:
        self._task_kinds = dict(task_kinds)
        self._durations = dict(durations or {})
        self._time_scale = time_scale
        self._objective_scale = objective_scale
        self._last_wall_time_s = 0.0

    @property
    def last_wall_time_s(self) -> float:
        """The wall-clock seconds the most recent :meth:`solve` consumed (for provenance)."""
        return self._last_wall_time_s

    # --- Solver strategy ---------------------------------------------------------

    def solve(
        self,
        ir: AllocationIR,
        budget: SolveBudget,
        *,
        hints: Mapping[str, float] | None = None,
    ) -> Iterator[Incumbent]:
        """Search the lowered IR within ``budget`` and stream improving incumbents."""
        compiled = lower_to_cpsat(
            ir, time_scale=self._time_scale, objective_scale=self._objective_scale
        )
        if hints:
            self._apply_hints(compiled, ir, hints)

        solver = cp_model.CpSolver()
        self._configure(solver.parameters, budget, has_objective=compiled.has_objective)
        collector = _IncumbentCollector(ir, compiled, self._task_kinds, self._durations)
        status = solver.Solve(compiled.model, collector)
        self._last_wall_time_s = solver.WallTime()

        return iter(self._finalize(ir, compiled, solver, status, collector.incumbents))

    # --- configuration -----------------------------------------------------------

    def _configure(self, params: Any, budget: SolveBudget, *, has_objective: bool) -> None:
        if budget.seed is not None:
            params.random_seed = budget.seed
        if budget.target_gap is not None and has_objective:
            params.relative_gap_limit = budget.target_gap
        if budget.deterministic:
            # A fixed worker count + a deterministic (not wall-clock) time bound make the solve
            # byte-reproducible; the wall-clock deadline is reinterpreted as deterministic units so
            # a requested budget still bounds the search without reintroducing machine-dependence.
            params.num_search_workers = _DETERMINISTIC_WORKERS
            params.max_deterministic_time = (
                budget.wall_clock_deadline_s
                if budget.wall_clock_deadline_s is not None
                else _DEFAULT_DETERMINISTIC_TIME
            )
        else:
            if budget.workers is not None:
                params.num_search_workers = budget.workers
            if budget.wall_clock_deadline_s is not None:
                params.max_time_in_seconds = budget.wall_clock_deadline_s

    def _apply_hints(
        self, compiled: CpSatModel, ir: AllocationIR, hints: Mapping[str, float]
    ) -> None:
        var_by_id = {v.id: v for v in ir.variables}
        for var_id, value in hints.items():
            cp_var = compiled.variables.get(var_id)
            decl = var_by_id.get(var_id)
            if cp_var is None or decl is None:
                continue  # a hint for a variable this IR does not carry is silently ignored
            if decl.kind is VariableKind.CONTINUOUS:
                compiled.model.AddHint(cp_var, round(value * compiled.time_scale))
            else:
                compiled.model.AddHint(cp_var, round(value))

    # --- finalization ------------------------------------------------------------

    def _finalize(
        self,
        ir: AllocationIR,
        compiled: CpSatModel,
        solver: Any,
        status: Any,
        stream: list[Incumbent],
    ) -> list[Incumbent]:
        """Append/patch the authoritative terminal incumbent onto the streamed intermediates."""
        if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            terminal_status = (
                AllocationStatus.OPTIMAL
                if status == cp_model.OPTIMAL
                else AllocationStatus.FEASIBLE
            )
            terminal = _extract_incumbent(
                solver, ir, compiled, self._task_kinds, self._durations, terminal_status
            )
            if status == cp_model.OPTIMAL and compiled.has_objective:
                terminal = replace(terminal, gap=0.0)
            if stream:
                stream[-1] = terminal
            else:
                stream = [terminal]
            return stream
        return [
            Incumbent(
                objective=0.0,
                bound=None,
                gap=None,
                plan=[],
                status=_map_terminal_status(status),
                is_feasible=False,
            )
        ]


class _IncumbentCollector(cp_model.CpSolverSolutionCallback):
    """Records an :class:`Incumbent` per improving CP-SAT solution — the anytime stream."""

    def __init__(
        self,
        ir: AllocationIR,
        compiled: CpSatModel,
        task_kinds: Mapping[str, TaskKind],
        durations: Mapping[Pair, float],
    ) -> None:
        super().__init__()
        self._ir = ir
        self._compiled = compiled
        self._task_kinds = task_kinds
        self._durations = durations
        self.incumbents: list[Incumbent] = []

    def on_solution_callback(self) -> None:
        self.incumbents.append(
            _extract_incumbent(
                self,
                self._ir,
                self._compiled,
                self._task_kinds,
                self._durations,
                AllocationStatus.FEASIBLE,
            )
        )


def _extract_incumbent(
    reader: Any,
    ir: AllocationIR,
    compiled: CpSatModel,
    task_kinds: Mapping[str, TaskKind],
    durations: Mapping[Pair, float],
    status: AllocationStatus,
) -> Incumbent:
    """Decode one CP-SAT solution (``reader`` exposes ``Value`` / ``BestObjectiveBound``)."""
    cp_vars = compiled.variables
    assignment: dict[str, str] = {}
    starts: dict[str, float] = {}
    for var in ir.variables:
        if var.semantic is VariableSemantic.ASSIGNMENT and var.task_ref and var.asset_ref:
            if reader.Value(cp_vars[var.id]) == 1:
                assignment[var.task_ref] = var.asset_ref
        elif var.semantic is VariableSemantic.START_TIME and var.task_ref:
            starts[var.task_ref] = reader.Value(cp_vars[var.id]) / compiled.time_scale

    objective = realized_objective(ir, assignment)
    if compiled.has_objective:
        bound = reader.BestObjectiveBound() / compiled.objective_scale
        gap = abs(bound - objective) / max(abs(objective), 1.0)
    else:
        bound = None
        gap = None
    plan = build_plan(assignment, starts, task_kinds, durations)
    return Incumbent(objective=objective, bound=bound, gap=gap, plan=plan, status=status)
