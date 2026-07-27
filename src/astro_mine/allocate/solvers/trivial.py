"""``TrivialStubSolver`` — the OR-Tools-free ``Solver`` for the no-solver path (RM-P1-ALLOC-02).

A deterministic greedy that satisfies the :class:`~astro_mine.allocate.solvers.base.Solver`
strategy **reading only the IR** — no OR-Tools, no request, no constraint report. It exists so the
strategy interface has a backend that always works on the local tier (allocate.md §7) — the
fast-test path and the graceful fallback when the CP-SAT wheel is unavailable — proving the
registry seam is real and that a plan is decoded identically whichever backend produced it.

It is *feasibility only* (no optimality bound): it assigns each task, in precedence order, to the
first eligible, non-kept-out asset that keeps every linear budget within capacity, has idle time for
it (the ``NO_OVERLAP`` scheduling constraints — one asset does one task at a time), and can host its
start inside one of its **disjoint** availability windows, then emits one feasible
:class:`~astro_mine.allocate.solvers.base.Incumbent` (or an honest infeasible one). Both temporal
families are read straight out of the IR (:mod:`astro_mine.allocate.model.ir.schedule` /
:func:`~astro_mine.allocate.model.ir.utils.window_choices`), never out of the request — the stub
solves the same model CP-SAT does. Like every backend, its plan is still re-checked against the IR
by :func:`~astro_mine.allocate.verify_feasible` — the stub is not trusted either (allocate.md §9).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from astro_mine.allocate.api.model import SolveBudget
from astro_mine.allocate.enums import (
    AllocationStatus,
    ConstraintKind,
    ConstraintSense,
    VariableSemantic,
)
from astro_mine.allocate.model.ir.model import AllocationIR, Constraint, DecisionVariable
from astro_mine.allocate.model.ir.schedule import SCHEDULING_KINDS, resource_id
from astro_mine.allocate.model.ir.utils import WindowChoice, window_choices
from astro_mine.allocate.solvers._common import Pair, build_plan, realized_objective
from astro_mine.allocate.solvers.base import Incumbent
from astro_mine.core.messages.enums import TaskKind

__all__ = ["TrivialStubSolver"]

#: Float comparison floor, matching the IR verifier's absolute epsilon.
_EPS = 1.0e-9


class _Model:
    """The structure the greedy reads out of an IR: eligibility, keep-outs, windows (single bound
    or disjoint disjunction), precedence, per-asset budgets, and per-resource interval sizes."""

    __slots__ = (
        "budget_cap",
        "budget_cost",
        "eligible",
        "forbidden",
        "precede",
        "sizes",
        "tasks",
        "win_hi",
        "win_lo",
        "windows",
    )

    def __init__(self, ir: AllocationIR) -> None:
        var_by_id = {v.id: v for v in ir.variables}
        self.tasks: set[str] = set()
        self.eligible: dict[str, list[str]] = {}
        self.win_lo: dict[str, float] = {}
        self.win_hi: dict[str, float] = {}
        self.forbidden: set[Pair] = set()
        self.precede: dict[str, set[str]] = {}
        self.budget_cap: dict[str, float] = {}
        self.budget_cost: dict[Pair, float] = {}
        #: Per ``(task, asset)`` interval size (s), from the NO_OVERLAP/CUMULATIVE constraints.
        self.sizes: dict[Pair, float] = {}
        #: Disjoint alternative windows per task (absent ⇒ a single bound in win_lo/win_hi).
        self.windows: dict[str, list[WindowChoice]] = window_choices(ir)

        for var in ir.variables:
            if var.semantic is VariableSemantic.ASSIGNMENT and var.task_ref and var.asset_ref:
                self.eligible.setdefault(var.task_ref, []).append(var.asset_ref)
            elif var.semantic is VariableSemantic.START_TIME and var.task_ref:
                # Every task carries a start variable (compile.py), whether or not any asset is
                # eligible — so a task no asset can do is *seen* here and surfaces as infeasible.
                self.tasks.add(var.task_ref)
                self.win_lo[var.task_ref] = var.lower if var.lower is not None else 0.0
                if var.upper is not None:
                    self.win_hi[var.task_ref] = var.upper
        for assets in self.eligible.values():
            assets.sort()

        for constraint in ir.constraints:
            self._ingest(constraint, var_by_id)

    def place(self, task: str, not_before: float) -> float | None:
        """The earliest start at or after ``not_before`` this task's window(s) can host.

        With a plural-window disjunction the start is *snapped forward out of an availability gap*
        into the next window that can still take it; with a single window it is simply clamped into
        the bound. ``None`` when no window can host it (infeasible at or after ``not_before``).
        """
        alternatives = self.windows.get(task)
        if alternatives:
            for window in alternatives:
                candidate = max(not_before, window.start_s)
                if candidate <= window.end_s + _EPS:
                    return candidate
            return None
        start = max(not_before, self.win_lo.get(task, 0.0))
        hi = self.win_hi.get(task)
        if hi is not None and start > hi + _EPS:
            return None
        return start

    def _ingest(self, c: Constraint, var_by_id: Mapping[str, DecisionVariable]) -> None:
        # A scheduling global carries the resource's per-task interval sizes as its coefficients.
        if c.kind in SCHEDULING_KINDS:
            resource = resource_id(c)
            for term in c.terms:
                scheduled = _task_of(var_by_id, term.var_ref)
                if scheduled is not None:
                    self.sizes[(scheduled, resource)] = term.coefficient
            return

        # A window disjunction is already resolved into `self.windows`; its mixed-term twin_* /
        # twin_pick constraints must not be mistaken for a plain window bound or a budget.
        if any(_semantic(var_by_id, t.var_ref) is VariableSemantic.WINDOW_SELECT for t in c.terms):
            return

        # A single assignment-var `== 0` / `<= 0` term is a keep-out (assign(task, asset) pinned 0).
        if (
            len(c.terms) == 1
            and c.sense in (ConstraintSense.EQ, ConstraintSense.LE)
            and abs(c.rhs) <= _EPS
        ):
            var = var_by_id.get(c.terms[0].var_ref)
            task_ref = getattr(var, "task_ref", None)
            asset_ref = getattr(var, "asset_ref", None)
            semantic = getattr(var, "semantic", None)
            if semantic is VariableSemantic.ASSIGNMENT and task_ref and asset_ref:
                self.forbidden.add((task_ref, asset_ref))
                return

        start_terms = [
            t for t in c.terms if _semantic(var_by_id, t.var_ref) is VariableSemantic.START_TIME
        ]
        # A single start-time term narrows the task's start window (time / comms-window bounds).
        if len(c.terms) == 1 and len(start_terms) == 1:
            task = _task_of(var_by_id, c.terms[0].var_ref)
            if task is not None:
                if c.sense is ConstraintSense.GE:
                    self.win_lo[task] = max(self.win_lo.get(task, 0.0), c.rhs)
                elif c.sense is ConstraintSense.LE:
                    self.win_hi[task] = min(self.win_hi.get(task, c.rhs), c.rhs)
            return

        # A `child - pred >= 0` pair over two start-time vars is a precedence edge.
        if c.kind is ConstraintKind.PRECEDENCE and len(c.terms) == 2:
            child = next((t.var_ref for t in c.terms if t.coefficient > 0), None)
            pred = next((t.var_ref for t in c.terms if t.coefficient < 0), None)
            child_task, pred_task = _task_of(var_by_id, child), _task_of(var_by_id, pred)
            if child_task is not None and pred_task is not None:
                self.precede.setdefault(child_task, set()).add(pred_task)
            return

        # A multi-term `<=` over one asset's assignment vars is that asset's linear (energy) budget.
        if c.sense is ConstraintSense.LE and len(c.terms) >= 1:
            asset = _common_asset(var_by_id, c)
            if asset is not None:
                self.budget_cap[asset] = c.rhs
                for term in c.terms:
                    task = _task_of(var_by_id, term.var_ref)
                    if task is not None:
                        self.budget_cost[(task, asset)] = term.coefficient


def _semantic(var_by_id: Mapping[str, DecisionVariable], var_id: str) -> VariableSemantic | None:
    return getattr(var_by_id.get(var_id), "semantic", None)


def _task_of(var_by_id: Mapping[str, DecisionVariable], var_id: str | None) -> str | None:
    if var_id is None:
        return None
    return getattr(var_by_id.get(var_id), "task_ref", None)


def _common_asset(var_by_id: Mapping[str, DecisionVariable], c: Constraint) -> str | None:
    """The single asset every assignment term of ``c`` shares, or ``None`` if not uniform."""
    assets = {
        getattr(var_by_id.get(t.var_ref), "asset_ref", None)
        for t in c.terms
        if _semantic(var_by_id, t.var_ref) is VariableSemantic.ASSIGNMENT
    }
    assets.discard(None)
    if len(assets) == 1 and len(assets) == len(c.terms):
        return next(iter(assets))
    return None


def _topological_order(model: _Model) -> list[str]:
    """Task ids in a precedence-respecting order (ties by id), over every task in the model."""
    tasks = set(model.tasks)
    preds = {t: {p for p in model.precede.get(t, set()) if p in tasks} for t in tasks}
    order: list[str] = []
    ready = sorted(t for t, ps in preds.items() if not ps)
    while ready:
        tid = ready.pop(0)
        order.append(tid)
        for other, ps in preds.items():
            if tid in ps:
                ps.discard(tid)
                if not ps and other not in order and other not in ready:
                    ready.append(other)
        ready.sort()
    return order


class TrivialStubSolver:
    """A no-dependency greedy :class:`~astro_mine.allocate.solvers.base.Solver` over the IR."""

    def __init__(
        self,
        *,
        task_kinds: Mapping[str, TaskKind],
        durations: Mapping[Pair, float] | None = None,
    ) -> None:
        self._task_kinds = dict(task_kinds)
        self._durations = dict(durations or {})

    def solve(
        self,
        ir: AllocationIR,
        budget: SolveBudget,
        *,
        hints: Mapping[str, float] | None = None,
    ) -> Iterator[Incumbent]:
        """Greedily assign + schedule from the IR, yielding one feasible-or-infeasible incumbent."""
        model = _Model(ir)
        assignment: dict[str, str] = {}
        starts: dict[str, float] = {}
        used: dict[str, float] = {}
        busy_until: dict[str, float] = {}

        for task in _topological_order(model):
            not_before = max(
                [0.0, *(starts[p] for p in model.precede.get(task, set()) if p in starts)]
            )
            placed = self._choose(model, task, not_before, used, busy_until)
            if placed is None:
                return iter([_infeasible()])
            chosen, start = placed
            assignment[task] = chosen
            starts[task] = start
            used[chosen] = used.get(chosen, 0.0) + model.budget_cost.get((task, chosen), 0.0)
            busy_until[chosen] = start + model.sizes.get((task, chosen), 0.0)

        plan = build_plan(assignment, starts, self._task_kinds, self._durations)
        incumbent = Incumbent(
            objective=realized_objective(ir, assignment),
            bound=None,
            gap=None,
            plan=plan,
            status=AllocationStatus.FEASIBLE,
        )
        return iter([incumbent])

    @staticmethod
    def _choose(
        model: _Model,
        task: str,
        not_before: float,
        used: dict[str, float],
        busy_until: dict[str, float],
    ) -> tuple[str, float] | None:
        """The first eligible asset that can take ``task``, and the start it can take it at.

        An asset is a candidate when it is not kept out, its linear (energy) budget still admits the
        task, and — once the start is pushed past whatever that asset is still busy with (the
        ``NO_OVERLAP`` scheduling constraint) — the task's window(s) can still host it.
        """
        for asset in model.eligible.get(task, []):
            if (task, asset) in model.forbidden:
                continue
            cap = model.budget_cap.get(asset)
            cost = model.budget_cost.get((task, asset), 0.0)
            if cap is not None and used.get(asset, 0.0) + cost > cap + _EPS:
                continue
            start = model.place(task, max(not_before, busy_until.get(asset, 0.0)))
            if start is None:
                continue  # this asset is busy past every window that could host the task
            return asset, start
        return None


def _infeasible() -> Incumbent:
    return Incumbent(
        objective=0.0,
        bound=None,
        gap=None,
        plan=[],
        status=AllocationStatus.INFEASIBLE,
        is_feasible=False,
    )
