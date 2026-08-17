# SPDX-License-Identifier: Apache-2.0
"""``lower_to_cpsat`` — pure Allocation IR → CP-SAT model lowering (RM-P1-ALLOC-02).

The IR → CP-SAT compiler (allocate.md §3, ``model/compile/``): a **pure function** of the
solver-neutral :class:`~astro_mine.allocate.AllocationIR` that builds an OR-Tools
``cp_model.CpModel`` plus the variable map a search driver reads a plan back through. It is
decoupled from the search itself (:mod:`astro_mine.allocate.solvers.cpsat`) so the encoding
is independently unit-testable and never entangled with anytime/streaming concerns.

**Encoding (allocate.md §4/§11):**

- ``BINARY`` → :meth:`NewBoolVar`; ``INTEGER`` → :meth:`NewIntVar` over its integer bounds.
- ``CONTINUOUS`` start-time → an **integer** time variable on a fixed grid (CP-SAT is
  integral): the value is ``seconds * TIME_SCALE``. Bounds are tightened conservatively
  (lower ``ceil``, upper ``floor``) so a solved start always satisfies the real-valued IR
  bound; an open horizon is clamped to a finite grid.
- ``ASSIGNMENT_COVER`` (all-unit ``== 1``) → :meth:`AddExactlyOne`; any other linear family
  (``TIME_WINDOW`` / ``PRECEDENCE`` / ``LINEAR`` keep-out / power budget) → a generic
  ``Add(weighted_sum <sense> rhs)``. A constraint that touches a time variable is lowered **in
  time units**: the time variable already holds ``seconds * TIME_SCALE``, so the coefficients of
  its 0/1 companions (the ``WINDOW_SELECT`` selectors of a disjoint-window disjunction, whose
  coefficients are window bounds in *seconds*) and the right-hand side are scaled to match. A
  constraint over 0/1 assignment variables alone (cover, keep-out, power budget) is exact.
- ``NO_OVERLAP`` / ``CUMULATIVE`` → :meth:`AddNoOverlap` / :meth:`AddCumulative` over
  **optional interval** variables channelled to the assignment variables (an interval is
  present iff its task is assigned to the constraint's resource). Interval sizes are read
  from the constraint's term coefficients — the per-asset "one task at a time" constraints the
  compiler emits from a task's duration (RM-P1-ALLOC-02); their semantics are defined once, in
  :mod:`astro_mine.allocate.model.ir.schedule`, which the independent verifier re-checks them with.
- Objective → :meth:`Maximize` / :meth:`Minimize` of the scaled objective terms. The
  **realized** objective a plan reports is recomputed from the float IR terms by the driver
  (never from this scaled integer form), so it never disagrees with
  :func:`~astro_mine.allocate.verify_feasible`.

The lowering itself asserts no feasibility: it is the driver that solves, and the IR that
verifies. ``TIME_SCALE`` / ``OBJECTIVE_SCALE`` are fixed backend constants — pinned (with the
OR-Tools version) by the plan's ``provenance.backend_version`` — so a seeded solve is
byte-reproducible (allocate.md §8; RM-P1-ALLOC-07).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ortools.sat.python import cp_model

from astro_mine.allocate.enums import ConstraintKind, ConstraintSense, ObjectiveSense, VariableKind
from astro_mine.allocate.model.ir.compile import (
    assignment_var_id,
    cumulative_constraint_id,
    no_overlap_constraint_id,
)
from astro_mine.allocate.model.ir.model import AllocationIR, Constraint, DecisionVariable
from astro_mine.allocate.model.ir.schedule import SCHEDULING_KINDS

if TYPE_CHECKING:
    from collections.abc import Collection, Mapping

__all__ = [
    "OBJECTIVE_SCALE",
    "TIME_SCALE",
    "CpSatIisModel",
    "CpSatModel",
    "cumulative_constraint_id",
    "lower_for_iis",
    "lower_to_cpsat",
    "no_overlap_constraint_id",
]

#: Time grid: one integer CP-SAT time unit = ``1 / TIME_SCALE`` seconds. ``1`` keeps the model
#: on an integer-second grid (the IR the RM-P1-ALLOC-03 builders emit is integer-second), which
#: keeps variable domains small and a solved start exactly re-checkable against the float IR.
TIME_SCALE: int = 1

#: Objective coefficients (task value / resource ROI / info-gain EVPI) are floats; CP-SAT
#: optimizes an integer objective, so they are scaled by this factor and rounded. The reported
#: objective is recomputed from the *float* IR terms, so this scale only steers the search.
OBJECTIVE_SCALE: int = 1_000_000

#: A finite stand-in for the IR's open start horizon (``compile.py`` uses ``1e12`` s). Clamping
#: the *upper* bound down is feasibility-safe — a start stays within the real interval — and keeps
#: CP-SAT domains sane. ``1e9`` s (~31 years) dwarfs any realistic episode horizon.
_MAX_HORIZON_S: float = 1.0e9

#: The IR's open-horizon sentinel (mirrors ``compile._UNBOUNDED_HORIZON_S``); an upper bound at or
#: above it is treated as "unbounded" and clamped to ``_MAX_HORIZON_S``.
_OPEN_HORIZON_S: float = 1.0e12


@dataclass(frozen=True, slots=True)
class CpSatModel:
    """A lowered CP-SAT model plus the seam a driver reads a plan back through.

    ``variables`` maps each IR decision-variable id to its CP-SAT variable (an assignment
    ``BoolVar`` or an integer time variable), so a solved value maps straight back onto the IR
    for plan assembly and independent verification. ``has_objective`` records whether an
    optimization objective was set (an IR with no objective terms is a pure feasibility model).
    """

    model: Any  # cp_model.CpModel (ortools ships no complete stubs)
    variables: dict[str, Any] = field(default_factory=dict)
    time_scale: int = TIME_SCALE
    objective_scale: int = OBJECTIVE_SCALE
    has_objective: bool = False


def _time_var_bounds(var: DecisionVariable, time_scale: int) -> tuple[int, int]:
    """Integer ``[lo, hi]`` grid bounds for a continuous start-time variable (conservative).

    The lower bound rounds **up** and the upper bound rounds **down** so any grid point in the
    integer domain also satisfies the real-valued IR bound; an open upper horizon is clamped to a
    finite grid.
    """
    lo_s = var.lower if var.lower is not None else 0.0
    hi_s = var.upper if var.upper is not None else _MAX_HORIZON_S
    if hi_s >= _OPEN_HORIZON_S:
        hi_s = _MAX_HORIZON_S
    lo = math.ceil(lo_s * time_scale)
    hi = math.floor(hi_s * time_scale)
    if hi < lo:
        hi = lo  # an empty sub-grid interval degenerates to a single point (still bounded)
    return lo, hi


def _make_variable(model: Any, var: DecisionVariable, time_scale: int) -> Any:
    if var.kind is VariableKind.BINARY:
        return model.NewBoolVar(var.id)
    if var.kind is VariableKind.CONTINUOUS:
        lo, hi = _time_var_bounds(var, time_scale)
        return model.NewIntVar(lo, hi, var.id)
    if var.kind is VariableKind.INTEGER:
        lo = math.ceil(var.lower) if var.lower is not None else -cp_model.INT32_MAX
        hi = math.floor(var.upper) if var.upper is not None else cp_model.INT32_MAX
        return model.NewIntVar(lo, hi, var.id)
    # INTERVAL variables are materialized by the NO_OVERLAP / CUMULATIVE lowering, never as a
    # standalone decision variable, so a bare INTERVAL variable is not a valid IR at v0.1.0.
    raise NotImplementedError(f"unsupported IR variable kind for CP-SAT: {var.kind}")


def _round_rhs(rhs: float, sense: ConstraintSense) -> int:
    """Round a (already scaled) right-hand side conservatively for its sense.

    ``LE`` floors and ``GE`` ceils so the integer constraint is no looser than the real one;
    ``EQ`` rounds to nearest (the equalities the IR emits — assignment cover, keep-out — carry
    integer right-hand sides, so this is exact).
    """
    if sense is ConstraintSense.LE:
        return math.floor(rhs)
    if sense is ConstraintSense.GE:
        return math.ceil(rhs)
    return round(rhs)


def _add_linear(
    model: Any,
    constraint: Constraint,
    cp_vars: dict[str, Any],
    semantics: Mapping[str, VariableKind],
    time_scale: int,
    *,
    enforce: Any | None = None,
) -> None:
    """Lower one linear IR constraint to ``Add(weighted_sum <sense> rhs)``.

    A constraint that touches a continuous start-time variable is lowered **in time units**: that
    variable already holds ``seconds * TIME_SCALE``, so every *other* term's coefficient (and the
    right-hand side) is scaled by ``TIME_SCALE`` to match. That is what makes the disjoint-window
    disjunction exact at any time scale — ``start - sum(lo_k * y_k) >= 0`` mixes a scaled time
    variable with 0/1 selectors whose coefficients are window bounds in *seconds*. A constraint over
    0/1 assignment variables alone (cover, keep-out, power budget) is untouched and exact.
    ``ASSIGNMENT_COVER`` collapses to :meth:`AddExactlyOne` when it is the canonical unit ``== 1``
    cover.

    When an ``enforce`` literal is supplied, the constraint is reified with
    :meth:`OnlyEnforceIf` (and the :meth:`AddExactlyOne` shortcut is skipped, since an enforced
    generic linear form is what the IIS assumption machinery relaxes) — the RM-P1-ALLOC-06 seam.
    """
    over_time = any(semantics[t.var_ref] is VariableKind.CONTINUOUS for t in constraint.terms)
    scale = time_scale if over_time else 1

    if (
        enforce is None
        and constraint.kind is ConstraintKind.ASSIGNMENT_COVER
        and constraint.sense is ConstraintSense.EQ
        and constraint.terms
        and round(constraint.rhs) == 1
        and all(t.coefficient == 1.0 for t in constraint.terms)
    ):
        model.AddExactlyOne(cp_vars[t.var_ref] for t in constraint.terms)
        return

    variables = [cp_vars[t.var_ref] for t in constraint.terms]
    coeffs = [
        # A continuous term's variable already carries the scale; every other term's coefficient
        # must be lifted into the same units.
        round(t.coefficient * (1 if semantics[t.var_ref] is VariableKind.CONTINUOUS else scale))
        for t in constraint.terms
    ]
    lhs = cp_model.LinearExpr.weighted_sum(variables, coeffs)
    rhs = _round_rhs(constraint.rhs * scale, constraint.sense)
    if constraint.sense is ConstraintSense.LE:
        ct = model.Add(lhs <= rhs)
    elif constraint.sense is ConstraintSense.GE:
        ct = model.Add(lhs >= rhs)
    else:
        ct = model.Add(lhs == rhs)
    if enforce is not None:
        ct.OnlyEnforceIf(enforce)


def _optional_interval(
    model: Any,
    constraint: Constraint,
    cp_vars: dict[str, Any],
    var_by_id: dict[str, DecisionVariable],
    time_scale: int,
    *,
    enforce: Any | None = None,
) -> list[Any]:
    """Build the (optionally channelled) interval variables of a scheduling constraint.

    The resource is the id's trailing segment (``no_overlap::{resource}``); each term references a
    start-time variable and carries the interval size (duration, s) as its coefficient. The
    interval is **optional**, present iff the task's ``assign::{task}::{resource}`` variable is 1,
    channelling the schedule to the assignment — or mandatory when no such assignment variable
    exists in the IR.

    Sizes round **up**, like every other bound this module lowers (``ceil`` a lower bound, ``floor``
    an upper one): a float duration of ``1799.1`` s must reserve ``1800`` grid seconds, because an
    integer interval *shorter* than the real one would let CP-SAT butt the next task against an end
    the float IR has not reached yet — a plan that looks conflict-free on the grid and double-books
    the asset by a fraction of a second against
    :func:`~astro_mine.allocate.verify_feasible`, which re-checks it in floats. Rounding up can only
    ever make the integer model *stricter* than the IR, never looser.

    ``enforce`` reifies the whole constraint for the IIS model (RM-P1-ALLOC-06): a scheduling global
    cannot take :meth:`OnlyEnforceIf`, so instead each interval's presence literal becomes
    ``assigned AND enforce``. With ``enforce`` false no interval is present and the global is
    vacuous — exactly the relaxation an assumption literal must express.
    """
    resource = constraint.id.split("::", 1)[1]
    intervals: list[Any] = []
    for term in constraint.terms:
        start = cp_vars[term.var_ref]
        size = math.ceil(term.coefficient * time_scale)
        task = var_by_id[term.var_ref].task_ref or term.var_ref
        presence = cp_vars.get(assignment_var_id(task, resource))
        name = f"iv::{constraint.id}::{task}"
        if enforce is not None:
            presence = _conjunction(model, presence, enforce, f"present::{constraint.id}::{task}")
        if presence is not None:
            intervals.append(model.NewOptionalFixedSizeIntervalVar(start, size, presence, name))
        else:
            intervals.append(model.NewFixedSizeIntervalVar(start, size, name))
    return intervals


def _conjunction(model: Any, left: Any | None, right: Any, name: str) -> Any:
    """A fresh Bool literal constrained to ``left AND right`` (just ``right`` when ``left`` is
    absent)."""
    if left is None:
        return right
    both = model.NewBoolVar(name)
    model.AddBoolAnd([left, right]).OnlyEnforceIf(both)
    model.AddBoolOr([left.Not(), right.Not(), both])
    return both


def lower_to_cpsat(
    ir: AllocationIR,
    *,
    time_scale: int = TIME_SCALE,
    objective_scale: int = OBJECTIVE_SCALE,
) -> CpSatModel:
    """Lower an :class:`~astro_mine.allocate.AllocationIR` to a CP-SAT model (pure)."""
    model = cp_model.CpModel()
    var_by_id = {v.id: v for v in ir.variables}
    semantics = {v.id: v.kind for v in ir.variables}
    cp_vars: dict[str, Any] = {v.id: _make_variable(model, v, time_scale) for v in ir.variables}

    for constraint in ir.constraints:
        if constraint.kind is ConstraintKind.NO_OVERLAP:
            model.AddNoOverlap(
                _optional_interval(model, constraint, cp_vars, var_by_id, time_scale)
            )
        elif constraint.kind is ConstraintKind.CUMULATIVE:
            intervals = _optional_interval(model, constraint, cp_vars, var_by_id, time_scale)
            model.AddCumulative(intervals, [1] * len(intervals), round(constraint.rhs))
        else:
            _add_linear(model, constraint, cp_vars, semantics, time_scale)

    has_objective = bool(ir.objective_terms)
    if has_objective:
        literals = [cp_vars[t.var_ref] for t in ir.objective_terms]
        coeffs = [round(t.coefficient * objective_scale) for t in ir.objective_terms]
        objective = cp_model.LinearExpr.weighted_sum(literals, coeffs)
        if ir.objective_sense is ObjectiveSense.MAXIMIZE:
            model.Maximize(objective)
        else:
            model.Minimize(objective)

    return CpSatModel(
        model=model,
        variables=cp_vars,
        time_scale=time_scale,
        objective_scale=objective_scale,
        has_objective=has_objective,
    )


@dataclass(frozen=True, slots=True)
class CpSatIisModel:
    """A CP-SAT model whose every relaxable IR constraint is reified for IIS extraction.

    The pure-feasibility (no-objective) lowering of an :class:`~astro_mine.allocate.AllocationIR`
    in which each constraint is enforced only when its ``relax::{constraint_id}`` Bool literal is
    true, and all such literals are added as **solver assumptions**. Given an infeasible IR,
    :meth:`CpSolver.SufficientAssumptionsForInfeasibility` then returns the literal indices of a
    conflicting subset — the irreducible infeasible set (RM-P1-ALLOC-06). ``literal_by_constraint``
    maps each IR constraint id to its literal; ``constraint_by_index`` maps the literal's proto
    variable index back to the constraint id (the key the solver reports the conflict in).
    """

    model: Any  # cp_model.CpModel
    literal_by_constraint: dict[str, Any] = field(default_factory=dict)
    constraint_by_index: dict[int, str] = field(default_factory=dict)


def _iis_variable(model: Any, var: DecisionVariable, time_scale: int) -> Any:
    """Make a decision variable for the IIS model — start-time domains are **widened**.

    Time-window enforcement is duplicated between a start-time variable's bounds and the
    ``twin_*`` / ``comms_*`` constraints; widening the variable domain here forces *all* window
    enforcement through the reified constraints, so a window conflict is attributed to a
    relaxable constraint rather than silently absorbed by a variable bound.
    """
    if var.kind is VariableKind.CONTINUOUS:
        return model.NewIntVar(0, math.floor(_MAX_HORIZON_S * time_scale), var.id)
    return _make_variable(model, var, time_scale)


def lower_for_iis(
    ir: AllocationIR,
    *,
    time_scale: int = TIME_SCALE,
    keep: Collection[str] | None = None,
) -> CpSatIisModel:
    """Lower an :class:`~astro_mine.allocate.AllocationIR` to an assumption-reified CP-SAT model.

    Every constraint is reified with a ``relax::{id}`` literal and added as a solver assumption; no
    objective is set. The **linear** families (cover, time-window, precedence, keep-out, power
    budget) reify with :meth:`OnlyEnforceIf`; the **scheduling** globals
    (``NO_OVERLAP``/``CUMULATIVE``), which cannot, reify by conjoining their intervals' presence
    literals with the relax literal — so an over-subscribed asset can appear *in* an irreducible
    infeasible set rather than being invisible to it. Constraints are lowered in ``id`` order so the
    literal indices are deterministic (allocate.md §8).

    ``keep`` restricts the model to a subset of constraint ids — the primitive the deletion filter
    (:func:`~astro_mine.allocate.explain.extract_iis`) re-solves over to prove a candidate core is
    *irreducible*. ``None`` (the default) keeps every constraint.
    """
    model = cp_model.CpModel()
    semantics = {v.id: v.kind for v in ir.variables}
    var_by_id = {v.id: v for v in ir.variables}
    cp_vars: dict[str, Any] = {v.id: _iis_variable(model, v, time_scale) for v in ir.variables}

    literal_by_constraint: dict[str, Any] = {}
    constraint_by_index: dict[int, str] = {}
    for constraint in sorted(ir.constraints, key=lambda c: c.id):
        if keep is not None and constraint.id not in keep:
            continue
        literal = model.NewBoolVar(f"relax::{constraint.id}")
        if constraint.kind in SCHEDULING_KINDS:
            intervals = _optional_interval(
                model, constraint, cp_vars, var_by_id, time_scale, enforce=literal
            )
            if constraint.kind is ConstraintKind.NO_OVERLAP:
                model.AddNoOverlap(intervals)
            else:
                model.AddCumulative(intervals, [1] * len(intervals), round(constraint.rhs))
        else:
            _add_linear(model, constraint, cp_vars, semantics, time_scale, enforce=literal)
        literal_by_constraint[constraint.id] = literal
        constraint_by_index[literal.Index()] = constraint.id

    model.AddAssumptions(list(literal_by_constraint.values()))
    return CpSatIisModel(
        model=model,
        literal_by_constraint=literal_by_constraint,
        constraint_by_index=constraint_by_index,
    )
