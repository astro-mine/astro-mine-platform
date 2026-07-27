"""Binding-constraint reporting over the solver-neutral IR (RM-P1-ALLOC-06).

*Which* constraint bound the result — the comms window, power floor, terrain keep-out,
time window, precedence, or the **saturated asset** that had no idle time left — at the returned
plan (allocate.md §9/§10; ``LUNAR-UX-004``). Computed purely over the solver-neutral
:class:`~astro_mine.allocate.AllocationIR` — no backend-specific leakage into the contract —
by mapping the plan back onto IR variable values (:func:`plan_variable_values`, the *same*
mapping the independent feasibility verifier uses, so an explanation never disagrees with
the shield) and evaluating each constraint's slack.

A constraint is **binding** when its slack is within a float epsilon of zero. The linear families
take their slack from the residual toward their bound; the ``NO_OVERLAP``/``CUMULATIVE`` scheduling
families take theirs from :mod:`astro_mine.allocate.model.ir.schedule` (the tightest idle gap on the
resource / its spare capacity at the busiest instant) — the same module the verifier checks them
with. The purely structural equalities are excluded — the ``ASSIGNMENT_COVER`` exactly-one and the
``twin_pick`` window selector are tight by construction and carry no "what limited the plan"
information; every other family (power budget, comms/time window, precedence, terrain keep-out,
resource contention) is reported with a ``source`` tracing it back to the upstream truth that
produced it (Link / Worlds / Fleet / the schedule / the request itself).
"""

from __future__ import annotations

from astro_mine.allocate.api.model import AssetSchedule, BindingConstraint
from astro_mine.allocate.enums import ConstraintKind, ConstraintSense, VariableSemantic
from astro_mine.allocate.model.ir.model import AllocationIR
from astro_mine.allocate.model.ir.schedule import SCHEDULING_KINDS, scheduling_slack
from astro_mine.allocate.model.ir.utils import window_choices

__all__ = ["binding_constraints", "binding_source", "plan_variable_values"]

#: Absolute floor for the tightness comparison — matches the IR verifier's epsilon.
_EPS = 1.0e-9

#: The upstream truth each IR constraint-id prefix traces back to (``LUNAR-UX-004``). The
#: ``::``-delimited id conventions are owned by the RM-P1-ALLOC-01 compiler and the
#: RM-P1-ALLOC-03 builders.
_SOURCE_BY_PREFIX: dict[str, str] = {
    "keepout": "worlds:traversability",
    "power": "fleet:power-budget",
    "comms_lo": "link:contact-window",
    "comms_hi": "link:contact-window",
    "twin_lo": "request:time-window",
    "twin_hi": "request:time-window",
    "twin_pick": "request:time-window",
    "prec": "request:precedence",
    "cover": "assignment:cover",
    "no_overlap": "schedule:resource",
    "cumulative": "schedule:resource",
}


def plan_variable_values(plan: list[AssetSchedule], ir: AllocationIR) -> dict[str, float]:
    """Map a plan's per-asset schedule back onto IR decision-variable values.

    An ``ASSIGNMENT`` variable is ``1.0`` iff the plan schedules its task on its asset, else
    ``0.0``; a ``START_TIME`` variable takes the task's scheduled start (``0.0`` when the task is
    absent from the plan); a ``WINDOW_SELECT`` variable is ``1.0`` for the **first** of its task's
    disjoint availability windows that actually contains the scheduled start, else ``0.0``.

    That last rule is what makes a gap detectable: a start placed *between* two disjoint windows
    satisfies no selector, so every selector is ``0``, the task's ``twin_pick`` exactly-one
    (``sum(y_k) == 1``) fails, and :func:`~astro_mine.allocate.verify_feasible` rejects the plan.
    Selecting the first containing window (never several) also keeps the mapping single-valued, so
    the ``twin_lo``/``twin_hi`` bounds evaluate against exactly one window.

    This is the one variable-value mapping the feasibility verifier
    (:func:`~astro_mine.allocate.verify_feasible`), the binding-constraint slack, and the objective
    decomposition all evaluate through — so the three never disagree, and the decomposition sums to
    the verified realized objective by construction.
    """
    scheduled: dict[str, tuple[str, float]] = {}
    for asset_schedule in plan:
        for st in asset_schedule.tasks:
            scheduled[st.task_id] = (asset_schedule.asset_id, st.start_s)

    choices = window_choices(ir)
    selected: dict[str, str] = {}
    for task_id, windows in choices.items():
        placed = scheduled.get(task_id)
        if placed is None:
            continue
        hosting = next((w for w in windows if w.contains(placed[1])), None)
        if hosting is not None:
            selected[task_id] = hosting.var_id

    values: dict[str, float] = {}
    for var in ir.variables:
        placed = scheduled.get(var.task_ref or "")
        if var.semantic is VariableSemantic.ASSIGNMENT:
            values[var.id] = 1.0 if placed is not None and placed[0] == var.asset_ref else 0.0
        elif var.semantic is VariableSemantic.START_TIME:
            values[var.id] = placed[1] if placed is not None else 0.0
        elif var.semantic is VariableSemantic.WINDOW_SELECT:
            values[var.id] = 1.0 if selected.get(var.task_ref or "") == var.id else 0.0
    return values


def binding_source(constraint_id: str) -> str | None:
    """The upstream source a constraint traces back to, from its id prefix (``LUNAR-UX-004``)."""
    return _SOURCE_BY_PREFIX.get(constraint_id.split("::", 1)[0])


def _slack(lhs: float, sense: ConstraintSense, rhs: float) -> float:
    """The constraint's residual toward its bound (``0`` when tight, ``> 0`` with room to spare)."""
    if sense is ConstraintSense.LE:
        return rhs - lhs
    if sense is ConstraintSense.GE:
        return lhs - rhs
    return abs(lhs - rhs)


def _is_structural(constraint_id: str, kind: ConstraintKind) -> bool:
    """Whether a constraint is tight purely by construction, so reporting it explains nothing.

    The ``ASSIGNMENT_COVER`` exactly-one ("some asset does this task") and the ``twin_pick`` window
    selector ("the task runs in one of its windows") are satisfied with zero slack by *every*
    feasible plan — they never tell an operator what limited the result.
    """
    return kind is ConstraintKind.ASSIGNMENT_COVER or constraint_id.startswith("twin_pick::")


def binding_constraints(plan: list[AssetSchedule], ir: AllocationIR) -> list[BindingConstraint]:
    """The constraints tight at ``plan`` — the plan's binding-constraint explanation.

    Every non-structural IR constraint whose slack is within an epsilon of zero, each carrying its
    :class:`~astro_mine.allocate.enums.ConstraintKind` and a ``source`` tracing it to its upstream
    truth. Returned sorted by ``constraint_id`` so the explanation is deterministic (allocate.md
    §8).
    """
    values = plan_variable_values(plan, ir)
    bindings: list[BindingConstraint] = []
    for constraint in ir.constraints:
        if _is_structural(constraint.id, constraint.kind):
            continue
        if constraint.kind in SCHEDULING_KINDS:
            # A saturated resource (an asset with no idle time left between two of its tasks) *is*
            # the thing that bound the plan — but its slack is an interval gap, not a linear
            # residual, so it comes from the shared scheduling semantics the verifier uses.
            slack = scheduling_slack(constraint, ir, values)
            if slack is None or abs(slack) > _EPS:
                continue
        else:
            lhs = sum(term.coefficient * values[term.var_ref] for term in constraint.terms)
            if abs(_slack(lhs, constraint.sense, constraint.rhs)) > _EPS:
                continue
        bindings.append(
            BindingConstraint(
                constraint_id=constraint.id,
                kind=constraint.kind,
                slack=0.0,
                source=binding_source(constraint.id),
            )
        )
    bindings.sort(key=lambda b: b.constraint_id)
    return bindings
