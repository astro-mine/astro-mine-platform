"""Semantics of the IR's **scheduling** constraint families (RM-P1-ALLOC-02).

``NO_OVERLAP`` and ``CUMULATIVE`` are the two IR families whose meaning is *not* the linear form
``sum(coefficient * variable) <sense> rhs``: each term names a task's **start** variable and carries
that task's **interval size** (SI seconds) as its coefficient, and the constraint id names the
resource the intervals contend for (``no_overlap::{asset}`` / ``cumulative::{resource}``). An
interval is **present** only when the resource actually takes the task — channelled to the
``assign::{task}::{resource}`` variable exactly as
:func:`~astro_mine.allocate.model.compile.cpsat.lower_to_cpsat` channels its optional CP-SAT
intervals.

This module is the **one** place that meaning is written down. The CP-SAT lowering encodes it, the
independent feasibility verifier (:func:`~astro_mine.allocate.verify_feasible`) re-checks it, and
the binding-constraint explanation reports its slack — all three reading *this* module, so a plan
the shield accepts, the model the solver solved, and the explanation an operator reads can never
drift apart (allocate.md §9/§10).

Feasibility is re-derived from the **IR's** declared sizes, never from the plan's own reported
``end_s``: a backend that under-reports a task's end cannot hide a double-booked asset.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping

from astro_mine.allocate.enums import ConstraintKind
from astro_mine.allocate.model.ir.compile import assignment_var_id
from astro_mine.allocate.model.ir.model import AllocationIR, Constraint, DecisionVariable

__all__ = [
    "SCHEDULING_KINDS",
    "cumulative_slack",
    "no_overlap_slack",
    "resource_intervals",
    "satisfies_scheduling",
    "scheduling_slack",
]

#: Absolute floor for the scheduling comparisons — matches the IR verifier's epsilon.
_EPS = 1.0e-9

#: The constraint kinds this module — not the generic linear evaluator — gives meaning to.
SCHEDULING_KINDS = frozenset({ConstraintKind.NO_OVERLAP, ConstraintKind.CUMULATIVE})


def resource_id(constraint: Constraint) -> str:
    """The resource a scheduling constraint contends over — the trailing segment of its id."""
    return constraint.id.split("::", 1)[1]


def resource_intervals(
    constraint: Constraint,
    ir: AllocationIR,
    values: Mapping[str, float],
) -> list[tuple[float, float]]:
    """The ``[start, end)`` intervals **present** on the constraint's resource, sorted by start.

    One interval per term whose task is actually placed on the resource: its start is the
    start-time variable's value under the plan and its length the term's coefficient. A term whose
    ``assign::{task}::{resource}`` variable is ``0`` contributes nothing (the interval is absent);
    a term with no such variable in the IR is **mandatory** and always present — the same rule the
    CP-SAT optional-interval channeling applies.
    """
    resource = resource_id(constraint)
    var_by_id: dict[str, DecisionVariable] = {v.id: v for v in ir.variables}
    intervals: list[tuple[float, float]] = []
    for term in constraint.terms:
        declared = var_by_id.get(term.var_ref)
        task = (declared.task_ref if declared is not None else None) or term.var_ref
        presence = values.get(assignment_var_id(task, resource))
        if presence is not None and presence < 0.5:
            continue  # this resource does not take this task — no interval
        start = values.get(term.var_ref, 0.0)
        intervals.append((start, start + term.coefficient))
    intervals.sort()
    return intervals


def no_overlap_slack(intervals: list[tuple[float, float]]) -> float | None:
    """The tightest idle gap between consecutive intervals on a single-capacity resource.

    ``0`` when two intervals abut exactly (the resource is *saturated* — the constraint is binding),
    positive when there is idle time to spare, and **negative** when the resource is double-booked
    (the infeasibility). ``None`` when fewer than two intervals are present: nothing can conflict,
    so the constraint is vacuous rather than binding.
    """
    if len(intervals) < 2:
        return None
    return min(later[0] - earlier[1] for earlier, later in itertools.pairwise(intervals))


def cumulative_slack(intervals: list[tuple[float, float]], capacity: float) -> float | None:
    """The resource's spare capacity at its busiest instant (``capacity - peak concurrency``).

    Each present interval consumes one unit (matching the CP-SAT lowering's unit demands), so the
    peak is the maximum number of intervals overlapping at any point. Computed by a sweep over the
    interval endpoints — ends are processed before starts at the same instant, so back-to-back
    intervals ``[0, 10)`` and ``[10, 20)`` do not contend. ``None`` when no interval is present.
    """
    if not intervals:
        return None
    events: list[tuple[float, int]] = []
    for start, end in intervals:
        events.append((start, 1))
        events.append((end, -1))
    events.sort(key=lambda e: (e[0], e[1]))  # -1 (end) sorts before +1 (start) at the same instant
    usage = 0
    peak = 0
    for _, delta in events:
        usage += delta
        peak = max(peak, usage)
    return capacity - peak


def scheduling_slack(
    constraint: Constraint, ir: AllocationIR, values: Mapping[str, float]
) -> float | None:
    """The slack of a scheduling constraint under ``values`` (``None`` when it is vacuous).

    Negative means violated; ``0`` means tight (binding); positive means room to spare. The single
    quantity both :func:`satisfies_scheduling` (the shield) and the binding-constraint explanation
    are derived from.
    """
    intervals = resource_intervals(constraint, ir, values)
    if constraint.kind is ConstraintKind.NO_OVERLAP:
        return no_overlap_slack(intervals)
    return cumulative_slack(intervals, constraint.rhs)


def satisfies_scheduling(
    constraint: Constraint, ir: AllocationIR, values: Mapping[str, float]
) -> bool:
    """Whether a scheduling constraint holds under ``values`` (a vacuous constraint holds)."""
    slack = scheduling_slack(constraint, ir, values)
    return slack is None or slack >= -_EPS
