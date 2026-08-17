# SPDX-License-Identifier: Apache-2.0
"""``verify_feasible`` — an independent structural feasibility check (RM-P1-ALLOC-01).

Allocate produces *feasible-by-construction* plans, but it is **not** the safety authority
(allocate.md §9): a plan must be re-checkable against the model by a party that does not
trust the solver — the [Guard](guard.md) shield at execution, and Bench's differential
tests. This is that re-check, and it is also the oracle the feasibility property test
asserts against (allocate.md §10: "returned plans are always feasible against the model").

Given an :class:`~astro_mine.allocate.Allocation` and the
:class:`~astro_mine.allocate.AllocationIR` it claims to solve, it maps the plan back onto IR
variable values (an assignment variable is ``1`` iff the plan schedules that task on that
asset; a start-time variable takes the task's scheduled start; a window-select variable is ``1``
for the disjoint availability window the start actually falls in) and checks, using only the
IR:

1. **assignment cover** — every required task is covered (its start variable resolves to a
   scheduled task; the cover constraint is satisfied);
2. **no time-window / precedence contradiction** — every linear constraint holds under the
   plan's variable values. A task scheduled in the **gap** between two disjoint time windows
   satisfies no window selector, so its ``twin_pick`` exactly-one fails and the plan is rejected
   (RM-P1-ALLOC-02);
3. **no double-booked resource** — every ``NO_OVERLAP``/``CUMULATIVE`` scheduling constraint holds
   (:mod:`astro_mine.allocate.model.ir.schedule`). Interval lengths are re-derived from the **IR's**
   declared sizes, *not* from the plan's own ``end_s``, so a backend that under-reports a task's end
   cannot hide an overlap;
4. **honest objective** — the reported ``realized_objective`` equals
   ``sum(coefficient * variable_value)`` over the IR objective terms.

It returns ``bool`` (never raises): ``True`` only when the plan is provably feasible and its
objective honest against *this* IR. A result with no plan is not verifiable-feasible.
"""

from __future__ import annotations

from astro_mine.allocate.api.model import Allocation
from astro_mine.allocate.enums import ConstraintSense, VariableSemantic
from astro_mine.allocate.explain.binding import plan_variable_values
from astro_mine.allocate.model.ir.model import AllocationIR
from astro_mine.allocate.model.ir.schedule import SCHEDULING_KINDS, satisfies_scheduling

__all__ = ["verify_feasible"]

#: Absolute floor for float comparisons; scaled by magnitude for the objective check.
_EPS = 1.0e-9


def _satisfies(lhs: float, sense: ConstraintSense, rhs: float) -> bool:
    if sense is ConstraintSense.LE:
        return lhs <= rhs + _EPS
    if sense is ConstraintSense.GE:
        return lhs >= rhs - _EPS
    return abs(lhs - rhs) <= _EPS


def verify_feasible(allocation: Allocation, ir: AllocationIR) -> bool:
    """Return whether ``allocation``'s plan is structurally feasible against ``ir``."""
    if allocation.plan is None:
        return False

    # Reject a task scheduled more than once (an ambiguous plan is not verifiable-feasible).
    scheduled_tasks: set[str] = set()
    for asset_schedule in allocation.plan:
        for st in asset_schedule.tasks:
            if st.task_id in scheduled_tasks:
                return False
            scheduled_tasks.add(st.task_id)

    # Map the plan onto IR variable values through the shared explain seam, so the verifier and
    # the binding/decomposition explanations never evaluate the same plan differently.
    values = plan_variable_values(allocation.plan, ir)
    for var in ir.variables:
        # A task carrying a start variable but absent from the plan → the cover is unmet.
        if (
            var.semantic is VariableSemantic.START_TIME
            and (var.task_ref or "") not in scheduled_tasks
        ):
            return False

    # Every referenced variable now has a value, and the IR's referential-integrity validator
    # guarantees each term references a declared variable — so indexing is total.
    for constraint in ir.constraints:
        if constraint.kind in SCHEDULING_KINDS:
            # A scheduling family carries interval sizes, not a linear left-hand side: evaluating it
            # as `sum(size * start) <= 0` would be meaningless. Its meaning lives in one place
            # (`model/ir/schedule.py`) that the CP-SAT lowering and the explanation share.
            if not satisfies_scheduling(constraint, ir, values):
                return False
            continue
        lhs = sum(term.coefficient * values[term.var_ref] for term in constraint.terms)
        if not _satisfies(lhs, constraint.sense, constraint.rhs):
            return False

    if allocation.realized_objective is None:
        return False
    expected = sum(term.coefficient * values[term.var_ref] for term in ir.objective_terms)
    tolerance = _EPS * max(1.0, abs(expected), abs(allocation.realized_objective))
    return abs(expected - allocation.realized_objective) <= tolerance
