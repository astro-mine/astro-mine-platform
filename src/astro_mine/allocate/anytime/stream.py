# SPDX-License-Identifier: Apache-2.0
"""The anytime incumbent stream + honest deadline status (RM-P1-ALLOC-05).

Layers the anytime contract over the :class:`~astro_mine.allocate.solvers.base.Solver` strategy
stream (allocate.md §2/§3): :func:`stream_incumbents` wraps a backend's incumbent stream with a
:class:`~astro_mine.allocate.anytime.bounds.BoundTracker` so the exposed bounds are monotone and
each gap is recomputed; :func:`finalize_status` maps a terminal solve verdict onto the honest
anytime outcome.

Deadline behavior (issue #5): at the budget, the best *feasible* incumbent found is returned with
its explicit gap (``FEASIBLE``), a proven optimum is ``OPTIMAL`` (gap 0), a proven-infeasible
model is ``INFEASIBLE``; a search that ran out of budget with no incumbent and no infeasibility
proof is an honest ``TIMEOUT`` — a late optimal answer is a wrong answer in operations (principle
1). The in-process iterator *is* the anytime contract (a server-streaming gRPC face is a
deployment of the same library; conventions.md §1.4).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from astro_mine.allocate.anytime.bounds import BoundTracker
from astro_mine.allocate.api.model import SolveBudget
from astro_mine.allocate.enums import AllocationStatus, ObjectiveSense
from astro_mine.allocate.model.ir.model import AllocationIR
from astro_mine.allocate.solvers.base import Incumbent, Solver

__all__ = ["finalize_status", "stream_incumbents"]

#: The honest anytime outcome for each terminal solver verdict: a searched-but-unresolved solve
#: (``UNKNOWN``) becomes an explicit ``TIMEOUT`` rather than a silent unknown.
_ANYTIME_FINAL: dict[AllocationStatus, AllocationStatus] = {
    AllocationStatus.OPTIMAL: AllocationStatus.OPTIMAL,
    AllocationStatus.FEASIBLE: AllocationStatus.FEASIBLE,
    AllocationStatus.INFEASIBLE: AllocationStatus.INFEASIBLE,
    AllocationStatus.UNKNOWN: AllocationStatus.TIMEOUT,
    AllocationStatus.TIMEOUT: AllocationStatus.TIMEOUT,
}


def stream_incumbents(
    solver: Solver,
    ir: AllocationIR,
    budget: SolveBudget,
    sense: ObjectiveSense,
    *,
    hints: Mapping[str, float] | None = None,
) -> Iterator[Incumbent]:
    """Stream ``solver``'s incumbents with monotone bounds and recomputed gaps.

    Wraps :meth:`Solver.solve` with a :class:`BoundTracker` for the objective ``sense`` so the
    exposed bound trajectory is provably monotone even if the backend reports its raw dual bound
    non-monotonically. ``hints`` is passed straight through to the backend's warm-start seam.
    """
    tracker = BoundTracker(sense)
    for incumbent in solver.solve(ir, budget, hints=hints):
        yield tracker.track(incumbent)


def finalize_status(status: AllocationStatus) -> AllocationStatus:
    """Map a terminal solver verdict onto the honest anytime outcome (``UNKNOWN`` → ``TIMEOUT``)."""
    return _ANYTIME_FINAL[status]
