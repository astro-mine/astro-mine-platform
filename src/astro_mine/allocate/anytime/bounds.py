"""Monotone bound tracking for the anytime incumbent stream (RM-P1-ALLOC-05).

Anytime by default (allocate.md §2 principle 2): every solve exposes incumbents with
**monotonically improving** dual bounds and an explicit optimality gap, so a caller can stop at
any deadline and take the best plan with a correct gap. A raw CP-SAT
:meth:`BestObjectiveBound` can move non-monotonically across solution callbacks (the search
tightens and loosens its bound as it explores); :class:`BoundTracker` clamps each incumbent's
bound to the best proven so far — non-increasing for an upper bound on a MAXIMIZE objective,
non-decreasing for a lower bound on a MINIMIZE objective — so the **exposed** stream is provably
monotone regardless of the backend's internal reporting, and recomputes the gap against the
clamped bound.
"""

from __future__ import annotations

from dataclasses import replace

from astro_mine.allocate.enums import ObjectiveSense
from astro_mine.allocate.solvers.base import Incumbent

__all__ = ["BoundTracker"]


class BoundTracker:
    """Clamps each incumbent's dual bound monotone and recomputes its optimality gap.

    Stateful across one solve: :meth:`track` returns the incumbent with its ``bound`` replaced by
    the tightest bound proven up to that point and its ``gap`` recomputed. A feasibility-only
    incumbent (no bound) passes through unchanged.
    """

    def __init__(self, sense: ObjectiveSense) -> None:
        self._sense = sense
        self._best_bound: float | None = None

    def track(self, incumbent: Incumbent) -> Incumbent:
        """Return ``incumbent`` with a monotone bound and a gap recomputed against it."""
        if incumbent.bound is None:
            return incumbent
        bound = incumbent.bound
        if self._best_bound is not None:
            if self._sense is ObjectiveSense.MAXIMIZE:
                # An upper bound on the optimum tightens downward as the search proves more.
                bound = min(self._best_bound, bound)
            else:
                # A lower bound on the optimum tightens upward.
                bound = max(self._best_bound, bound)
        self._best_bound = bound
        gap = abs(bound - incumbent.objective) / max(abs(incumbent.objective), 1.0)
        return replace(incumbent, bound=bound, gap=gap)
