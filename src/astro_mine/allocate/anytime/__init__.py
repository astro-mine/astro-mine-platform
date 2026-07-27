"""Anytime contract — streaming incumbents, monotone bounds, deadline management (RM-P1-ALLOC-05).

Anytime by default (allocate.md §2 principle 2): every solve exposes incumbent solutions with
monotonically improving bounds and an explicit optimality gap, so a caller (online re-solve driven
through [Mind](mind.md), later [Ops](ops.md)) can stop at any deadline and take the best *feasible*
plan found — "a late optimal answer is a wrong answer in operations" (principle 1). Layered over
the :class:`~astro_mine.allocate.solvers.base.Solver` strategy stream, exposed through the Core
allocation sub-interface (:meth:`~astro_mine.allocate.AllocationPlanner.solve_anytime`); the
in-process iterator *is* the contract (a server-streaming gRPC face is a deployment of the same
library, conventions.md §1.4).

- :class:`BoundTracker` — clamps each incumbent's dual bound monotone and recomputes its gap.
- :func:`stream_incumbents` — the tracked incumbent stream over a backend's raw stream.
- :func:`finalize_status` — the honest anytime deadline outcome (``UNKNOWN`` → ``TIMEOUT``).
- :func:`hints_from` — the warm-start seam for incremental online re-solve (delta-path stub).
"""

from __future__ import annotations

from astro_mine.allocate.anytime.bounds import BoundTracker
from astro_mine.allocate.anytime.stream import finalize_status, stream_incumbents
from astro_mine.allocate.anytime.warmstart import hints_from

__all__ = [
    "BoundTracker",
    "finalize_status",
    "hints_from",
    "stream_incumbents",
]
