"""The motion-feasibility contract (RM-P1-MIND-03).

The geometric half of TAMP, expressed as the narrow seam every motion backend fills: given a
start, a goal, and the keep-outs to avoid, return a collision-free waypoint path. Making it an
explicit :class:`~typing.Protocol` (rather than leaving it implicit in the reference planner's
shape) is what lets the TAMP tier hold *any* backend — the pure-Python reference RRT
(:mod:`~astro_mine.mind.tamp.motion.reference`, the CI-tested bit-exact default) or the native
OMPL/FCL planner (:mod:`~astro_mine.mind.tamp.motion.native`, the ``[native]`` extra) — without
either knowing about the other. The framework commits to the interface, not the backend
(mind.md §2, principle 2).

``rng`` is the caller's seeded :class:`random.Random`, threaded from ``DecisionContext.seed`` so a
seeded run reproduces. A backend MUST NOT source randomness anywhere else; where a native library
owns a global RNG it cannot fully honour this, its manifest declares
``determinism_class: tolerance`` and says so (the OMPL adapter does).

A planner returns a **best-effort defined result**, never an exception: if the sample budget is
exhausted it returns the direct ``[goal]`` rather than failing (degrade-not-collapse, principle 4).
"""

from __future__ import annotations

from collections.abc import Sequence
from random import Random
from typing import Protocol, runtime_checkable

from astro_mine.mind.tamp.motion.reference import Obstacle, Point

__all__ = ["MotionPlanner"]


@runtime_checkable
class MotionPlanner(Protocol):
    """A sampling-based (or any) motion planner over the ground plane."""

    def plan(
        self, start: Point, goal: Point, obstacles: Sequence[Obstacle], rng: Random
    ) -> list[Point]:
        """A collision-free waypoint path from ``start`` to ``goal`` avoiding ``obstacles``.

        The path excludes ``start`` (the caller already sits there) and ends at ``goal``.
        """
        ...
