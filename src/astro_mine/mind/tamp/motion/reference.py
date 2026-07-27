"""Reference RRT motion planner (RM-P1-MIND-03).

A deterministic, pure-Python rapidly-exploring random tree in the ground plane: the CI-tested
default behind the motion-feasibility contract (mind.md §4, "OMPL sampling-based motion"). It
returns a straight-line path when the goal is directly reachable (the common unobstructed case,
so the toy scenario touches no RNG and stays trivially reproducible) and grows a tree around
circular keep-outs otherwise. Given a seeded ``random.Random`` it is bit-reproducible; a real
OMPL planner (RRT*/PRM*/BIT*) with FCL collision drops in behind the same
:class:`ReferenceMotionPlanner.plan` shape via the ``[native]`` extra.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from random import Random

__all__ = ["Obstacle", "Point", "ReferenceMotionPlanner"]

Point = tuple[float, float]


@dataclass(frozen=True, slots=True)
class Obstacle:
    """A circular keep-out (center + radius) in the ground plane — the reference collision
    primitive (a Guard/Worlds keep-out reduced to 2-D for the toy scenario)."""

    center: Point
    radius: float


class ReferenceMotionPlanner:
    """A seeded RRT over the ground plane with circular keep-outs."""

    def __init__(
        self, *, step_m: float = 1.5, max_samples: int = 512, bound_m: float = 64.0
    ) -> None:
        self._step_m = step_m
        self._max_samples = max_samples
        self._bound_m = bound_m

    def plan(
        self, start: Point, goal: Point, obstacles: Sequence[Obstacle], rng: Random
    ) -> list[Point]:
        """A collision-free waypoint path from ``start`` to ``goal``.

        Straight line when unobstructed; otherwise an RRT grown with ``rng`` (bit-reproducible).
        Falls back to the direct ``[goal]`` if the sample budget is exhausted — a defined,
        best-effort result rather than a failure (degrade-not-collapse)."""
        if self._segment_free(start, goal, obstacles):
            return [goal]
        nodes: list[Point] = [start]
        parents: dict[int, int] = {0: -1}
        for _ in range(self._max_samples):
            sample = self._sample(goal, rng)
            nearest = min(range(len(nodes)), key=lambda i: _dist(nodes[i], sample))
            new = self._steer(nodes[nearest], sample)
            if not self._segment_free(nodes[nearest], new, obstacles):
                continue
            parents[len(nodes)] = nearest
            nodes.append(new)
            if self._segment_free(new, goal, obstacles):
                parents[len(nodes)] = len(nodes) - 1
                nodes.append(goal)
                return self._path(nodes, parents)
        return [goal]

    def _sample(self, goal: Point, rng: Random) -> Point:
        # 10% goal-biased sampling (deterministic given rng) — standard RRT heuristic.
        if rng.random() < 0.1:
            return goal
        return (
            rng.uniform(-self._bound_m, self._bound_m),
            rng.uniform(-self._bound_m, self._bound_m),
        )

    def _steer(self, origin: Point, toward: Point) -> Point:
        distance = _dist(origin, toward)
        if distance <= self._step_m:
            return toward
        scale = self._step_m / distance
        return (
            origin[0] + (toward[0] - origin[0]) * scale,
            origin[1] + (toward[1] - origin[1]) * scale,
        )

    def _segment_free(self, a: Point, b: Point, obstacles: Sequence[Obstacle]) -> bool:
        return all(_segment_clear_of(a, b, o) for o in obstacles)

    @staticmethod
    def _path(nodes: list[Point], parents: dict[int, int]) -> list[Point]:
        index = len(nodes) - 1
        chain: list[Point] = []
        while index != -1:
            chain.append(nodes[index])
            index = parents[index]
        chain.reverse()
        return chain[1:]  # drop the start node; the caller already sits there


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _segment_clear_of(a: Point, b: Point, obstacle: Obstacle) -> bool:
    """Whether segment ``a→b`` stays outside ``obstacle`` (point-to-segment distance ≥ r)."""
    cx, cy = obstacle.center
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    if length_sq == 0.0:
        return _dist(a, obstacle.center) >= obstacle.radius
    t = max(0.0, min(1.0, ((cx - a[0]) * dx + (cy - a[1]) * dy) / length_sq))
    closest = (a[0] + t * dx, a[1] + t * dy)
    return _dist(closest, obstacle.center) >= obstacle.radius
