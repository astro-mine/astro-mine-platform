# SPDX-License-Identifier: Apache-2.0
"""Pareto math (studio.md §3 ``designspace/pareto``) — dominance, non-dominated sorting,
crowding distance, hypervolume, ranking.

Pure and deterministic (no RNG): the same objective points always produce the same front.
Each objective carries a **sense** — ``True`` = maximize, ``False`` = minimize — so the
same routines serve higher-better and lower-better metrics without the caller pre-negating.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

Point = Sequence[float]


def dominates(a: Point, b: Point, senses: Sequence[bool]) -> bool:
    """``a`` Pareto-dominates ``b``: at least as good on every objective and strictly
    better on one (per each objective's sense)."""
    at_least_as_good = True
    strictly_better = False
    for ai, bi, maximize in zip(a, b, senses, strict=True):
        if maximize:
            if ai < bi:
                at_least_as_good = False
            elif ai > bi:
                strictly_better = True
        else:
            if ai > bi:
                at_least_as_good = False
            elif ai < bi:
                strictly_better = True
    return at_least_as_good and strictly_better


def non_dominated_sort(points: Sequence[Point], senses: Sequence[bool]) -> list[list[int]]:
    """Deb's fast non-dominated sort → fronts of point indices, best front first."""
    n = len(points)
    dominated: list[list[int]] = [[] for _ in range(n)]
    domination_count = [0] * n
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(points[p], points[q], senses):
                dominated[p].append(q)
            elif dominates(points[q], points[p], senses):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        nxt: list[int] = []
        for p in fronts[i]:
            for q in dominated[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def pareto_front(points: Sequence[Point], senses: Sequence[bool]) -> list[int]:
    """Indices of the non-dominated (first) front; empty for an empty input."""
    fronts = non_dominated_sort(points, senses)
    return fronts[0] if fronts else []


def crowding_distance(
    points: Sequence[Point], front: Sequence[int], senses: Sequence[bool]
) -> dict[int, float]:
    """NSGA-II crowding distance over one front (boundary points get ``inf``)."""
    distance = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: math.inf for i in front}
    for k in range(len(senses)):
        ordered = sorted(front, key=lambda i: points[i][k])
        distance[ordered[0]] = math.inf
        distance[ordered[-1]] = math.inf
        span = points[ordered[-1]][k] - points[ordered[0]][k]
        if span == 0:
            continue
        for j in range(1, len(ordered) - 1):
            gap = points[ordered[j + 1]][k] - points[ordered[j - 1]][k]
            distance[ordered[j]] += gap / span
    return distance


def _to_maximization(
    points: Sequence[Point], reference: Point, senses: Sequence[bool]
) -> tuple[list[tuple[float, ...]], tuple[float, ...]]:
    converted = [
        tuple(pi if maximize else -pi for pi, maximize in zip(p, senses, strict=True))
        for p in points
    ]
    ref = tuple(ri if maximize else -ri for ri, maximize in zip(reference, senses, strict=True))
    return converted, ref


def _hso(points: Sequence[tuple[float, ...]], reference: tuple[float, ...]) -> float:
    """Hypervolume by Slicing Objectives (HSO) for maximization above ``reference``,
    exact for any number of objectives."""
    active = [p for p in points if all(p[k] > reference[k] for k in range(len(reference)))]
    if not active:
        return 0.0
    if len(reference) == 1:
        return max(p[0] for p in active) - reference[0]
    ordered = sorted(active, key=lambda p: p[0], reverse=True)
    volume = 0.0
    for i, point in enumerate(ordered):
        lower = ordered[i + 1][0] if i + 1 < len(ordered) else reference[0]
        depth = point[0] - lower
        if depth <= 0:
            continue
        slab = [q[1:] for q in ordered[: i + 1]]
        volume += depth * _hso(slab, reference[1:])
    return volume


def hypervolume(points: Sequence[Point], reference: Point, senses: Sequence[bool]) -> float:
    """Dominated hypervolume of a point set relative to a reference point (the nadir /
    worst-acceptable corner). Larger is a better-spread, better-converged front."""
    if not points:
        return 0.0
    converted, ref = _to_maximization(points, reference, senses)
    return _hso(converted, ref)
