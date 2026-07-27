"""STUDIO-02 — Pareto math: dominance, sorting, crowding, hypervolume."""

from __future__ import annotations

import math

from astro_mine.studio.designspace import (
    crowding_distance,
    dominates,
    hypervolume,
    non_dominated_sort,
    pareto_front,
)

MAX2 = (True, True)


def test_dominates_maximize_and_minimize() -> None:
    assert dominates((2.0, 2.0), (1.0, 1.0), MAX2)  # strictly better on both (maximize)
    assert not dominates((2.0, 1.0), (1.0, 2.0), MAX2)  # trade-off — neither dominates
    assert not dominates((1.0, 1.0), (1.0, 1.0), MAX2)  # equal — not strict
    # minimize: lower is better
    assert dominates((1.0, 1.0), (2.0, 2.0), (False, False))
    assert not dominates((2.0, 2.0), (1.0, 1.0), (False, False))


def test_non_dominated_sort_layers_fronts() -> None:
    points = [(3.0, 3.0), (1.0, 1.0), (3.0, 1.0), (1.0, 3.0)]
    fronts = non_dominated_sort(points, MAX2)
    assert 0 in fronts[0]  # (3,3) dominates everything → alone on the first front
    assert fronts[0] == [0]
    assert set(fronts[1]) == {2, 3}
    assert fronts[-1] == [1]


def test_pareto_front_empty_and_nonempty() -> None:
    assert pareto_front([], MAX2) == []
    assert pareto_front([(1.0, 2.0), (2.0, 1.0)], MAX2) == [0, 1]  # both non-dominated


def test_crowding_distance_boundaries_and_interior() -> None:
    front = [0, 1]
    assert crowding_distance([(1.0, 1.0), (2.0, 2.0)], front, MAX2) == {0: math.inf, 1: math.inf}
    points = [(0.0, 3.0), (1.0, 2.0), (2.0, 1.0), (3.0, 0.0)]
    dist = crowding_distance(points, [0, 1, 2, 3], MAX2)
    assert dist[0] == math.inf and dist[3] == math.inf  # extremes
    assert dist[1] > 0.0 and dist[2] > 0.0  # interior points get finite spread


def test_crowding_distance_zero_span_objective() -> None:
    # every point identical on objective 0 → that objective's span is 0 (continue branch)
    points = [(1.0, 0.0), (1.0, 1.0), (1.0, 2.0)]
    dist = crowding_distance(points, [0, 1, 2], MAX2)
    assert dist[1] > 0.0  # still gets spread from objective 1


def test_hypervolume_two_dimensions() -> None:
    assert hypervolume([(3.0, 1.0), (1.0, 3.0)], (0.0, 0.0), MAX2) == 5.0
    assert hypervolume([], (0.0, 0.0), MAX2) == 0.0


def test_hypervolume_three_dimensions() -> None:
    assert hypervolume([(2.0, 2.0, 2.0)], (0.0, 0.0, 0.0), (True, True, True)) == 8.0


def test_hypervolume_minimize_sense() -> None:
    assert hypervolume([(1.0, 1.0)], (3.0, 3.0), (False, False)) == 4.0


def test_hypervolume_point_below_reference_is_zero() -> None:
    assert hypervolume([(1.0, 1.0)], (2.0, 2.0), MAX2) == 0.0


def test_hypervolume_equal_first_objective_slab() -> None:
    # two points share objective-0 → a zero-depth slab is skipped
    assert hypervolume([(2.0, 1.0), (2.0, 3.0)], (0.0, 0.0), MAX2) == 6.0
