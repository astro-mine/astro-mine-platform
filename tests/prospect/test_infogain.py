"""RM-P0-PROSPECT-06 — information-gain maps for active perception.

Proves the deliverables and acceptance criteria (prospect.md §3, §6; LUNAR-FR-002):

- the **variance** and **mutual-information** maps over a belief, and that they devalue
  densely-observed regions — so active perception heads for the unobserved ground (AC1);
- the **information-gain** scalar (entropy drop) that Bench's metric scores (AC2; RM-P0-BENCH-03);
- determinism and the fail-loud guards.

Beliefs are built with a uniform prior via the public constructor so the assertions isolate the
effect of conditioning from the dataset prior's spatial structure (a realistic-prior smoke test
covers ``load_prior``).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from astro_mine.core.resource import Position
from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.prospect.belief import BeliefField, FieldObservation
from astro_mine.prospect.field import FieldGrid, FieldMetadata
from astro_mine.prospect.infogain import (
    best_sample_position,
    entropy_map,
    field_entropy,
    information_gain,
    information_gain_map,
    mutual_information_map,
    variance_map,
)
from astro_mine.prospect.priors import SHACKLETON_CRS, SPECIES, UNIT, load_prior

_GRID = FieldGrid(
    min_x_m=-1_000.0, min_y_m=-1_000.0, max_x_m=1_000.0, max_y_m=1_000.0, n_rows=8, n_cols=8
)
_A = (900.0, 900.0, 0.0)  # a corner we will observe densely  -> grid cell (row 7, col 7)
_B = (-900.0, -900.0, 0.0)  # the opposite, unobserved corner -> grid cell (row 0, col 0)


def _metadata(grid: FieldGrid = _GRID) -> FieldMetadata:
    return FieldMetadata(
        species=SPECIES, unit=UNIT, frame=MOON_BODY_FIXED, crs=SHACKLETON_CRS, grid=grid
    )


def _uniform_belief(
    grid: FieldGrid = _GRID, *, mean: float = 0.0, var: float = 1.0, length_scale: float = 300.0
) -> BeliefField:
    shape = (grid.n_rows, grid.n_cols)
    return BeliefField(
        _metadata(grid),
        np.full(shape, mean),
        np.full(shape, var),
        (),
        length_scale=length_scale,
        prior_hash="test-prior",
    )


def _cluster(center: Position, *, noise: float = 0.01) -> list[FieldObservation]:
    cx, cy, _ = center
    return [
        FieldObservation(x_m=cx + dx, y_m=cy + dy, value=0.5, noise_sigma=noise)
        for dx in (-50.0, 0.0, 50.0)
        for dy in (-50.0, 0.0, 50.0)
    ]


# --- variance map ----------------------------------------------------------------------------


def test_variance_map_is_the_posterior_variance_and_a_copy() -> None:
    belief = _uniform_belief()
    vm = variance_map(belief)
    assert vm.shape == (8, 8)
    np.testing.assert_array_equal(vm, belief.variance_grid())
    vm[0, 0] = 1e9  # mutating the returned map must not bleed back into the belief
    assert belief.variance_grid()[0, 0] != 1e9


def test_variance_map_drops_where_observed() -> None:
    observed = _uniform_belief().update(_cluster(_A))
    vm = variance_map(observed)
    assert vm[7, 7] < 0.5  # the observed (+,+) corner collapsed below the prior
    assert vm[0, 0] == pytest.approx(1.0)  # the opposite corner stayed at the prior


# --- entropy / information gain (the Bench-shared definition) ---------------------------------


def test_entropy_map_matches_the_gaussian_formula() -> None:
    expected = 0.5 * (math.log(2.0 * math.pi * math.e) + math.log(2.0))
    np.testing.assert_allclose(entropy_map(_uniform_belief(var=2.0)), np.full((8, 8), expected))


def test_field_entropy_sums_the_entropy_map() -> None:
    belief = _uniform_belief()
    assert field_entropy(belief) == pytest.approx(float(entropy_map(belief).sum()))


def test_information_gain_is_nonnegative_and_equals_the_entropy_drop() -> None:
    prior = _uniform_belief()
    observed = prior.update(_cluster(_A))
    gain = information_gain(prior, observed)
    assert gain > 0.0  # conditioning removed uncertainty
    assert gain == pytest.approx(field_entropy(prior) - field_entropy(observed))


def test_information_gain_is_zero_for_an_unchanged_belief() -> None:
    belief = _uniform_belief()
    assert information_gain(belief, belief) == pytest.approx(0.0)


def test_information_gain_rejects_mismatched_grids() -> None:
    coarse = FieldGrid(
        min_x_m=-1_000.0, min_y_m=-1_000.0, max_x_m=1_000.0, max_y_m=1_000.0, n_rows=4, n_cols=4
    )
    with pytest.raises(ValueError, match="same grid shape"):
        information_gain(_uniform_belief(), _uniform_belief(coarse))


# --- mutual-information map -------------------------------------------------------------------


def test_mutual_information_map_is_nonnegative_and_shaped() -> None:
    mi = mutual_information_map(_uniform_belief(), noise_sigma=0.1)
    assert mi.shape == (8, 8)
    assert bool(np.all(mi >= 0.0))


def test_mutual_information_decreases_with_noisier_sensors() -> None:
    belief = _uniform_belief()
    sharp = mutual_information_map(belief, noise_sigma=0.05).sum()
    blunt = mutual_information_map(belief, noise_sigma=1.0).sum()
    assert sharp > blunt


def test_mutual_information_rejects_nonpositive_noise() -> None:
    with pytest.raises(ValueError, match="noise_sigma must be positive"):
        mutual_information_map(_uniform_belief(), noise_sigma=0.0)


# --- active perception: devalue observed regions, target unobserved ones (AC1) ----------------


def test_both_maps_devalue_the_densely_observed_corner() -> None:
    observed = _uniform_belief().update(_cluster(_A))
    vm = variance_map(observed)
    mi = mutual_information_map(observed, noise_sigma=0.1)
    assert vm[7, 7] < vm[0, 0]  # the sampled (+,+) corner is now worth less than the untouched one
    assert mi[7, 7] < mi[0, 0]


def test_best_sample_position_targets_the_unobserved_region() -> None:
    observed = _uniform_belief().update(_cluster(_A))
    pos = best_sample_position(variance_map(observed), observed)
    assert math.dist(pos, _B) < math.dist(pos, _A)  # next sample heads away from what we know


def test_best_sample_position_returns_the_argmax_cell_center() -> None:
    belief = _uniform_belief()
    info_map = np.zeros((8, 8))
    info_map[2, 5] = 1.0
    dx = dy = 2_000.0 / 8
    expected = (-1_000.0 + (5 + 0.5) * dx, -1_000.0 + (2 + 0.5) * dy, 0.0)
    assert best_sample_position(info_map, belief) == pytest.approx(expected)


def test_best_sample_position_rejects_a_mismatched_map() -> None:
    with pytest.raises(ValueError, match="does not match the belief grid"):
        best_sample_position(np.zeros((3, 3)), _uniform_belief())


# --- unified dispatch + determinism -----------------------------------------------------------


def test_information_gain_map_dispatches_on_kind() -> None:
    belief = _uniform_belief()
    np.testing.assert_array_equal(information_gain_map(belief), variance_map(belief))  # default
    np.testing.assert_array_equal(
        information_gain_map(belief, kind="variance"), variance_map(belief)
    )
    np.testing.assert_array_equal(
        information_gain_map(belief, kind="mutual_information", noise_sigma=0.1),
        mutual_information_map(belief, noise_sigma=0.1),
    )
    with pytest.raises(ValueError, match="requires noise_sigma"):
        information_gain_map(belief, kind="mutual_information")
    with pytest.raises(ValueError, match="unknown info-gain kind"):
        information_gain_map(belief, kind="entropy")  # type: ignore[arg-type]


def test_maps_are_deterministic() -> None:
    a, b = _uniform_belief(), _uniform_belief()
    np.testing.assert_array_equal(variance_map(a), variance_map(b))
    np.testing.assert_array_equal(
        mutual_information_map(a, noise_sigma=0.2), mutual_information_map(b, noise_sigma=0.2)
    )


def test_belief_exposes_length_scale_and_a_variance_grid_copy() -> None:
    belief = _uniform_belief(length_scale=300.0)
    assert belief.length_scale == 300.0
    grid_copy = belief.variance_grid()
    grid_copy[0, 0] = 5.0
    assert belief.variance_grid()[0, 0] != 5.0


def test_dataset_prior_produces_well_formed_maps() -> None:
    belief = BeliefField.from_prior(load_prior(grid=_GRID))
    assert variance_map(belief).shape == (8, 8)
    assert bool(np.all(mutual_information_map(belief, noise_sigma=0.1) >= 0.0))
