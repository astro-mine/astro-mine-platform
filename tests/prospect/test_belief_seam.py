"""The conditionable-belief seam a simulator drives (astro-mine-sim#66).

The contract that matters here is the *boundary*: everything crossing it is a Core type, because
the consumer cannot name a Prospect one. These tests pin that, the cell-id convention, and the
projection — and they check the seam refuses to invent a likelihood it was not given.
"""

from __future__ import annotations

import math

import pytest

from astro_mine.core.messages.model import SensorReading
from astro_mine.prospect.belief import BeliefField, GriddedBelief, cell_id
from astro_mine.prospect.belief.seam import _to_lat_lon
from astro_mine.prospect.priors.catalog import SHACKLETON_PRIOR_GRID
from astro_mine.prospect.priors.recipe import shackleton_water_ice_v1

_RADIUS_M = 1737400.0


@pytest.fixture(scope="module")
def belief() -> GriddedBelief:
    prior = shackleton_water_ice_v1(SHACKLETON_PRIOR_GRID)
    return GriddedBelief(BeliefField.from_prior(prior), prior.metadata)


def _reading(value: float, *, sigma: float | None = 0.01, valid: bool = True) -> SensorReading:
    return SensorReading(
        sensor="neutron_spectrometer",
        values=[value],
        unit="mass_fraction",
        resource_species="water_equivalent_hydrogen",
        noise_sigma=sigma,
        valid=valid,
    )


# --- cell identity ---------------------------------------------------------------------


def test_cell_ids_are_zero_padded_and_sort_in_grid_order() -> None:
    assert cell_id(0, 0) == "r0000c0000"
    assert cell_id(119, 7) == "r0119c0007"
    # Lexicographic order is grid order — the property the padding buys.
    assert sorted([cell_id(10, 2), cell_id(2, 10), cell_id(2, 2)]) == [
        cell_id(2, 2),
        cell_id(2, 10),
        cell_id(10, 2),
    ]


def test_every_grid_cell_is_addressed_exactly_once(belief: GriddedBelief) -> None:
    cells = belief.cells()
    grid = SHACKLETON_PRIOR_GRID
    assert len(cells) == grid.n_rows * grid.n_cols
    assert cell_id(0, 0) in cells and cell_id(grid.n_rows - 1, grid.n_cols - 1) in cells


def test_cells_carry_the_fields_own_species_and_unit(belief: GriddedBelief) -> None:
    # The consumer scores against these, so they must be the field's rather than a default.
    sample = belief.cells()[cell_id(120, 120)]
    assert sample.species == "water_equivalent_hydrogen"
    assert sample.unit == "mass_fraction"
    assert sample.variance > 0.0


# --- the projection --------------------------------------------------------------------


def test_the_pole_projects_to_the_pole() -> None:
    assert _to_lat_lon(0.0, 0.0, _RADIUS_M) == (-90.0, 0.0)


def test_longitude_follows_the_projections_axis_convention() -> None:
    # +y is the prime meridian and longitude increases toward +x, per `atan2(x, y)`.
    assert _to_lat_lon(0.0, 30_000.0, _RADIUS_M)[1] == pytest.approx(0.0)
    assert _to_lat_lon(30_000.0, 0.0, _RADIUS_M)[1] == pytest.approx(90.0)
    assert _to_lat_lon(-30_000.0, 0.0, _RADIUS_M)[1] == pytest.approx(270.0)


def test_latitude_matches_the_arc_distance_to_within_the_projections_own_distortion() -> None:
    # Stereographic rho is 2R*tan(c/2), which *exceeds* R*c — the projection stretches distance
    # away from the tangent point. So inverting a fixed rho of 30 km yields a colatitude slightly
    # *smaller* than rho/R, i.e. a point marginally closer to the pole than a naive arc-length
    # reading suggests. Asserting the sign of that third-order difference is what distinguishes a
    # correct inverse from one that merely looks plausible: a wrong-but-plausible implementation
    # (say, treating rho as arc length) passes the magnitude check below and fails this line.
    lat, _ = _to_lat_lon(30_000.0, 0.0, _RADIUS_M)
    arc_colat_deg = math.degrees(30_000.0 / _RADIUS_M)
    assert lat == pytest.approx(-90.0 + arc_colat_deg, abs=1e-3)
    assert lat < -90.0 + arc_colat_deg


def test_a_polar_region_selects_a_cap_of_cells(belief: GriddedBelief) -> None:
    wide = belief.cells_in_region(lat_deg=(-90.0, -89.5), lon_deg=(0.0, 360.0))
    narrow = belief.cells_in_region(lat_deg=(-90.0, -89.9), lon_deg=(0.0, 360.0))
    assert narrow < wide, "a tighter cap must select a strict subset"
    assert narrow, "the cap around the pole cannot be empty"
    assert len(wide) < len(belief.cells()), "the cap is not the whole grid"


def test_a_longitude_window_may_cross_the_prime_meridian(belief: GriddedBelief) -> None:
    # Authored unwrapped so `min < max` holds: [350, 370] spans 350 deg -> 10 deg.
    crossing = belief.cells_in_region(lat_deg=(-90.0, -89.0), lon_deg=(350.0, 370.0))
    assert crossing, "a meridian-crossing window selected nothing"
    # Cells just east of the meridian are inside it.
    east = belief.cells_in_region(lat_deg=(-90.0, -89.0), lon_deg=(0.0, 10.0))
    assert east <= crossing


# --- conditioning ----------------------------------------------------------------------


def test_observing_reduces_uncertainty_where_it_was_observed(belief: GriddedBelief) -> None:
    centre = cell_id(120, 120)
    before = belief.cells()[centre]
    after = belief.observe([(_reading(0.03), (125.0, 125.0, 0.0), 0.0)]).cells()[centre]
    assert after.variance < before.variance, "conditioning did not reduce the posterior variance"
    # Still a distribution, never collapsed to certainty — `information_gain` errors on a
    # non-positive variance, so a belief that zeroed one would take the metric down with it.
    assert after.variance > 0.0


def test_the_prior_is_not_mutated_by_conditioning(belief: GriddedBelief) -> None:
    centre = cell_id(120, 120)
    before = belief.cells()[centre].variance
    belief.observe([(_reading(0.03), (125.0, 125.0, 0.0), 0.0)])
    assert belief.cells()[centre].variance == before


def test_a_reading_without_a_likelihood_is_skipped_not_guessed_at(belief: GriddedBelief) -> None:
    # No `noise_sigma` means no likelihood. Inventing one would fabricate the confidence this
    # seam exists to avoid, so the reading is skipped and the belief is returned unchanged.
    same = belief.observe([(_reading(0.03, sigma=None), (125.0, 125.0, 0.0), 0.0)])
    assert same is belief


def test_an_invalid_reading_is_skipped(belief: GriddedBelief) -> None:
    # `valid=False` is the sensor saying it measured nothing — not a measurement of zero.
    assert belief.observe([(_reading(0.03, valid=False), (125.0, 125.0, 0.0), 0.0)]) is belief


def test_conditioning_is_order_stable_over_the_same_log(belief: GriddedBelief) -> None:
    # The replay property, at the seam: one batch equals two halves.
    log = [
        (_reading(0.03), (125.0, 125.0, 0.0), 0.0),
        (_reading(0.04), (1125.0, 125.0, 0.0), 1.0),
    ]
    batch = belief.observe(log).cells()
    split = belief.observe(log[:1]).observe(log[1:]).cells()
    centre = cell_id(120, 120)
    assert split[centre].mean == pytest.approx(batch[centre].mean)
    assert split[centre].variance == pytest.approx(batch[centre].variance)


def test_a_belief_without_a_grid_is_refused() -> None:
    prior = shackleton_water_ice_v1(SHACKLETON_PRIOR_GRID)
    ungridded = prior.metadata.model_copy(update={"grid": None})
    with pytest.raises(ValueError, match="needs the field's grid"):
        GriddedBelief(BeliefField.from_prior(prior), ungridded)
