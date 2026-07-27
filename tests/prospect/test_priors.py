"""RM-P0-PROSPECT-03 — the dataset-derived, provenance-tracked water-ice prior.

Proves the deliverable and its two acceptance criteria: (1) the prior is reconstructable from cited
public inputs via a documented recipe — every dataset is cited and the fit reproduces byte-for-byte;
(2) the prior aligns to the Worlds Shackleton CRS/grid. Also pins the forward-compatible seams the
Phase-1 raster-ingest recipe (#11) plugs into: the recipe registry, the `Provenance` schema, and the
`coldness` conditioning hook.
"""

from __future__ import annotations

import numpy as np
import pytest

from astro_mine.core.resource import ResourceField, check_resource_field
from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.prospect.field import FieldGrid, FieldMetadata
from astro_mine.prospect.priors import (
    SHACKLETON_CRS,
    SHACKLETON_PRIOR_GRID,
    SPECIES,
    UNIT,
    Prior,
    list_priors,
    load_prior,
    register_recipe,
    shackleton_water_ice_v1,
)
from astro_mine.prospect.priors.catalog import LCROSS, LEND_BACKGROUND_WEH

_CANON = "shackleton_water_ice_v1"
_CENTER = (0.0, 0.0, 0.0)
_CORNER = (-29_900.0, -29_900.0, 0.0)
_DATASETS = {"LOLA", "Diviner", "LEND", "M3", "LCROSS"}


def _small_grid() -> FieldGrid:
    return FieldGrid(
        min_x_m=-1_000.0, min_y_m=-1_000.0, max_x_m=1_000.0, max_y_m=1_000.0, n_rows=8, n_cols=8
    )


# --- alignment & contract (AC2) --------------------------------------------------------------


def test_load_prior_is_aligned_to_the_shackleton_grid() -> None:
    prior = load_prior(_CANON)
    md = prior.metadata
    assert isinstance(prior, Prior)
    assert md.species == SPECIES == "water_equivalent_hydrogen"
    assert md.unit == UNIT == "mass_fraction"
    assert md.frame == MOON_BODY_FIXED
    assert md.crs == SHACKLETON_CRS
    assert md.grid == SHACKLETON_PRIOR_GRID
    assert prior.mean.shape == (SHACKLETON_PRIOR_GRID.n_rows, SHACKLETON_PRIOR_GRID.n_cols)


def test_prior_realizes_as_a_core_resource_field() -> None:
    field = load_prior().as_field()
    assert isinstance(field, ResourceField)
    assert check_resource_field(field) is None


def test_load_prior_default_name_matches_canonical() -> None:
    assert load_prior().provenance.recipe == _CANON


def test_custom_grid_alignment() -> None:
    grid = _small_grid()
    prior = load_prior(grid=grid)
    assert prior.metadata.grid == grid
    assert prior.metadata.crs == SHACKLETON_CRS
    assert prior.mean.shape == (8, 8)
    assert check_resource_field(prior.as_field()) is None


# --- provenance & reconstructability (AC1) ---------------------------------------------------


def test_every_dataset_is_cited() -> None:
    citations = load_prior().provenance.citations
    assert {c.short_name for c in citations} >= _DATASETS
    for c in citations:
        assert c.reference and c.product and c.role


def test_fit_is_deterministic_and_content_addressed() -> None:
    a, b = load_prior(), load_prior()
    assert a.provenance.content_hash == b.provenance.content_hash
    assert a.content_hash == b.content_hash
    np.testing.assert_array_equal(a.mean, b.mean)
    np.testing.assert_array_equal(a.variance, b.variance)


def test_provenance_records_the_fit_params() -> None:
    params = load_prior().provenance.params
    assert params["peak_weh"] == pytest.approx(0.056)
    assert params["background_weh"] == pytest.approx(0.005)


def test_parametric_recipe_leaves_source_hash_unset() -> None:
    # The Phase-0 recipe cites published characterizations; raster source hashes are #11's job.
    assert LCROSS.source_hash is None


# --- physical sanity & honest uncertainty ----------------------------------------------------


def test_cold_traps_carry_more_ice_than_the_far_field() -> None:
    field = load_prior().as_field()
    assert field.mean(_CENTER) > field.mean(_CORNER)


def test_peak_mean_is_anchored_to_the_lcross_band() -> None:
    prior = load_prior()
    assert 0.03 <= float(prior.mean.max()) <= 0.08  # ~5.6 wt% LCROSS anchor
    assert 0.0 <= float(prior.mean.min()) <= 0.01  # ~LEND background away from cold traps


def test_mean_is_nonnegative_and_bounded() -> None:
    prior = load_prior()
    assert float(prior.mean.min()) >= 0.0
    assert float(prior.mean.max()) <= 0.1


def test_variance_is_positive_everywhere() -> None:
    assert bool(np.all(load_prior().variance > 0.0))


# --- the registry & conditioning-hook seams (for the Phase-1 raster recipe) -------------------


def test_list_priors_contains_the_canonical_recipe() -> None:
    assert _CANON in list_priors()


def test_load_prior_unknown_name_fails_loudly() -> None:
    with pytest.raises(ValueError, match="unknown prior"):
        load_prior("no_such_prior")


def test_register_recipe_rejects_a_duplicate_name() -> None:
    with pytest.raises(ValueError, match="already registered"):
        register_recipe(_CANON, shackleton_water_ice_v1)


def test_coldness_hook_overrides_the_parametric_default() -> None:
    shape = (SHACKLETON_PRIOR_GRID.n_rows, SHACKLETON_PRIOR_GRID.n_cols)
    driven = shackleton_water_ice_v1(SHACKLETON_PRIOR_GRID, coldness=np.zeros(shape))
    assert np.allclose(driven.mean, LEND_BACKGROUND_WEH)  # coldness=0 → background everywhere
    assert not np.allclose(driven.mean, load_prior().mean)  # differs from the radial default


def test_coldness_hook_rejects_a_mismatched_shape() -> None:
    with pytest.raises(ValueError, match="coldness must have grid shape"):
        shackleton_water_ice_v1(_small_grid(), coldness=np.zeros((3, 3)))


def test_coldness_hook_rejects_out_of_range_weights() -> None:
    with pytest.raises(ValueError, match="must lie in"):
        shackleton_water_ice_v1(_small_grid(), coldness=np.full((8, 8), 2.0))


# --- Prior construction guards ---------------------------------------------------------------


def _metadata(grid: FieldGrid | None) -> FieldMetadata:
    return FieldMetadata(
        species=SPECIES, unit=UNIT, frame=MOON_BODY_FIXED, crs=SHACKLETON_CRS, grid=grid
    )


def test_prior_requires_a_grid_domain() -> None:
    prov = load_prior(grid=_small_grid()).provenance
    arr = np.zeros((1, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="requires metadata"):
        Prior(_metadata(None), arr, arr, prov)


def test_prior_rejects_mismatched_array_shapes() -> None:
    prov = load_prior(grid=_small_grid()).provenance
    bad = np.zeros((2, 2), dtype=np.float64)
    with pytest.raises(ValueError, match="must have shape"):
        Prior(_metadata(_small_grid()), bad, bad, prov)


def test_prior_rejects_negative_variance() -> None:
    prov = load_prior(grid=_small_grid()).provenance
    mean = np.zeros((8, 8), dtype=np.float64)
    variance = np.full((8, 8), -1.0, dtype=np.float64)
    with pytest.raises(ValueError, match="non-negative"):
        Prior(_metadata(_small_grid()), mean, variance, prov)
