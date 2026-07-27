"""Illumination + PSR detection (RM-P0-WORLDS-03).

Two layers, mirroring the repo's "real pipeline against synthetic inputs in CI" pattern:

- **Analytic kernel tests** drive the pure ``_horizon`` kernels on hand-built terrain with
  a known skyline, gating on explicit error budgets (a flat plane, a single wall, a crater
  rim) — the in-CI stand-in for "regression vs. published lunar references". The real
  Shackleton validation runs outside CI via ``scripts/validate_illumination.py``.
- **Integration tests** build an :class:`IlluminationModel` from an ingested synthetic DEM
  and the ``synthetic_spice`` kernel set (``abcorr="NONE"``), exercising the O(1) Sun-
  visibility path, the PSR-mask window, and content-addressed determinism.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import rasterio.transform

from astro_mine.spice import Site, SpiceGeometryError, epoch_range, sun_geometry
from astro_mine.worlds.illumination import (
    BUNDLE_SCHEMA,
    IlluminationError,
    IlluminationModel,
    PsrEpochSemantics,
    PsrResult,
    _require_polar_stereographic,
)
from astro_mine.worlds.illumination._horizon import (
    FLAT_HORIZON_DEG,
    azimuth_bin,
    curvature_drop_m,
    horizon_field,
    horizon_hash,
    sun_visibility_raster,
    topocentric_to_world_azimuth,
)
from astro_mine.worlds.terrain import ingest_dem

_FLAT = 1.0e12  # an effectively flat planet radius — disables the curvature term in kernels
_PX = (1.0, -1.0)  # a north-up 1 m grid: +x = east (col+), +y = north (row-)


# --- pure kernels ----------------------------------------------------------------


def test_azimuth_bin_wraps_and_bins() -> None:
    assert azimuth_bin(0.0, 4) == 0
    assert azimuth_bin(90.0, 4) == 1
    assert azimuth_bin(359.9, 4) == 3
    assert azimuth_bin(360.0, 4) == 0  # wraps
    assert azimuth_bin(-90.0, 4) == 3  # wraps


def test_azimuth_bin_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="n_azimuth must be positive"):
        azimuth_bin(10.0, 0)


def test_topocentric_to_world_azimuth() -> None:
    assert topocentric_to_world_azimuth(30.0, 0.0) == pytest.approx(30.0)  # lon 0 => identity
    assert topocentric_to_world_azimuth(350.0, 20.0) == pytest.approx(10.0)  # wraps mod 360
    assert topocentric_to_world_azimuth(30.0, -40.0) == pytest.approx(350.0)


def test_curvature_drop_matches_formula() -> None:
    assert curvature_drop_m(1000.0, 1_737_400.0) == pytest.approx(1000.0**2 / (2 * 1_737_400.0))
    drops = curvature_drop_m(np.array([0.0, 100.0, 200.0]), 1000.0)
    assert drops[0] == 0.0 and drops[2] > drops[1]  # monotone in distance


def test_flat_plane_horizon_is_zero() -> None:
    horizon = horizon_field(
        np.full((15, 15), 42.0),
        pixel_size_m=_PX,
        n_azimuth=16,
        max_radius_m=10.0,
        body_radius_m=_FLAT,
    )
    assert horizon.shape == (15, 15, 16)
    np.testing.assert_allclose(horizon, FLAT_HORIZON_DEG, atol=1e-4)


def test_wall_raises_horizon_in_its_azimuth() -> None:
    # A tall wall along the east columns: the skyline is high looking east, flat west.
    elev = np.zeros((21, 21), dtype=np.float64)
    elev[:, 18:] = 100.0
    horizon = horizon_field(
        elev, pixel_size_m=_PX, n_azimuth=72, max_radius_m=15.0, body_radius_m=_FLAT
    )
    east = horizon[10, 10, azimuth_bin(90.0, 72)]  # wall 8 m east, 100 m tall
    west = horizon[10, 10, azimuth_bin(270.0, 72)]
    assert east == pytest.approx(math.degrees(math.atan2(100.0, 8.0)), abs=2.0)
    assert west < 1.0


def test_crater_floor_is_psr_adjacent_to_lit_terrain() -> None:
    # A circular rim around the centre: the floor sees a high horizon in every azimuth, so a
    # low Sun never clears it (a PSR), while a high Sun does and outer cells stay lit.
    yy, xx = np.mgrid[0:21, 0:21]
    r = np.hypot(xx - 10, yy - 10)
    crater = np.where((r >= 6) & (r <= 9), 60.0, 0.0).astype(np.float64)
    horizon = horizon_field(
        crater, pixel_size_m=_PX, n_azimuth=72, max_radius_m=15.0, body_radius_m=_FLAT
    )
    for az in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0):
        assert not sun_visibility_raster(horizon, 5.0, az)[10, 10]  # deep-shadow PSR
    assert sun_visibility_raster(horizon, 89.0, 0.0)[10, 10]  # a high Sun reaches the floor
    assert sun_visibility_raster(horizon, 5.0, 0.0).any()  # outer terrain is lit


def test_sun_visibility_raster_thresholds_elevation() -> None:
    horizon = np.zeros((3, 3, 4), dtype=np.float32)
    horizon[1, 1, 0] = 20.0
    assert sun_visibility_raster(horizon, 10.0, 0.0)[0, 0]  # flat cell, sun up
    assert not sun_visibility_raster(horizon, 10.0, 0.0)[1, 1]  # blocked by its 20 deg horizon
    assert sun_visibility_raster(horizon, 25.0, 0.0)[1, 1]  # clears it


def test_horizon_field_validates_inputs() -> None:
    with pytest.raises(ValueError, match="must be 2-D"):
        horizon_field(
            np.zeros((2, 2, 2)),
            pixel_size_m=_PX,
            n_azimuth=4,
            max_radius_m=5.0,
            body_radius_m=_FLAT,
        )
    with pytest.raises(ValueError, match="non-zero"):
        horizon_field(
            np.zeros((4, 4)),
            pixel_size_m=(0.0, -1.0),
            n_azimuth=4,
            max_radius_m=5.0,
            body_radius_m=_FLAT,
        )
    with pytest.raises(ValueError, match="smaller than one pixel"):
        horizon_field(
            np.zeros((4, 4)), pixel_size_m=_PX, n_azimuth=4, max_radius_m=0.5, body_radius_m=_FLAT
        )


def test_horizon_field_skips_degenerate_zero_steps() -> None:
    # Anisotropic pixels: a near-axis ray whose cross-axis step rounds to 0 pixels lands on
    # the same cell and is skipped without error, leaving the flat horizon.
    horizon = horizon_field(
        np.zeros((5, 5)),
        pixel_size_m=(1.0, -50.0),
        n_azimuth=4,
        max_radius_m=2.0,
        body_radius_m=_FLAT,
    )
    assert horizon.shape == (5, 5, 4)
    np.testing.assert_allclose(horizon, FLAT_HORIZON_DEG, atol=1e-6)


def test_horizon_hash_is_deterministic_and_content_sensitive() -> None:
    horizon = np.zeros((4, 4, 8), dtype=np.float32)
    meta = {"schema": "x", "n": 1}
    assert horizon_hash(horizon, meta) == horizon_hash(horizon, meta)
    assert horizon_hash(horizon, meta).startswith("sha256:")
    other = horizon.copy()
    other[0, 0, 0] = 1.0
    assert horizon_hash(other, meta) != horizon_hash(horizon, meta)
    assert horizon_hash(horizon, {"schema": "y"}) != horizon_hash(horizon, meta)


def test_require_polar_stereographic_rejects_other_crs() -> None:
    with pytest.raises(IlluminationError, match="RM-P1-WORLDS-12"):
        _require_polar_stereographic("+proj=longlat +R=1737400 +no_defs")
    with pytest.raises(IlluminationError, match="south-polar stereographic"):
        _require_polar_stereographic(None)
    # the anchor CRS passes
    _require_polar_stereographic("+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +R=1737400")


# --- integration: model over an ingested synthetic DEM + synthetic SPICE ----------


@pytest.fixture
def illum(synthetic_dem, synthetic_spice, tmp_path):
    """An IlluminationModel over a coarsely-ingested synthetic DEM (SPICE furnished)."""
    product = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    model = IlluminationModel(product, n_azimuth=16, max_radius_m=8000.0, abcorr="NONE")
    return model, synthetic_spice


def test_model_builds_with_horizon_and_manifest(illum) -> None:
    model, _ = illum
    assert model.horizon.shape == (model.height, model.width, 16)
    assert model.horizon.dtype == np.float32
    assert model.illumination_hash.startswith("sha256:")
    manifest = model.to_manifest()
    assert manifest["schema"] == BUNDLE_SCHEMA
    assert manifest["params"]["n_azimuth"] == 16
    assert manifest["terrain_hash"] == model.terrain_hash


def test_hash_is_reproducible_across_builds(synthetic_dem, synthetic_spice, tmp_path) -> None:
    product = ingest_dem(synthetic_dem, tmp_path / "t", resolution_m=2000.0)
    a = IlluminationModel(product, n_azimuth=16, max_radius_m=8000.0, abcorr="NONE")
    b = IlluminationModel(product, n_azimuth=16, max_radius_m=8000.0, abcorr="NONE")
    assert a.illumination_hash == b.illumination_hash


def test_sun_visible_matches_independent_geometry(illum) -> None:
    model, spice = illum
    row, col = model.height // 2, model.width // 2
    x, y = rasterio.transform.xy(model.transform, row, col)
    public = model.sun_visible(float(x), float(y), spice.epoch)

    lon, lat = model._lonlat(float(x), float(y))
    geom = sun_geometry(Site.lunar_from_latlon(lat, lon), spice.epoch, abcorr="NONE")
    world_az = topocentric_to_world_azimuth(geom.azimuth_deg, lon)
    expected = bool(geom.elevation_deg > model.horizon[row, col, azimuth_bin(world_az, 16)])
    assert public == expected


def test_sun_visible_out_of_bounds_raises(illum) -> None:
    model, spice = illum
    with pytest.raises(IlluminationError, match="outside the terrain grid"):
        model.sun_visible(1.0e9, 1.0e9, spice.epoch)


def test_illuminated_mask_shape_and_dtype(illum) -> None:
    model, spice = illum
    mask = model.illuminated_mask(spice.epoch)
    assert mask.shape == (model.height, model.width)
    assert mask.dtype == np.bool_


def test_psr_mask_over_window(illum) -> None:
    model, spice = illum
    result = model.psr_mask(spice.window, 6.0 * 3600.0, semantics=PsrEpochSemantics.MISSION)
    assert isinstance(result, PsrResult)
    assert result.mask.shape == (model.height, model.width)
    assert result.mask.dtype == np.bool_
    assert 0.0 <= result.ever_lit_fraction <= 1.0
    assert result.n_epochs == 4  # 24 h window, 6 h step (half-open)
    assert result.semantics is PsrEpochSemantics.MISSION
    assert result.illumination_hash == model.illumination_hash
    assert result.void_mask.shape == (model.height, model.width)
    # The PSR mask is exactly the complement of the ever-lit set over the sampled epochs.
    ever = np.zeros((model.height, model.width), dtype=np.bool_)
    for epoch in epoch_range(spice.window, 6.0 * 3600.0):
        ever |= model.illuminated_mask(epoch)
    np.testing.assert_array_equal(result.mask, ~ever)


def test_psr_mask_rejects_nonpositive_step(illum) -> None:
    model, spice = illum
    with pytest.raises(SpiceGeometryError, match="step_s must be positive"):
        model.psr_mask(spice.window, 0.0)


def test_from_product_defaults_max_radius(synthetic_dem, synthetic_spice, tmp_path) -> None:
    product = ingest_dem(synthetic_dem, tmp_path / "t", resolution_m=2000.0)
    model = IlluminationModel.from_product(product, n_azimuth=8, abcorr="NONE")
    assert model.max_radius_m > 0.0  # defaulted from the grid extent
    assert model.horizon.shape[2] == 8
