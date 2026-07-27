"""Per-cell topocentric horizon maps (RM-P1-WORLDS-12).

Two layers, mirroring the repo's "real pipeline against synthetic inputs" pattern:

- **Pure-kernel tests** drive the engine-neutral ``_topocentric`` array kernels on hand-built
  geometry with a known answer (a flat plane, a single wall, agreement with the scalar
  ``provider._geometry`` topocentric formula).
- **Integration tests** build a TOPOCENTRIC :class:`IlluminationModel` over an ingested
  synthetic DEM + the ``synthetic_spice`` kernels and check it (a) matches SPICE topocentric
  geometry with **no** grid-convergence correction, (b) keeps the public API/PSR semantics,
  (c) generalizes to a non-polar CRS the grid frame rejects, and (d) reports a P0→P1 budget.
"""

from __future__ import annotations

import numpy as np
import pytest
import rasterio.transform

from astro_mine.core.units import MOON, MOON_BODY_FIXED, PlanetaryCRS
from astro_mine.spice import Site, sun_geometry
from astro_mine.worlds.crs import MOON_RADIUS_M
from astro_mine.worlds.illumination import (
    HorizonFrame,
    IlluminationError,
    IlluminationModel,
    PsrEpochSemantics,
)
from astro_mine.worlds.illumination._horizon import azimuth_bin
from astro_mine.worlds.illumination._topocentric import (
    body_fixed_positions,
    horizon_frame_delta,
    topocentric_elevation_azimuth_grid,
    topocentric_horizon_field,
    topocentric_horizon_hash,
)
from astro_mine.worlds.provider._geometry import topocentric_elevation_azimuth
from astro_mine.worlds.terrain import ingest_dem

_PX = (1.0, -1.0)  # 1 m north-up grid


# --- body_fixed_positions --------------------------------------------------------


def test_body_fixed_positions_places_cells_on_the_sphere() -> None:
    lon = np.array([[0.0, 90.0]])
    lat = np.array([[0.0, 90.0]])
    elev = np.array([[0.0, 10.0]])
    pos = body_fixed_positions(lon, lat, elev, 1000.0)
    assert pos.shape == (1, 2, 3)
    np.testing.assert_allclose(pos[0, 0], (1000.0, 0.0, 0.0), atol=1e-6)  # lat0 lon0
    np.testing.assert_allclose(pos[0, 1], (0.0, 0.0, 1010.0), atol=1e-6)  # north pole + 10 m


def test_body_fixed_positions_propagates_voids() -> None:
    pos = body_fixed_positions(np.zeros((1, 1)), np.zeros((1, 1)), np.array([[np.nan]]), 1000.0)
    assert np.isnan(pos).all()


def test_body_fixed_positions_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="share a shape"):
        body_fixed_positions(np.zeros((2, 2)), np.zeros((2, 3)), np.zeros((2, 2)), 1000.0)


# --- topocentric_elevation_azimuth_grid ------------------------------------------


def test_elaz_grid_matches_scalar_geometry() -> None:
    obs = np.array([[[1000.0, 0.0, 0.0], [0.0, 1000.0, 200.0]]])
    tgt = np.array([[[1000.0, 50.0, 30.0], [80.0, 1200.0, 260.0]]])
    elev, az = topocentric_elevation_azimuth_grid(obs, tgt)
    for j in range(2):
        s_elev, s_az = topocentric_elevation_azimuth(tuple(obs[0, j]), tuple(tgt[0, j]))
        assert elev[0, j] == pytest.approx(s_elev, abs=1e-9)
        assert az[0, j] == pytest.approx(s_az, abs=1e-9)


def test_elaz_grid_nan_for_degenerate_geometry() -> None:
    obs = np.array([[[1000.0, 0.0, 0.0], [0.0, 0.0, 1000.0]]])  # 2nd is on the spin axis
    tgt = np.array([[[1000.0, 0.0, 0.0], [10.0, 0.0, 1000.0]]])  # 1st coincides with obs
    elev, az = topocentric_elevation_azimuth_grid(obs, tgt)
    assert np.isnan(elev[0, 0]) and np.isnan(az[0, 0])  # coincident
    assert np.isnan(az[0, 1])  # pole azimuth undefined


# --- topocentric_horizon_field ---------------------------------------------------


def _flat_positions(n: int, radius: float) -> np.ndarray:
    """A tiny tangent patch on a huge sphere — effectively a flat plane."""
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float64)
    lon = (xx - n / 2) * 1e-4  # ~metre spacing at this radius
    lat = -89.0 + (yy - n / 2) * 1e-4
    return body_fixed_positions(lon, lat, np.zeros((n, n)), radius)


def test_flat_patch_horizon_is_near_zero() -> None:
    pos = _flat_positions(11, 1.0e9)  # near-flat: huge radius
    horizon = topocentric_horizon_field(pos, pixel_size_m=_PX, n_azimuth=16, max_radius_m=5.0)
    assert horizon.shape == (11, 11, 16)
    assert horizon.dtype == np.float32
    assert float(np.nanmax(horizon)) < 1.0  # no terrain rises above the local horizontal


def test_wall_raises_topocentric_horizon() -> None:
    # A tall ridge one metre away raises the skyline sharply in the cell's local frame.
    n = 9
    pos = _flat_positions(n, 1.0e9)
    pos[:, n // 2 + 1 :, :] *= 1.0  # keep geometry; add a vertical wall via radius bump
    # Add 50 m of elevation to the eastern columns by pushing them radially outward.
    up = pos / np.linalg.norm(pos, axis=-1, keepdims=True)
    pos[:, n // 2 + 1 :, :] += up[:, n // 2 + 1 :, :] * 50.0
    horizon = topocentric_horizon_field(pos, pixel_size_m=_PX, n_azimuth=72, max_radius_m=6.0)
    center_max = float(horizon[n // 2, n // 2].max())
    assert center_max > 30.0  # a 50 m wall ~1-2 m away subtends a steep skyline


def test_topocentric_horizon_field_validates_inputs() -> None:
    pos = _flat_positions(4, 1.0e6)
    with pytest.raises(ValueError, match=r"must be \(H, W, 3\)"):
        topocentric_horizon_field(pos[..., 0], pixel_size_m=_PX, n_azimuth=4, max_radius_m=2.0)
    with pytest.raises(ValueError, match="n_azimuth must be positive"):
        topocentric_horizon_field(pos, pixel_size_m=_PX, n_azimuth=0, max_radius_m=2.0)
    with pytest.raises(ValueError, match="non-zero"):
        topocentric_horizon_field(pos, pixel_size_m=(0.0, -1.0), n_azimuth=4, max_radius_m=2.0)
    with pytest.raises(ValueError, match="smaller than one pixel"):
        topocentric_horizon_field(pos, pixel_size_m=_PX, n_azimuth=4, max_radius_m=0.5)


def test_topocentric_horizon_hash_is_deterministic_and_sensitive() -> None:
    horizon = np.zeros((4, 4, 8), dtype=np.float32)
    meta = {"schema": "x", "n": 1}
    assert topocentric_horizon_hash(horizon, meta) == topocentric_horizon_hash(horizon, meta)
    assert topocentric_horizon_hash(horizon, meta).startswith("sha256:")
    other = horizon.copy()
    other[0, 0, 0] = 1.0
    assert topocentric_horizon_hash(other, meta) != topocentric_horizon_hash(horizon, meta)
    assert topocentric_horizon_hash(horizon, {"schema": "y"}) != topocentric_horizon_hash(
        horizon, meta
    )


# --- horizon_frame_delta ---------------------------------------------------------


def test_horizon_frame_delta_zero_for_identical_and_keys() -> None:
    horizon = np.random.default_rng(0).random((6, 6, 12)).astype(np.float32)
    lat = np.full((6, 6), -88.0)
    delta = horizon_frame_delta(horizon, horizon, lat)
    assert delta["max_abs_deg"] == 0.0
    assert set(delta) == {"max_abs_deg", "mean_abs_deg", "max_abs_deg_high_lat"}


def test_horizon_frame_delta_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shapes differ"):
        horizon_frame_delta(
            np.zeros((2, 2, 3), np.float32), np.zeros((2, 2, 4), np.float32), np.zeros((2, 2))
        )


# --- integration: TOPOCENTRIC IlluminationModel over a real ingested product ------


@pytest.fixture
def topo_model(synthetic_dem, synthetic_spice, tmp_path):
    product = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    model = IlluminationModel(
        product,
        n_azimuth=16,
        max_radius_m=8000.0,
        abcorr="NONE",
        horizon_frame=HorizonFrame.TOPOCENTRIC,
    )
    return model, synthetic_spice


def test_topocentric_model_builds(topo_model) -> None:
    model, _ = topo_model
    assert model.horizon_frame is HorizonFrame.TOPOCENTRIC
    assert model.horizon.shape == (model.height, model.width, 16)
    assert model.illumination_hash.startswith("sha256:")
    assert model.to_manifest()["params"]["horizon_frame"] == "topocentric"


def test_sun_visible_uses_raw_topocentric_azimuth(topo_model) -> None:
    # The public verdict must equal an independent SPICE topocentric evaluation with NO
    # grid-convergence correction (the whole point of RM-P1-WORLDS-12).
    model, spice = topo_model
    row, col = model.height // 2, model.width // 2
    x, y = rasterio.transform.xy(model.transform, row, col)
    lon, lat = model._lonlat(float(x), float(y))
    geom = sun_geometry(Site.lunar_from_latlon(lat, lon), spice.epoch, abcorr="NONE")
    expected = bool(geom.elevation_deg > model.horizon[row, col, azimuth_bin(geom.azimuth_deg, 16)])
    assert model.sun_visible(float(x), float(y), spice.epoch) == expected


def test_topocentric_masks_and_psr(topo_model) -> None:
    from astro_mine.spice import epoch_range

    model, spice = topo_model
    mask = model.illuminated_mask(spice.epoch)
    assert mask.shape == (model.height, model.width) and mask.dtype == np.bool_
    psr = model.psr_mask(spice.window, 6.0 * 3600.0, semantics=PsrEpochSemantics.MISSION)
    assert psr.mask.shape == (model.height, model.width)
    # PSR mask is exactly the complement of the ever-lit union over the sampled epochs.
    ever = np.zeros((model.height, model.width), dtype=np.bool_)
    for epoch in epoch_range(spice.window, 6.0 * 3600.0):
        ever |= model.illuminated_mask(epoch)
    np.testing.assert_array_equal(psr.mask, ~ever)


def test_frame_delta_reports_budget(topo_model) -> None:
    model, _ = topo_model
    budget = model.frame_delta()
    assert budget["max_abs_deg"] >= budget["mean_abs_deg"] >= 0.0
    assert "max_abs_deg_high_lat" in budget


def test_frame_delta_requires_topocentric(synthetic_dem, synthetic_spice, tmp_path) -> None:
    product = ingest_dem(synthetic_dem, tmp_path / "t", resolution_m=2000.0)
    grid_model = IlluminationModel(product, n_azimuth=8, max_radius_m=8000.0, abcorr="NONE")
    with pytest.raises(IlluminationError, match="TOPOCENTRIC"):
        grid_model.frame_delta()


def test_topocentric_generalizes_to_non_polar_crs(synthetic_dem, synthetic_spice, tmp_path) -> None:
    # An equidistant-cylindrical lunar CRS: the GRID frame rejects it, TOPOCENTRIC accepts it.
    eqc = PlanetaryCRS(
        body=MOON,
        body_fixed_frame=MOON_BODY_FIXED.name,
        reference_radius_m=MOON_RADIUS_M,
        projection=f"+proj=eqc +lat_ts=0 +lon_0=0 +R={MOON_RADIUS_M:.1f} +units=m +no_defs",
    )
    product = ingest_dem(synthetic_dem, tmp_path / "eqc", target_crs=eqc, resolution_m=2000.0)
    with pytest.raises(IlluminationError, match="south-polar stereographic"):
        IlluminationModel(product, n_azimuth=8, max_radius_m=8000.0, abcorr="NONE")
    model = IlluminationModel(
        product,
        n_azimuth=8,
        max_radius_m=8000.0,
        abcorr="NONE",
        horizon_frame=HorizonFrame.TOPOCENTRIC,
    )
    assert model.sun_visible(
        *rasterio.transform.xy(model.transform, 1, 1), synthetic_spice.epoch
    ) in (
        True,
        False,
    )
