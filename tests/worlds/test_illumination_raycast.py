"""Fine on-demand ray-cast illumination (RM-P1-WORLDS-10; worlds.md §8, §11).

Two layers, mirroring the repo's "real pipeline against synthetic inputs" pattern:

- **Pure-kernel tests** drive the engine-neutral ``_raycast`` kernels on hand-built terrain with a
  known skyline and check the fine ray-cast **agrees with the horizon map on unambiguous cells** — a
  deep crater floor stays a PSR under a low Sun, a high Sun reaches it, outer terrain is lit.
- **Integration tests** build a :class:`RayCastIlluminationModel` over an ingested synthetic DEM +
  ``synthetic_spice`` and check it keeps the public API / PSR semantics, is deterministic, folds the
  backend into the hash, and registers as a ``FIELD_MODEL`` field-model plugin — while the default
  horizon backend's hash is left byte-for-byte unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from astro_mine.core.registry import PluginKind, PluginRegistry
from astro_mine.worlds.illumination import (
    DEFAULT_BACKEND,
    RAYCAST_CPU_BACKEND,
    RAYCAST_GPU_BACKEND,
    IlluminationError,
    IlluminationModel,
    PsrEpochSemantics,
    PsrResult,
    RayCastIlluminationModel,
    SunVisibilityModel,
    build_illumination_field_manifest,
    build_illumination_model,
)
from astro_mine.worlds.illumination._horizon import horizon_field, sun_visibility_raster
from astro_mine.worlds.illumination._raycast import (
    raycast_cell_lit,
    raycast_lit_mask,
    raycast_skyline_deg,
)
from astro_mine.worlds.illumination._registry import (
    horizon_field_model,
    raycast_cpu_field_model,
    raycast_gpu_field_model,
)
from astro_mine.worlds.provider import DemWorldProvider
from astro_mine.worlds.regolith import build_regolith_field
from astro_mine.worlds.terrain import ingest_dem

_FLAT = 1.0e12  # an effectively flat planet radius — disables the curvature term
_PX = (1.0, -1.0)  # a north-up 1 m grid: +x = east (col+), +y = north (row-)


# --- pure kernels ----------------------------------------------------------------


def test_raycast_flat_plane_skyline_is_zero() -> None:
    skyline = raycast_skyline_deg(
        np.full((15, 15), 42.0),
        pixel_size_m=_PX,
        azimuth_world_deg=90.0,
        max_radius_m=10.0,
        body_radius_m=_FLAT,
    )
    assert skyline.shape == (15, 15)
    np.testing.assert_allclose(skyline, 0.0, atol=1e-4)


def test_raycast_wall_raises_skyline_in_its_azimuth() -> None:
    elev = np.zeros((21, 21), dtype=np.float64)
    elev[:, 18:] = 100.0  # a tall wall along the east columns
    east = raycast_skyline_deg(
        elev, pixel_size_m=_PX, azimuth_world_deg=90.0, max_radius_m=15.0, body_radius_m=_FLAT
    )
    west = raycast_skyline_deg(
        elev, pixel_size_m=_PX, azimuth_world_deg=270.0, max_radius_m=15.0, body_radius_m=_FLAT
    )
    assert east[10, 10] == pytest.approx(math.degrees(math.atan2(100.0, 8.0)), abs=2.0)
    assert west[10, 10] < 1.0


def test_raycast_agrees_with_horizon_map_on_unambiguous_crater() -> None:
    # The same crater as test_illumination's horizon-map test: the fine ray-cast and the binned
    # horizon map must agree on the unambiguous cells (deep-shadow floor, high-Sun reach, lit rim).
    yy, xx = np.mgrid[0:21, 0:21]
    r = np.hypot(xx - 10, yy - 10)
    crater = np.where((r >= 6) & (r <= 9), 60.0, 0.0).astype(np.float64)
    horizon = horizon_field(
        crater, pixel_size_m=_PX, n_azimuth=72, max_radius_m=15.0, body_radius_m=_FLAT
    )
    for az in (0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0):
        horizon_lit = bool(sun_visibility_raster(horizon, 5.0, az)[10, 10])
        raycast_lit = bool(
            raycast_lit_mask(
                crater,
                pixel_size_m=_PX,
                sun_elevation_deg=5.0,
                sun_azimuth_world_deg=az,
                max_radius_m=15.0,
                body_radius_m=_FLAT,
            )[10, 10]
        )
        assert horizon_lit == raycast_lit is False  # a low Sun never clears the crater rim (PSR)
    # A high Sun reaches the floor under both models; the outer terrain is lit under both.
    assert raycast_lit_mask(
        crater,
        pixel_size_m=_PX,
        sun_elevation_deg=89.0,
        sun_azimuth_world_deg=0.0,
        max_radius_m=15.0,
        body_radius_m=_FLAT,
    )[10, 10]
    assert sun_visibility_raster(horizon, 89.0, 0.0)[10, 10]


def test_raycast_cell_lit_matches_the_mask() -> None:
    elev = np.zeros((21, 21), dtype=np.float64)
    elev[:, 15:] = 40.0
    kwargs = dict(
        pixel_size_m=_PX,
        sun_elevation_deg=10.0,
        sun_azimuth_world_deg=90.0,
        max_radius_m=15.0,
        body_radius_m=_FLAT,
    )
    mask = raycast_lit_mask(elev, **kwargs)  # type: ignore[arg-type]
    for row, col in ((10, 10), (5, 3), (18, 19)):
        assert raycast_cell_lit(elev, row, col, **kwargs) == bool(mask[row, col])  # type: ignore[arg-type]


def test_raycast_skyline_validates_inputs() -> None:
    with pytest.raises(ValueError, match="non-zero"):
        raycast_skyline_deg(
            np.zeros((4, 4)),
            pixel_size_m=(0.0, -1.0),
            azimuth_world_deg=0.0,
            max_radius_m=5.0,
            body_radius_m=_FLAT,
        )
    with pytest.raises(ValueError, match="smaller than one pixel"):
        raycast_skyline_deg(
            np.zeros((4, 4)),
            pixel_size_m=_PX,
            azimuth_world_deg=0.0,
            max_radius_m=0.5,
            body_radius_m=_FLAT,
        )


# --- integration: model over an ingested synthetic DEM + synthetic SPICE ----------


@pytest.fixture
def raycast(synthetic_dem, synthetic_spice, tmp_path):
    product = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    model = RayCastIlluminationModel(product, n_azimuth=16, max_radius_m=8000.0, abcorr="NONE")
    return model, synthetic_spice


def test_raycast_model_backend_and_hash(synthetic_dem, synthetic_spice, tmp_path) -> None:
    product = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    horizon = IlluminationModel(product, n_azimuth=16, max_radius_m=8000.0, abcorr="NONE")
    ray = RayCastIlluminationModel(product, n_azimuth=16, max_radius_m=8000.0, abcorr="NONE")
    assert ray.backend == RAYCAST_CPU_BACKEND
    assert ray.to_manifest()["params"]["backend"] == RAYCAST_CPU_BACKEND
    # The backend is folded into the hash, so selecting it honestly moves the world hash...
    assert ray.illumination_hash != horizon.illumination_hash
    # ...but the horizon LOS map itself is still built and shared (identical to the default model).
    np.testing.assert_array_equal(ray.horizon, horizon.horizon)


def test_raycast_is_deterministic(synthetic_dem, synthetic_spice, tmp_path) -> None:
    product = ingest_dem(synthetic_dem, tmp_path / "t", resolution_m=2000.0)
    a = RayCastIlluminationModel(product, n_azimuth=16, max_radius_m=8000.0, abcorr="NONE")
    b = RayCastIlluminationModel(product, n_azimuth=16, max_radius_m=8000.0, abcorr="NONE")
    assert a.illumination_hash == b.illumination_hash
    np.testing.assert_array_equal(
        a.illuminated_mask(synthetic_spice.epoch), b.illuminated_mask(synthetic_spice.epoch)
    )


def test_raycast_preserves_public_api(raycast) -> None:
    model, spice = raycast
    assert isinstance(model.sun_visible(*_centre(model), spice.epoch), bool)
    lit, elev = model.illumination_at(*_centre(model), spice.epoch)
    assert isinstance(lit, bool) and isinstance(elev, float)
    with pytest.raises(IlluminationError, match="outside the terrain grid"):
        model.sun_visible(1.0e9, 1.0e9, spice.epoch)
    # PSR semantics unchanged: the mask is exactly the complement of the ever-lit union (over the
    # ray-cast masks the inherited psr_mask now routes through).
    result = model.psr_mask(spice.window, 6.0 * 3600.0, semantics=PsrEpochSemantics.MISSION)
    assert isinstance(result, PsrResult)
    from astro_mine.spice import epoch_range

    ever = np.zeros((model.height, model.width), dtype=np.bool_)
    for epoch in epoch_range(spice.window, 6.0 * 3600.0):
        ever |= model.illuminated_mask(epoch)
    np.testing.assert_array_equal(result.mask, ~ever)


def test_raycast_via_factory(synthetic_dem, synthetic_spice, tmp_path) -> None:
    product = ingest_dem(synthetic_dem, tmp_path / "t", resolution_m=2000.0)
    model = build_illumination_model(
        product, backend=RAYCAST_CPU_BACKEND, n_azimuth=16, max_radius_m=8000.0, abcorr="NONE"
    )
    assert isinstance(model, RayCastIlluminationModel)
    assert model.backend == RAYCAST_CPU_BACKEND


def test_factory_horizon_default_is_hash_stable(synthetic_dem, synthetic_spice, tmp_path) -> None:
    # The default factory path must be byte-identical to constructing an IlluminationModel directly:
    # no "backend" key in the manifest, so existing world hashes are unperturbed (RM-P1-WORLDS-10).
    product = ingest_dem(synthetic_dem, tmp_path / "t", resolution_m=2000.0)
    direct = IlluminationModel(product, n_azimuth=16, max_radius_m=8000.0, abcorr="NONE")
    viafac = build_illumination_model(product, n_azimuth=16, max_radius_m=8000.0, abcorr="NONE")
    assert type(viafac) is IlluminationModel
    assert viafac.backend == DEFAULT_BACKEND
    assert "backend" not in viafac.to_manifest()["params"]
    assert viafac.illumination_hash == direct.illumination_hash


def test_unknown_backend_is_rejected(synthetic_dem, tmp_path) -> None:
    product = ingest_dem(synthetic_dem, tmp_path / "t", resolution_m=2000.0)
    with pytest.raises(IlluminationError, match="unknown illumination backend"):
        build_illumination_model(product, backend="bogus", abcorr="NONE")


def test_illumination_field_manifest_declares_field_model(raycast) -> None:
    model, _ = raycast
    manifest = build_illumination_field_manifest(model, name="raycast-illum", version="0.1.0")
    assert manifest.kind == PluginKind.FIELD_MODEL
    assert manifest.core_interfaces == {"world_provider": "0.1.0"}
    assert manifest.license == "Apache-2.0"
    assert manifest.provenance is not None
    assert manifest.provenance.digest == model.illumination_hash
    assert manifest.attributes["backend"] == RAYCAST_CPU_BACKEND
    # The manifest passes the full Core load-time gate (a second backend registers as a plugin).
    PluginRegistry(require_signature=False).register(manifest)


def test_illumination_model_satisfies_backend_protocol(raycast) -> None:
    model, _ = raycast
    backend: SunVisibilityModel = model  # structural: the model IS a SunVisibilityModel
    for attr in ("horizon", "n_azimuth", "transform", "height", "width", "abcorr", "crs"):
        assert hasattr(backend, attr)
    for method in ("sun_visible", "illumination_at", "illuminated_mask", "psr_mask"):
        assert callable(getattr(backend, method))


def test_raycast_kernel_skips_degenerate_zero_step() -> None:
    # Anisotropic pixels: a near-axis ray whose cross-axis step rounds to 0 pixels lands on the same
    # cell and is skipped without error, leaving the flat skyline (mirrors the _horizon test).
    skyline = raycast_skyline_deg(
        np.zeros((5, 5)),
        pixel_size_m=(1.0, -50.0),
        azimuth_world_deg=5.0,
        max_radius_m=2.0,
        body_radius_m=_FLAT,
    )
    np.testing.assert_allclose(skyline, 0.0, atol=1e-6)
    assert raycast_cell_lit(
        np.zeros((5, 5)),
        2,
        2,
        pixel_size_m=(1.0, -50.0),
        sun_elevation_deg=1.0,
        sun_azimuth_world_deg=5.0,
        max_radius_m=2.0,
        body_radius_m=_FLAT,
    )


def test_entry_point_field_models(synthetic_dem, tmp_path) -> None:
    # The `astro_mine.field_models` entry-point callables build the right backend.
    product = ingest_dem(synthetic_dem, tmp_path / "t", resolution_m=2000.0)
    assert type(horizon_field_model(product, n_azimuth=8, abcorr="NONE")) is IlluminationModel
    assert isinstance(
        raycast_cpu_field_model(product, n_azimuth=8, abcorr="NONE"), RayCastIlluminationModel
    )
    # No CuPy here → the GPU entry point degrades to the CPU ray-cast (still labelled raycast_gpu).
    gpu = raycast_gpu_field_model(product, n_azimuth=8, abcorr="NONE")
    assert isinstance(gpu, RayCastIlluminationModel)
    assert gpu.backend == RAYCAST_GPU_BACKEND


def test_provider_open_with_raycast_backend(synthetic_dem, tmp_path) -> None:
    # The WorldSpec's illumination_backend threads through DemWorldProvider.open (RM-P1-WORLDS-10).
    terrain = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    build_regolith_field(terrain, tmp_path / "regolith")
    provider = DemWorldProvider.open(
        terrain,
        tmp_path / "regolith",
        illumination_backend=RAYCAST_CPU_BACKEND,
        n_azimuth=16,
        max_radius_m=8000.0,
        abcorr="NONE",
    )
    assert isinstance(provider.illumination, RayCastIlluminationModel)
    assert provider.illumination.backend == RAYCAST_CPU_BACKEND


def _centre(model: IlluminationModel) -> tuple[float, float]:
    import rasterio.transform

    x, y = rasterio.transform.xy(model.transform, model.height // 2, model.width // 2)
    return float(x), float(y)
