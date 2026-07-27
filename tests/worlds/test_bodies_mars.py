"""Mars body pack — a new world is a package, not a core change (RM-P1-WORLDS-11).

Validates the body-pack abstraction: the lunar pack is byte-identical to the Phase-0
constants, and Mars ships purely as a plugin — its frames, radius/geoid, point-mass + J2
gravity, thermophysics, and dust model plug into the *unchanged* Worlds machinery, ingest,
provider, and content-addressed bundle. A synthetic Mars kernel set drives one illumination
query to prove the same Environment-API surface serves Mars.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import rasterio
import rasterio.crs
import rasterio.transform
from rasterio.transform import from_bounds

from astro_mine.core.units import MOON, MOON_BODY_FIXED, require_crs
from astro_mine.core.world import IlluminationState, SurfacePoint, check_world_provider
from astro_mine.spice import Site
from astro_mine.worlds.bodies import (
    MARS,
    MARS_GM_M3_S2,
    MARS_PACK,
    MARS_RADIUS_M,
    MOON_PACK,
    BodyPack,
    DustModel,
    mars_geographic_proj4,
)
from astro_mine.worlds.crs import MOON_RADIUS_M
from astro_mine.worlds.provider import DemWorldProvider
from astro_mine.worlds.provider._geometry import (
    MOON_GM_M3_S2,
    gravity_j2,
    point_mass_gravity,
)
from astro_mine.worlds.regolith import build_regolith_field
from astro_mine.worlds.spec import Region, SourceRef, WorldSpec, build_world_bundle
from astro_mine.worlds.terrain import TerrainModel, ingest_dem

# --- body packs ------------------------------------------------------------------


def test_moon_pack_constants() -> None:
    assert MOON_PACK.body == MOON
    assert MOON_PACK.body_fixed_frame == MOON_BODY_FIXED
    assert MOON_PACK.reference_radius_m == MOON_RADIUS_M
    assert MOON_PACK.gm_m3_s2 == MOON_GM_M3_S2
    # The pack's site reproduces the lunar helper exactly.
    assert MOON_PACK.site(-89.0, 0.0) == Site.lunar_from_latlon(-89.0, 0.0)
    # Gravity is point-mass + the GRAIL zonal harmonics (issue #40): the Moon carried j2 = 0.0
    # ("point-mass only, per the Phase-0 model") until worlds.md §12's low-order spherical-harmonic
    # lunar gravity landed. It is a small correction, so it stays close to the point-mass value —
    # but it is no longer identical to it, and it is latitude-dependent. See tests/test_gravity.py
    # for the regression against the published GRAIL coefficients.
    assert MOON_PACK.j2 > 2.0e-4
    pos = (MOON_RADIUS_M, 0.0, 0.0)
    assert MOON_PACK.gravity(pos) != point_mass_gravity(pos)
    assert MOON_PACK.gravity(pos)[2] == pytest.approx(point_mass_gravity(pos)[2], rel=1e-3)


def test_mars_pack_constants() -> None:
    assert MARS_PACK.body == MARS
    assert MARS_PACK.reference_radius_m == MARS_RADIUS_M
    assert MARS_PACK.j2 > 0.0  # a real oblateness
    assert MARS_PACK.default_crs.body == MARS
    assert "mars_equatorial" in MARS_PACK.thermal_classes
    require_crs(MARS_PACK.default_crs)  # a valid planetary CRS


def test_mars_surface_gravity_is_physical() -> None:
    surface = (MARS_RADIUS_M, 0.0, 0.0)  # equator
    g = MARS_PACK.gravity(surface)
    assert abs(g[2]) == pytest.approx(3.71, abs=0.05)  # Mars surface gravity ~3.71 m/s^2


def test_gravity_j2_reduces_to_point_mass_at_zero() -> None:
    pos = (0.0, 0.0, MOON_RADIUS_M + 5000.0)
    assert gravity_j2(pos, gm_m3_s2=MOON_GM_M3_S2, reference_radius_m=MOON_RADIUS_M, j2=0.0) == (
        point_mass_gravity(pos)
    )
    # A non-zero J2 makes gravity latitude-dependent (pole vs equator differ).
    pole = (0.0, 0.0, MARS_RADIUS_M)
    equator = (MARS_RADIUS_M, 0.0, 0.0)
    g_pole = gravity_j2(
        pole, gm_m3_s2=MARS_GM_M3_S2, reference_radius_m=MARS_RADIUS_M, j2=MARS_PACK.j2
    )
    g_eq = gravity_j2(
        equator, gm_m3_s2=MARS_GM_M3_S2, reference_radius_m=MARS_RADIUS_M, j2=MARS_PACK.j2
    )
    assert abs(g_pole[2]) != abs(g_eq[2])


def test_dust_model_seasonal_and_lifting() -> None:
    dust = MARS_PACK.dust
    assert dust.optical_depth(0.25) > dust.optical_depth(0.75)  # dust-storm season peak
    assert dust.is_lifting(5.0) and not dust.is_lifting(0.1)
    assert MOON_PACK.dust.base_optical_depth == 0.0  # airless Moon
    assert not MOON_PACK.dust.is_lifting(1e6)  # never lifts (infinite threshold)


def test_dust_model_is_a_plain_field_plugin() -> None:
    custom = DustModel("x", 0.2, 0.1, 3.0, 1e-9)
    assert isinstance(custom, DustModel)
    assert isinstance(MARS_PACK, BodyPack)


# --- Mars ingest + provider (gravity/terrain/thermal via the same surface) --------


def _mars_dem(path: Path) -> Path:
    """A small Mars-geographic DEM (equatorial band) with an explicit Mars CRS."""
    width, height = 40, 40
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    elev = (500.0 + 3.0 * xx + 2.0 * yy).astype(np.float32)
    crs = rasterio.crs.CRS.from_proj4(mars_geographic_proj4())
    transform = from_bounds(-2.0, -2.0, 2.0, 2.0, width, height)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=float("nan"),
    ) as dst:
        dst.write(elev, 1)
    return path


@pytest.fixture
def mars_terrain(tmp_path):
    dem = _mars_dem(tmp_path / "mola.tif")
    return ingest_dem(
        dem, tmp_path / "mars_terrain", target_crs=MARS_PACK.default_crs, resolution_m=2000.0
    )


def _mars_position(terrain_product):
    cx, cy = rasterio.transform.xy(
        rasterio.transform.Affine(*terrain_product.transform),
        terrain_product.height // 2,
        terrain_product.width // 2,
    )
    geo = rasterio.crs.CRS.from_proj4(mars_geographic_proj4())
    proj = rasterio.crs.CRS.from_proj4(terrain_product.crs.projection)
    lons, lats = rasterio.warp.transform(proj, geo, [cx], [cy])
    lon, lat = math.radians(lons[0]), math.radians(lats[0])
    r = MARS_RADIUS_M
    return (r * math.cos(lat) * math.cos(lon), r * math.cos(lat) * math.sin(lon), r * math.sin(lat))


def test_mars_terrain_ingests_with_mars_crs(mars_terrain) -> None:
    assert mars_terrain.crs.body == MARS
    assert mars_terrain.crs.reference_radius_m == MARS_RADIUS_M
    with rasterio.open(mars_terrain.layers["elevation"]) as ds:
        assert ds.crs == rasterio.crs.CRS.from_proj4(
            mars_terrain.crs.projection
        )  # explicit Mars CRS


def test_mars_provider_serves_gravity_and_terrain(mars_terrain, tmp_path) -> None:
    import rasterio.warp  # noqa: F401 (used by _mars_position)

    build_regolith_field(mars_terrain, tmp_path / "mars_regolith")
    provider = DemWorldProvider.open(
        mars_terrain,
        tmp_path / "mars_regolith",
        body_pack=MARS_PACK,
        n_azimuth=8,
        max_radius_m=8000.0,
        abcorr="NONE",
    )
    check_world_provider(provider)  # honours the Core WorldProvider contract, on Mars
    assert provider.frame.center == MARS
    position = _mars_position(mars_terrain)
    point = provider.sample(position)  # epoch=None -> dark baseline, no kernels needed
    assert isinstance(point, SurfacePoint)
    assert abs(point.gravity[2]) == pytest.approx(3.71, abs=0.1)  # Mars gravity, not lunar 1.62
    assert point.temperature_k == MARS_PACK.shadow_floor_k  # Mars night floor (~150 K)
    assert point.frame.center == MARS


def test_mars_provider_serves_illumination_and_thermal(
    mars_terrain, synthetic_mars_spice, tmp_path
) -> None:
    import rasterio.warp  # noqa: F401

    build_regolith_field(mars_terrain, tmp_path / "mars_regolith")
    provider = DemWorldProvider.open(
        mars_terrain,
        tmp_path / "mars_regolith",
        body_pack=MARS_PACK,
        n_azimuth=8,
        max_radius_m=8000.0,
        abcorr="NONE",
    )
    position = _mars_position(mars_terrain)
    point = provider.sample(position, epoch=synthetic_mars_spice.epoch)
    # The Mars Sun geometry drives illumination + the (Mars-albedo) equilibrium thermal.
    assert point.illumination.state in (IlluminationState.LIT, IlluminationState.SHADOW)
    if point.illumination.state is IlluminationState.LIT:
        assert point.illumination.solar_flux_w_m2 <= 600.0  # Mars insolation << lunar 1361
        assert point.temperature_k > MARS_PACK.shadow_floor_k


# --- Mars WorldSpec + content-addressed bundle -----------------------------------


def test_mars_world_bundle_is_content_addressed(mars_terrain, tmp_path) -> None:
    spec = WorldSpec(
        world_id="mars-test-region",
        crs=MARS_PACK.default_crs,
        region=Region(min_x_m=-1e5, min_y_m=-1e5, max_x_m=1e5, max_y_m=1e5, resolution_m=2000.0),
        source_dem=SourceRef(id="MOLA-synthetic", description="synthetic MOLA stand-in"),
        description="Mars body-pack validation world (RM-P1-WORLDS-11)",
    )
    bundle = build_world_bundle(spec, terrain=mars_terrain, out_dir=tmp_path / "bundle")
    assert bundle.world_hash.startswith("sha256:")
    assert bundle.manifest["crs"]["body"] == MARS
    # Reproducible from the spec + toolchain: a second build hashes identically.
    bundle2 = build_world_bundle(spec, terrain=mars_terrain, out_dir=tmp_path / "bundle2")
    assert bundle2.world_hash == bundle.world_hash
    # The bundled terrain carries the explicit Mars CRS.
    model = TerrainModel.open(mars_terrain)
    assert model.crs.projection == MARS_PACK.default_crs.projection
