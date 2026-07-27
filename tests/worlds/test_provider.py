"""Environment-API world provider + LOS service (RM-P0-WORLDS-06).

Mirrors the repo's "real pipeline against synthetic inputs" pattern:

- **Pure kernel tests** drive the IO-free ``_geometry`` helpers (gravity, the
  illumination-derived temperature stand-in, topocentric elevation/azimuth) on hand-built
  vectors with known answers.
- **Integration tests** assemble a :class:`DemWorldProvider` over an ingested synthetic DEM
  (+ regolith field), assert it honors the Core ``WorldProvider`` contract, and exercise the
  body-fixed⇄map query surface, the DEM ``ray_intersect``, and the horizon ``line_of_sight``
  (``synthetic_spice`` furnished only where the epoch-dependent illumination is exercised).
"""

from __future__ import annotations

import math

import pytest
import rasterio.transform

from astro_mine.core.units import MOON_BODY_FIXED, FrameClass
from astro_mine.core.world import (
    IlluminationState,
    RegolithParams,
    SurfacePoint,
    check_world_provider,
)
from astro_mine.worlds.bodies import MOON_PACK
from astro_mine.worlds.provider import DemWorldProvider
from astro_mine.worlds.provider._geometry import (
    MOON_GM_M3_S2,
    SHADOW_FLOOR_K,
    SOLAR_CONSTANT_W_M2,
    add_scaled,
    cross,
    equilibrium_temperature,
    norm,
    point_mass_gravity,
    solar_flux,
    topocentric_elevation_azimuth,
    unit,
)
from astro_mine.worlds.regolith import build_regolith_field
from astro_mine.worlds.terrain import ingest_dem

# --- pure kernels ----------------------------------------------------------------


def test_point_mass_gravity() -> None:
    assert point_mass_gravity((0.0, 0.0, 0.0)) == (0.0, 0.0, 0.0)
    radius = 1_737_400.0
    gx, gy, gz = point_mass_gravity((radius, 0.0, 0.0))
    assert (gx, gy) == (0.0, 0.0)  # radial-down in the local surface frame
    assert gz == pytest.approx(-MOON_GM_M3_S2 / radius**2)
    assert -gz == pytest.approx(1.62, abs=0.02)  # lunar surface gravity


def test_solar_flux() -> None:
    assert solar_flux(45.0, lit=False) == 0.0  # in shadow
    assert solar_flux(-5.0, lit=True) == 0.0  # Sun below the local horizontal
    assert solar_flux(90.0, lit=True) == pytest.approx(SOLAR_CONSTANT_W_M2)  # Sun at zenith


def test_equilibrium_temperature() -> None:
    assert equilibrium_temperature(0.0) == SHADOW_FLOOR_K
    assert equilibrium_temperature(-1.0) == SHADOW_FLOOR_K
    sublunar = equilibrium_temperature(SOLAR_CONSTANT_W_M2)
    assert sublunar > SHADOW_FLOOR_K
    assert 350.0 < sublunar < 420.0  # physically plausible peak dayside temperature


def test_unit_and_norm() -> None:
    assert unit((0.0, 0.0, 0.0)) is None
    assert unit((3.0, 0.0, 0.0)) == (1.0, 0.0, 0.0)
    assert norm((3.0, 4.0, 0.0)) == pytest.approx(5.0)


def test_topocentric_elevation_azimuth() -> None:
    radius = 1_737_400.0
    observer = (radius, 0.0, 0.0)  # on the equator at lon 0; local up = +x
    # straight up (radially outward) -> elevation 90 deg
    elev_up, _ = topocentric_elevation_azimuth(observer, (2.0 * radius, 0.0, 0.0))
    assert elev_up == pytest.approx(90.0)
    # toward +z is local north here -> horizontal, azimuth 0
    elev_n, az_n = topocentric_elevation_azimuth(observer, (radius, 0.0, radius))
    assert elev_n == pytest.approx(0.0, abs=1e-9)
    assert az_n == pytest.approx(0.0)
    # toward +y is local east here -> horizontal, azimuth 90
    _, az_e = topocentric_elevation_azimuth(observer, (radius, radius, 0.0))
    assert az_e == pytest.approx(90.0)
    # on the spin axis (a pole), azimuth is undefined and reported as 0
    _, az_pole = topocentric_elevation_azimuth((0.0, 0.0, radius), (0.0, 0.0, 2.0 * radius))
    assert az_pole == 0.0


def test_topocentric_rejects_degenerate_geometry() -> None:
    observer = (1_737_400.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="coincides"):
        topocentric_elevation_azimuth(observer, observer)
    with pytest.raises(ValueError, match="body centre"):
        topocentric_elevation_azimuth((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))


# --- integration: a provider over an ingested synthetic DEM + regolith field -------


@pytest.fixture
def provider(synthetic_dem, tmp_path):
    """A DemWorldProvider over a coarsely-ingested synthetic DEM and its regolith field."""
    product = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    build_regolith_field(product, tmp_path / "regolith")
    return DemWorldProvider.open(
        product, tmp_path / "regolith", n_azimuth=16, max_radius_m=8000.0, abcorr="NONE"
    )


def _surface_position(provider: DemWorldProvider, row: int, col: int):
    """A body-fixed position on the terrain surface at grid cell ``(row, col)``."""
    map_x, map_y = rasterio.transform.xy(provider._transform, row, col)
    map_x, map_y = float(map_x), float(map_y)
    sample = provider.terrain.sample(map_x, map_y)
    lon, lat = provider._map_to_lonlat(map_x, map_y)
    radius = provider._radius_m + float(sample.elevation_m)
    lon_r, lat_r = math.radians(lon), math.radians(lat)
    position = (
        radius * math.cos(lat_r) * math.cos(lon_r),
        radius * math.cos(lat_r) * math.sin(lon_r),
        radius * math.sin(lat_r),
    )
    return position, (map_x, map_y), sample


def _off_grid_position(provider: DemWorldProvider, lat_deg: float, lon_deg: float):
    """A body-fixed surface position at a lat/lon outside the small synthetic DEM patch."""
    radius = provider._radius_m
    lat_r, lon_r = math.radians(lat_deg), math.radians(lon_deg)
    return (
        radius * math.cos(lat_r) * math.cos(lon_r),
        radius * math.cos(lat_r) * math.sin(lon_r),
        radius * math.sin(lat_r),
    )


def test_frame_is_body_fixed(provider) -> None:
    assert provider.frame == MOON_BODY_FIXED


def test_conformance_at_default_and_on_surface(provider) -> None:
    # The Core contract test, both at its body-centre default and a real surface point.
    check_world_provider(provider)
    position, _, _ = _surface_position(provider, provider._height // 2, provider._width // 2)
    check_world_provider(provider, position=position)


def test_sample_on_surface_matches_models(provider) -> None:
    position, (map_x, map_y), sample = _surface_position(
        provider, provider._height // 2, provider._width // 2
    )
    assert not math.isnan(sample.elevation_m)  # the centre cell is not a void
    point = provider.sample(position)

    assert isinstance(point, SurfacePoint)
    assert point.frame.frame_class is FrameClass.TOPOCENTRIC
    assert point.elevation_m == pytest.approx(sample.elevation_m)
    assert point.surface_normal == sample.normal
    gx, gy, gz = point.gravity
    assert (gx, gy) == (0.0, 0.0)
    # Point-mass + the GRAIL zonal harmonics the lunar pack now carries (issue #40) — so no longer
    # exactly GM/r^2, but within the ~6e-4 the J2 oblateness term contributes at these latitudes.
    assert gz == pytest.approx(MOON_PACK.gravity(position)[2])
    assert gz == pytest.approx(-MOON_GM_M3_S2 / norm(position) ** 2, rel=1e-3)
    assert point.regolith == provider.regolith.params(map_x, map_y)
    # No epoch -> the static dark baseline.
    assert point.illumination.state is IlluminationState.SHADOW
    assert point.illumination.solar_flux_w_m2 == 0.0
    assert point.temperature_k == SHADOW_FLOOR_K


def test_sample_off_grid_returns_default(provider) -> None:
    point = provider.sample(_off_grid_position(provider, -80.0, 45.0))
    assert point.elevation_m == 0.0
    assert point.surface_normal == (0.0, 0.0, 1.0)
    assert point.regolith == RegolithParams()
    assert point.illumination.state is IlluminationState.SHADOW
    assert point.temperature_k == SHADOW_FLOOR_K
    _, _, gz = point.gravity
    assert gz < 0.0  # radial gravity still resolves off-grid


def test_sample_at_body_centre_returns_default(provider) -> None:
    point = provider.sample((0.0, 0.0, 0.0))
    assert point.gravity == (0.0, 0.0, 0.0)  # undefined at the centre
    assert point.regolith == RegolithParams()
    assert point.temperature_k == SHADOW_FLOOR_K


def test_sample_with_epoch_is_consistent_with_illumination(
    synthetic_dem, synthetic_spice, tmp_path
) -> None:
    product = ingest_dem(synthetic_dem, tmp_path / "t", resolution_m=2000.0)
    build_regolith_field(product, tmp_path / "r")
    provider = DemWorldProvider.open(
        product, tmp_path / "r", n_azimuth=16, max_radius_m=8000.0, abcorr="NONE"
    )
    epoch = synthetic_spice.epoch
    cells = [
        (provider._height // 2, provider._width // 2),
        (1, 1),
        (provider._height - 2, provider._width - 2),
        (provider._height // 2, 1),
        (1, provider._width // 2),
    ]
    for row, col in cells:
        position, _, _ = _surface_position(provider, row, col)
        point = provider.sample(position, epoch=epoch)
        if point.illumination.state is IlluminationState.LIT:
            assert point.illumination.solar_flux_w_m2 > 0.0
            assert point.temperature_k > SHADOW_FLOOR_K
        else:
            assert point.illumination.state is IlluminationState.SHADOW
            assert point.illumination.solar_flux_w_m2 == 0.0
            assert point.temperature_k == SHADOW_FLOOR_K


def test_sample_is_deterministic(provider) -> None:
    position, _, _ = _surface_position(provider, provider._height // 2, provider._width // 2)
    assert provider.sample(position) == provider.sample(position)


def test_ray_intersect_hits_surface_from_above(provider) -> None:
    position, _, sample = _surface_position(provider, provider._height // 2, provider._width // 2)
    up = unit(position)
    assert up is not None
    origin = add_scaled(position, up, 1000.0)  # 1 km above the surface
    downward = (-up[0], -up[1], -up[2])
    hit = provider.ray_intersect(origin, downward)
    assert hit is not None
    assert norm(hit) == pytest.approx(provider._radius_m + sample.elevation_m, abs=2.0)
    offset = (hit[0] - position[0], hit[1] - position[1], hit[2] - position[2])
    assert norm(offset) < 5.0  # the hit is essentially the surface point below the origin


def test_ray_intersect_misses_pointing_away(provider) -> None:
    position, _, _ = _surface_position(provider, provider._height // 2, provider._width // 2)
    up = unit(position)
    assert up is not None
    assert provider.ray_intersect(position, up) is None  # straight out to space


def test_ray_intersect_zero_direction_returns_none(provider) -> None:
    position, _, _ = _surface_position(provider, provider._height // 2, provider._width // 2)
    assert provider.ray_intersect(position, (0.0, 0.0, 0.0)) is None


def test_ray_intersect_leaving_grid_returns_none(provider) -> None:
    position, _, _ = _surface_position(provider, provider._height // 2, provider._width // 2)
    up = unit(position)
    assert up is not None
    origin = add_scaled(position, up, 5000.0)  # high above the grid
    tangent = unit(cross(position, (0.0, 0.0, 1.0)))  # horizontal -> exits the grid laterally
    assert tangent is not None
    assert provider.ray_intersect(origin, tangent) is None


def test_line_of_sight_zenith_is_visible(provider) -> None:
    position, _, _ = _surface_position(provider, provider._height // 2, provider._width // 2)
    up = unit(position)
    assert up is not None
    overhead = add_scaled(position, up, 1.0e6)
    assert provider.line_of_sight(position, overhead) is True


def test_line_of_sight_below_horizon_is_blocked(provider) -> None:
    position, _, _ = _surface_position(provider, provider._height // 2, provider._width // 2)
    up = unit(position)
    assert up is not None
    below = add_scaled(position, up, -100.0)  # straight down, elevation -90 deg
    assert provider.line_of_sight(position, below) is False


def test_line_of_sight_body_centre_observer_is_false(provider) -> None:
    assert provider.line_of_sight((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)) is False


def test_line_of_sight_off_grid_observer_is_false(provider) -> None:
    observer = _off_grid_position(provider, -80.0, 45.0)
    assert provider.line_of_sight(observer, (1.0, 0.0, 0.0)) is False


def test_line_of_sight_coincident_target_is_false(provider) -> None:
    position, _, _ = _surface_position(provider, provider._height // 2, provider._width // 2)
    assert provider.line_of_sight(position, position) is False


# --- injectable thermal source ---------------------------------------------------


class _RecordingThermal:
    """A stub ThermalSource: records its call and returns a fixed temperature."""

    def __init__(self, value_k: float) -> None:
        self.value_k = value_k
        self.calls: list[dict[str, float]] = []

    def temperature_k(
        self, *, map_x: float, map_y: float, epoch: object, solar_flux_w_m2: float
    ) -> float:
        self.calls.append({"map_x": map_x, "map_y": map_y, "solar_flux_w_m2": solar_flux_w_m2})
        return self.value_k


def test_injected_thermal_source_overrides_temperature(
    synthetic_dem, synthetic_spice, tmp_path
) -> None:
    product = ingest_dem(synthetic_dem, tmp_path / "t", resolution_m=2000.0)
    build_regolith_field(product, tmp_path / "r")
    thermal = _RecordingThermal(231.5)
    provider = DemWorldProvider.open(
        product, tmp_path / "r", n_azimuth=16, max_radius_m=8000.0, abcorr="NONE", thermal=thermal
    )
    position, (map_x, map_y), _ = _surface_position(
        provider, provider._height // 2, provider._width // 2
    )
    point = provider.sample(position, epoch=synthetic_spice.epoch)
    # The injected source replaces the coarse equilibrium placeholder.
    assert point.temperature_k == 231.5
    assert thermal.calls  # it was consulted, with the cell coords and the computed flux
    call = thermal.calls[-1]
    assert call["map_x"] == pytest.approx(map_x)
    assert call["map_y"] == pytest.approx(map_y)
    assert call["solar_flux_w_m2"] >= 0.0


def test_thermal_source_not_consulted_without_epoch(synthetic_dem, tmp_path) -> None:
    product = ingest_dem(synthetic_dem, tmp_path / "t", resolution_m=2000.0)
    build_regolith_field(product, tmp_path / "r")
    thermal = _RecordingThermal(231.5)
    provider = DemWorldProvider.open(
        product, tmp_path / "r", n_azimuth=16, max_radius_m=8000.0, abcorr="NONE", thermal=thermal
    )
    position, _, _ = _surface_position(provider, provider._height // 2, provider._width // 2)
    point = provider.sample(position)  # no epoch -> the source needs one, so it is not consulted
    assert point.temperature_k == SHADOW_FLOOR_K
    assert thermal.calls == []
