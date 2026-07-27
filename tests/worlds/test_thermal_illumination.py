"""Illumination-driven per-cell surface thermal (RM-P1-WORLDS-13).

- **Unit tests** drive the forcing + solver against a *fake* illumination model (scripted
  Sun visibility), so the physics — a PSR cell settling to the shadow floor, a lit cell
  tracking real insolation, periodicity, caching, determinism — is exercised without SPICE.
- **Integration test** wires a real :class:`IlluminationThermalSource` into a
  :class:`DemWorldProvider` over an ingested synthetic DEM + ``synthetic_spice`` and confirms
  the provider returns the per-cell curve instead of the RM-P0-WORLDS-06 equilibrium stand-in.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import rasterio.transform
from affine import Affine

from astro_mine.core.units import Epoch, TimeScale
from astro_mine.worlds.illumination import IlluminationModel
from astro_mine.worlds.provider import DemWorldProvider
from astro_mine.worlds.provider._geometry import SHADOW_FLOOR_K, equilibrium_temperature
from astro_mine.worlds.regolith import RegolithField, build_regolith_field
from astro_mine.worlds.terrain import TerrainModel, ingest_dem
from astro_mine.worlds.thermal import (
    TERRAIN_CLASSES,
    DiurnalCurve,
    IlluminationThermalSource,
    illumination_insolation,
    solve_illumination_driven_curve,
)

_EPOCH0 = Epoch(tdb_seconds=0.0, scale=TimeScale.TDB)
_PERIOD = 1000.0  # a short synthetic diurnal period keeps the unit tests fast

# Tiny solver settings so the column solve is fast in unit tests (not fully converged).
_FAST = {"n_layers": 10, "n_phase": 12, "substeps": 2, "max_periods": 6, "depth_skin_depths": 5.0}


class FakeIllum:
    """A minimal illumination stand-in: scripted per-cell Sun visibility + elevation."""

    def __init__(self, lit_fn) -> None:
        self._lit_fn = lit_fn
        self.transform = Affine(1.0, 0.0, 0.0, 0.0, -1.0, 0.0)
        self.illumination_hash = "sha256:fake"

    def illumination_at(self, x: float, y: float, epoch: Epoch) -> tuple[bool, float]:
        return self._lit_fn(x, y, epoch)


def _daylit(x: float, y: float, epoch: Epoch) -> tuple[bool, float]:
    """A cell lit for the first half of the period; Sun elevation follows a half-sine arc."""
    phase = (epoch.tdb_seconds / _PERIOD) % 1.0
    if phase < 0.5:
        return True, 60.0 * np.sin(np.pi * phase / 0.5)
    return False, -10.0


def _dark(x: float, y: float, epoch: Epoch) -> tuple[bool, float]:
    return False, -30.0  # a permanently shadowed cell — never lit


# --- forcing ---------------------------------------------------------------------


def test_insolation_zero_for_permanently_shadowed_cell() -> None:
    flux = illumination_insolation(
        FakeIllum(_dark),
        0.0,
        0.0,
        reference_epoch=_EPOCH0,
        albedo=0.12,
        period_s=_PERIOD,
        n_samples=16,
    )
    assert all(flux(p / 20) == 0.0 for p in range(20))


def test_insolation_positive_and_periodic_for_lit_cell() -> None:
    flux = illumination_insolation(
        FakeIllum(_daylit),
        0.0,
        0.0,
        reference_epoch=_EPOCH0,
        albedo=0.12,
        period_s=_PERIOD,
        n_samples=24,
    )
    assert flux(0.25) > 0.0  # midday, Sun high
    assert flux(0.75) == 0.0  # night
    assert flux(0.0) == pytest.approx(flux(1.0))  # periodic wraparound


# --- column solve ----------------------------------------------------------------


def test_psr_cell_settles_near_shadow_floor() -> None:
    curve = solve_illumination_driven_curve(
        FakeIllum(_dark),
        0.0,
        0.0,
        TERRAIN_CLASSES["crater_floor"],
        reference_epoch=_EPOCH0,
        period_s=_PERIOD,
        n_samples=12,
        **_FAST,
    )
    assert isinstance(curve, DiurnalCurve)
    assert curve.params["forcing"] == "illumination_driven"
    # No direct insolation -> the whole curve sits at the deep-shadow (environment-flux) floor.
    assert curve.peak_k < 80.0
    assert curve.night_floor_k > 20.0


def test_lit_cell_is_warmer_than_a_psr_cell() -> None:
    lit = solve_illumination_driven_curve(
        FakeIllum(_daylit),
        0.0,
        0.0,
        TERRAIN_CLASSES["polar_lit"],
        reference_epoch=_EPOCH0,
        period_s=_PERIOD,
        n_samples=12,
        **_FAST,
    )
    dark = solve_illumination_driven_curve(
        FakeIllum(_dark),
        0.0,
        0.0,
        TERRAIN_CLASSES["polar_lit"],
        reference_epoch=_EPOCH0,
        period_s=_PERIOD,
        n_samples=12,
        **_FAST,
    )
    assert lit.peak_k > dark.peak_k + 50.0  # real insolation drives a real diurnal swing


def test_curve_is_deterministic() -> None:
    kwargs = dict(reference_epoch=_EPOCH0, period_s=_PERIOD, n_samples=12, **_FAST)
    a = solve_illumination_driven_curve(
        FakeIllum(_daylit), 0.0, 0.0, TERRAIN_CLASSES["polar_lit"], **kwargs
    )
    b = solve_illumination_driven_curve(
        FakeIllum(_daylit), 0.0, 0.0, TERRAIN_CLASSES["polar_lit"], **kwargs
    )
    assert a.thermal_hash == b.thermal_hash


# --- ThermalSource ---------------------------------------------------------------


def test_source_caches_and_interpolates() -> None:
    source = IlluminationThermalSource(
        FakeIllum(_daylit),
        TERRAIN_CLASSES["polar_lit"],
        reference_epoch=_EPOCH0,
        period_s=_PERIOD,
        n_samples=12,
        **_FAST,
    )
    first = source.curve_at(5.0, 5.0)
    assert source.curve_at(5.0, 5.0) is first  # same grid cell -> cached identity
    noon = source.temperature_k(
        map_x=5.0,
        map_y=5.0,
        epoch=Epoch(tdb_seconds=0.25 * _PERIOD, scale=TimeScale.TDB),
        solar_flux_w_m2=1.0,
    )
    midnight = source.temperature_k(
        map_x=5.0,
        map_y=5.0,
        epoch=Epoch(tdb_seconds=0.75 * _PERIOD, scale=TimeScale.TDB),
        solar_flux_w_m2=0.0,
    )
    assert noon > midnight  # warmer under the modelled midday insolation


# --- integration: wired into DemWorldProvider ------------------------------------


def test_provider_uses_illumination_driven_thermal(
    synthetic_dem, synthetic_spice, tmp_path
) -> None:
    product = ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)
    build_regolith_field(product, tmp_path / "regolith")
    terrain = TerrainModel.open(product)
    illum = IlluminationModel(product, n_azimuth=16, max_radius_m=8000.0, abcorr="NONE")
    regolith = RegolithField.open(tmp_path / "regolith")
    source = IlluminationThermalSource(
        illum,
        TERRAIN_CLASSES["polar_lit"],
        reference_epoch=synthetic_spice.epoch,
        period_s=synthetic_spice.window.end.tdb_seconds - synthetic_spice.epoch.tdb_seconds,
        n_samples=6,
        **_FAST,
    )
    driven = DemWorldProvider(terrain, illum, regolith, thermal=source)

    x, y = rasterio.transform.xy(illum.transform, illum.height // 2, illum.width // 2)
    lon, lat = illum._lonlat(float(x), float(y))
    r = float(product.crs.reference_radius_m)
    position = (
        r * math.cos(math.radians(lat)) * math.cos(math.radians(lon)),
        r * math.cos(math.radians(lat)) * math.sin(math.radians(lon)),
        r * math.sin(math.radians(lat)),
    )
    driven_point = driven.sample(position, epoch=synthetic_spice.epoch)
    # The provider routes surface temperature through the injected source — so the returned
    # value is exactly the cell's illumination-forced curve, not the coarse equilibrium stand-in.
    expected = source.temperature_k(
        map_x=float(x), map_y=float(y), epoch=synthetic_spice.epoch, solar_flux_w_m2=0.0
    )
    assert driven_point.temperature_k == pytest.approx(expected)
    assert driven_point.temperature_k >= SHADOW_FLOOR_K - 1.0  # physically floored
    assert equilibrium_temperature(0.0) == SHADOW_FLOOR_K  # the P0 placeholder is unchanged
