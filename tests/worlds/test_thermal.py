"""Surface thermal (first-cut) - 1-D thermophysical diurnal curves (RM-P0-WORLDS-04).

Two layers, mirroring the repo's pattern:

- **Pure kernel tests** drive the IO-free ``_solver`` helpers (skin depth, layer grid,
  insolation profile, deterministic hash) with hand-checked answers.
- **Model tests** solve the per-terrain-class diurnal curves and gate on physically plausible
  ranges (subsolar peak vs radiative equilibrium; permanently-shadowed floor at tens of K;
  sunlit night floor sustained by conduction), plus convergence, determinism, interpolation,
  and the unknown-class error path.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from astro_mine.worlds.thermal import (
    BUNDLE_SCHEMA,
    TERRAIN_CLASSES,
    DiurnalCurve,
    ThermalClass,
    ThermalError,
    diurnal_curve,
    solve_diurnal_curve,
)
from astro_mine.worlds.thermal._solver import (
    SOLAR_CONSTANT_W_M2,
    STEFAN_BOLTZMANN_W_M2_K4,
    build_layers,
    diurnal_hash,
    insolation_flux,
    skin_depth,
    solve_surface_curve,
    thermal_conductivity,
)

# Fast solver settings for the model tests (coarse grid; still converges to periodic state).
_FAST = {"n_phase": 24, "substeps": 4, "n_layers": 18, "max_periods": 60}


# --- pure kernels ----------------------------------------------------------------


def test_thermal_conductivity_from_inertia() -> None:
    # k = I^2 / (rho c)
    assert thermal_conductivity(100.0, 1500.0, 700.0) == pytest.approx(100.0**2 / (1500.0 * 700.0))


def test_skin_depth_matches_formula() -> None:
    rho, c, period = 1500.0, 700.0, 2.55e6
    k = thermal_conductivity(55.0, rho, c)
    kappa = k / (rho * c)
    expected = math.sqrt(kappa * period / math.pi)
    assert skin_depth(55.0, rho, c, period) == pytest.approx(expected)
    # Lunar diurnal skin depth is centimetres-scale.
    assert 0.01 < skin_depth(55.0, rho, c, period) < 0.2


def test_build_layers_geometry() -> None:
    thickness, centers = build_layers(20, 1.0, 1.2)
    assert thickness.shape == (20,) and centers.shape == (20,)
    assert thickness.sum() == pytest.approx(1.0)  # sums to the requested depth
    assert thickness[0] < thickness[-1]  # thin at the surface, thick at depth
    assert np.all(np.diff(centers) > 0.0)  # monotonically deeper


def test_insolation_flux_profile() -> None:
    # Noon at phase 0.5: absorbed (1-A) S at a 90 deg peak elevation.
    assert insolation_flux(0.5, 90.0, 0.12) == pytest.approx((1.0 - 0.12) * SOLAR_CONSTANT_W_M2)
    # Midnight is dark.
    assert insolation_flux(0.0, 90.0, 0.12) == 0.0
    # A permanently shadowed floor (0 deg peak) is dark at all phases.
    assert insolation_flux(0.5, 0.0, 0.12) == 0.0
    # A low grazing Sun delivers a smaller noon flux than an overhead one.
    assert 0.0 < insolation_flux(0.5, 12.0, 0.12) < insolation_flux(0.5, 90.0, 0.12)


def test_diurnal_hash_is_deterministic_and_content_sensitive() -> None:
    temps = np.array([100.0, 200.0, 300.0])
    meta = {"schema": "x", "n": 1}
    assert diurnal_hash(temps, meta) == diurnal_hash(temps, meta)
    assert diurnal_hash(temps, meta).startswith("sha256:")
    assert diurnal_hash(temps + 1.0, meta) != diurnal_hash(temps, meta)
    assert diurnal_hash(temps, {"schema": "y"}) != diurnal_hash(temps, meta)


def test_solve_surface_curve_crater_floor_is_cold_and_flat() -> None:
    # No insolation: the column relaxes to the lumped-environment radiative-equilibrium floor.
    cls = TERRAIN_CLASSES["crater_floor"]
    _phases, temps, converged, periods = solve_surface_curve(
        thermal_inertia_tiu=cls.thermal_inertia_tiu,
        density_kg_m3=cls.density_kg_m3,
        specific_heat_j_kg_k=cls.specific_heat_j_kg_k,
        albedo=cls.albedo,
        emissivity=cls.emissivity,
        peak_sun_elevation_deg=cls.peak_sun_elevation_deg,
        environment_flux_w_m2=cls.environment_flux_w_m2,
        **_FAST,
    )
    assert converged and periods >= 1
    floor = (cls.environment_flux_w_m2 / (cls.emissivity * STEFAN_BOLTZMANN_W_M2_K4)) ** 0.25
    np.testing.assert_allclose(temps, floor, atol=2.0)  # flat near the env-flux floor
    assert 30.0 < float(temps.min()) < 60.0  # PSR floor ~tens of K


# --- per-terrain-class diurnal curves --------------------------------------------


@pytest.fixture(scope="module")
def equatorial() -> DiurnalCurve:
    return solve_diurnal_curve(TERRAIN_CLASSES["equatorial"], **_FAST)


def test_equatorial_peak_tracks_radiative_equilibrium(equatorial) -> None:
    cls = TERRAIN_CLASSES["equatorial"]
    subsolar_eq = (
        (1.0 - cls.albedo) * SOLAR_CONSTANT_W_M2 / (cls.emissivity * STEFAN_BOLTZMANN_W_M2_K4)
    ) ** 0.25
    # Daytime peak approaches the radiative-equilibrium subsolar value (conduction draws it down).
    assert equatorial.converged
    assert 0.85 * subsolar_eq < equatorial.peak_k <= subsolar_eq + 1.0
    assert 350.0 < equatorial.peak_k < 400.0


def test_equatorial_night_floor_is_plausible(equatorial) -> None:
    # The 14-day night floor is sustained by substrate conduction, well above the radiative floor.
    assert 70.0 < equatorial.night_floor_k < 130.0
    assert equatorial.night_floor_k < equatorial.peak_k


def test_class_temperature_ordering() -> None:
    eq = solve_diurnal_curve(TERRAIN_CLASSES["equatorial"], **_FAST)
    polar = solve_diurnal_curve(TERRAIN_CLASSES["polar_lit"], **_FAST)
    crater = solve_diurnal_curve(TERRAIN_CLASSES["crater_floor"], **_FAST)
    # Overhead sun > grazing polar sun > permanently shadowed floor.
    assert eq.peak_k > polar.peak_k > crater.peak_k


def test_diurnal_curve_metadata_and_interpolation() -> None:
    curve = solve_diurnal_curve(TERRAIN_CLASSES["equatorial"], **_FAST)
    assert curve.terrain_class == "equatorial"
    assert curve.thermal_hash.startswith("sha256:")
    assert curve.params["name"] == "equatorial"
    assert curve.phases.shape == curve.temperatures_k.shape == (_FAST["n_phase"],)
    assert curve.period_s > 0.0
    # temperature_at hits the samples and wraps around.
    assert curve.temperature_at(0.0) == pytest.approx(float(curve.temperatures_k[0]))
    assert curve.temperature_at(1.0) == pytest.approx(curve.temperature_at(0.0))
    assert curve.night_floor_k <= curve.temperature_at(0.37) <= curve.peak_k


def test_solve_is_deterministic() -> None:
    a = solve_diurnal_curve(TERRAIN_CLASSES["crater_floor"], **_FAST)
    b = solve_diurnal_curve(TERRAIN_CLASSES["crater_floor"], **_FAST)
    assert a.thermal_hash == b.thermal_hash
    np.testing.assert_array_equal(a.temperatures_k, b.temperatures_k)


def test_bundle_schema_constant() -> None:
    # The schema string is carried in the hashed provenance metadata.
    assert BUNDLE_SCHEMA == "astro-mine-worlds/thermal/v0.1"


def test_diurnal_curve_registry_lookup_is_cached() -> None:
    first = diurnal_curve("crater_floor")
    second = diurnal_curve("crater_floor")
    assert first is second  # lru_cache returns the same object
    assert isinstance(first, DiurnalCurve)
    assert 30.0 < first.night_floor_k < 60.0


def test_diurnal_curve_unknown_class_raises() -> None:
    with pytest.raises(ThermalError, match="unknown terrain class"):
        diurnal_curve("atlantis")


def test_thermal_class_is_frozen() -> None:
    cls = TERRAIN_CLASSES["equatorial"]
    assert isinstance(cls, ThermalClass)
    with pytest.raises(AttributeError):
        cls.albedo = 0.5  # type: ignore[misc]
