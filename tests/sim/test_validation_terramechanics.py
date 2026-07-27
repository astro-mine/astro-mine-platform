"""RM-P0-SIM-10 — terramechanics validation against analytic cases.

Proves the surface engines reproduce the closed form of the models they implement, within an
explicit error budget (sim.md §2.9, §10): the mobility rover's drawbar-pull-limited speed profile,
the granular excavator's rate/mass/energy balance, and (RM-P1-SIM-06) the DEM engine's static
force balance against Newtonian equilibrium.
"""

from __future__ import annotations

import math

from astro_mine.sim.validation import (
    drawbar_pull_speed,
    validate_dem_granular_engine,
    validate_granular_engine,
    validate_mobility_engine,
)

# --- mobility: drawbar-pull-limited acceleration + top-speed cap -------------------------------


def test_drawbar_pull_speed_ramps_then_saturates() -> None:
    # a = traction/mass = 50/100 = 0.5 m/s^2; cap = 20 m/s → saturates at t = 40 s
    assert drawbar_pull_speed(10.0, 50.0, 100.0, 20.0) == 5.0
    assert drawbar_pull_speed(40.0, 50.0, 100.0, 20.0) == 20.0
    assert drawbar_pull_speed(100.0, 50.0, 100.0, 20.0) == 20.0  # capped


def test_mobility_engine_tracks_the_drawbar_pull_profile() -> None:
    report = validate_mobility_engine(
        mass_kg=100.0, max_traction_n=50.0, max_speed_mps=20.0, dt_s=1.0, steps=60, budget=1e-9
    )
    assert report.passed
    assert report.max_error < 1e-9  # the velocity profile matches min(a·t, v_max) exactly


def test_mobility_acceleration_equals_the_traction_limit() -> None:
    # a heavier rover (same traction) accelerates slower — the gate still holds at its own a_max
    report = validate_mobility_engine(
        mass_kg=400.0, max_traction_n=50.0, max_speed_mps=5.0, dt_s=2.0, steps=30, budget=1e-9
    )
    assert report.passed


# --- granular: excavation rate, mass, and energy ----------------------------------------------


def test_granular_engine_excavates_at_the_capped_rate() -> None:
    report = validate_granular_engine(
        regolith_density_kg_m3=1500.0,
        specific_energy_j_per_m3=1.0e5,
        max_dig_rate_m3_s=0.01,
        target_volume_m3=0.5,
        dt_s=1.0,
        steps=80,
        budget=1e-9,
    )
    assert report.passed
    assert report.max_error < 1e-9  # BIT_EXACT: cumulative volume == min(k·rate·dt, target)


def test_granular_stops_at_the_target_volume() -> None:
    # target 0.05 m³ at 0.01 m³/s is reached in 5 steps; the rest of the run holds steady
    report = validate_granular_engine(
        regolith_density_kg_m3=1800.0,
        specific_energy_j_per_m3=2.0e5,
        max_dig_rate_m3_s=0.01,
        target_volume_m3=0.05,
        dt_s=1.0,
        steps=20,
        budget=1e-9,
    )
    assert report.passed
    assert math.isclose(report.budget, 1e-9)


# --- DEM granular: static force balance (Newtonian equilibrium) --------------------------------


def test_dem_settled_bed_balances_its_weight() -> None:
    report = validate_dem_granular_engine(n_particles=60, settle_substeps=1500, bed_width_m=0.5)
    assert report.passed  # Σ floor_normal ≈ N·m·g within the 5% TOLERANCE budget
    assert report.name == "dem-granular-force-balance"


def test_dem_force_balance_fails_a_too_tight_budget() -> None:
    # A ground-truth DEM bed is not bit-exact equilibrium; an unphysically tight budget must fail.
    report = validate_dem_granular_engine(n_particles=40, settle_substeps=400, budget=1e-9)
    assert not report.passed
