"""RM-P0-SIM-10 — orbital regression against the analytic two-body (Kepler) oracle.

Proves the in-CI, dependency-free half of the orbital validation (sim.md §2.9, §10): the analytic
universal-variable propagator is correct (period closure, time-reversal), the engine's RK4
propagation matches it and conserves the two-body invariants within an explicit error budget, and
the budget is a real gate. The live STK/GMAT external cross-check lives in
``test_validation_orbital_gmat.py``.
"""

from __future__ import annotations

import math

import pytest

from astro_mine.sim.engines._vecmath import norm
from astro_mine.sim.validation import (
    engine_positions_at_times,
    kepler_propagate,
    specific_angular_momentum,
    specific_energy,
    validate_against_oracle,
    validate_orbital_conservation,
    validate_orbital_engine,
)

_MU = 4.902800118e12  # Moon GM (m^3/s^2)
_R = 1_837_400.0  # ~100 km circular lunar orbit radius
_V_CIRC = math.sqrt(_MU / _R)
_PERIOD = 2.0 * math.pi * math.sqrt(_R**3 / _MU)
_R0 = (_R, 0.0, 0.0)
_V0 = (0.0, _V_CIRC, 0.0)
_STEPS = round(_PERIOD / 10.0)  # one orbit at a 10 s step


# --- the analytic oracle is correct -----------------------------------------------------------


def test_kepler_closes_the_orbit_after_one_period() -> None:
    r, v = kepler_propagate(_R0, _V0, _MU, _PERIOD)
    assert math.dist(r, _R0) < 1e-6  # back to the start (sub-micron over a 100 km orbit)
    assert math.dist(v, _V0) < 1e-9


def test_kepler_quarter_period_is_a_quarter_turn() -> None:
    r, _ = kepler_propagate(_R0, _V0, _MU, _PERIOD / 4.0)
    assert math.isclose(r[1], _R, rel_tol=1e-9)  # +x rotates to +y
    assert abs(r[0]) < 1e-3


def test_kepler_is_time_reversible_for_a_hyperbolic_orbit() -> None:
    v_escape = math.sqrt(2.0 * _MU / _R)
    v0 = (0.0, 1.4 * v_escape, 0.0)  # unbound (alpha < 0 → the z<0 Stumpff branch)
    r1, v1 = kepler_propagate(_R0, v0, _MU, 600.0)
    r0, _ = kepler_propagate(r1, v1, _MU, -600.0)
    assert math.dist(r0, _R0) < 1e-3


def test_kepler_tiny_step_is_near_identity() -> None:
    r, v = kepler_propagate(_R0, _V0, _MU, 1e-4)  # |z| ~ 0 → the Stumpff series branch
    assert math.dist(r, _R0) < 1.0
    assert math.dist(v, _V0) < 1e-3


def test_kepler_zero_step_is_the_identity() -> None:
    assert kepler_propagate(_R0, _V0, _MU, 0.0) == (_R0, _V0)


def test_kepler_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="mu must be positive"):
        kepler_propagate(_R0, _V0, 0.0, 10.0)
    with pytest.raises(ValueError, match="non-zero"):
        kepler_propagate((0.0, 0.0, 0.0), _V0, _MU, 10.0)


def test_invariants_match_a_circular_orbit() -> None:
    # circular: energy = -mu/(2r), |h| = r * v_circ
    assert math.isclose(specific_energy(_R0, _V0, _MU), -_MU / (2.0 * _R), rel_tol=1e-12)
    assert math.isclose(norm(specific_angular_momentum(_R0, _V0)), _R * _V_CIRC, rel_tol=1e-12)


# --- the engine matches the oracle within budget ----------------------------------------------


def test_orbital_engine_matches_kepler_over_one_orbit() -> None:
    report = validate_orbital_engine(_R0, _V0, _MU, dt_s=10.0, steps=_STEPS, budget=1e-9)
    assert report.passed
    assert report.max_error < 1e-9  # in practice ~1e-13 at 8 substeps


def test_orbital_engine_conserves_the_two_body_invariants() -> None:
    report = validate_orbital_conservation(_R0, _V0, _MU, dt_s=10.0, steps=_STEPS, budget=1e-9)
    assert report.passed
    assert report.max_error < 1e-9


def test_finer_substeps_reduce_the_error() -> None:
    coarse = validate_orbital_engine(_R0, _V0, _MU, dt_s=10.0, steps=_STEPS, substeps=2, budget=1.0)
    fine = validate_orbital_engine(_R0, _V0, _MU, dt_s=10.0, steps=_STEPS, substeps=16, budget=1.0)
    assert fine.max_error < coarse.max_error  # RK4 converges as the substep shrinks


def test_the_budget_is_a_real_gate() -> None:
    # an impossibly tight budget the (already very accurate) engine cannot meet must fail
    report = validate_orbital_engine(_R0, _V0, _MU, dt_s=10.0, steps=_STEPS, budget=1e-18)
    assert not report.passed


def test_engine_positions_at_irregular_times_match_kepler() -> None:
    # the seam the live GMAT regression uses: sample the engine at an oracle's (irregular) report
    # epochs. At a realistic ~60 s cadence the 8-substep RK4 tracks Kepler to ~1e-9.
    times = [0.0, 31.0, 60.0, 95.0, 150.0, 205.0, 260.0, 318.0, 360.0]
    actual = engine_positions_at_times(_R0, _V0, _MU, times)
    reference = [kepler_propagate(_R0, _V0, _MU, t)[0] for t in times]
    assert actual[0] == _R0  # t=0 returns the initial state, no advance
    assert validate_against_oracle(actual, reference, budget=1e-7).passed


# --- the shared comparator --------------------------------------------------------------------


def test_validate_against_oracle_flags_a_divergent_trajectory() -> None:
    actual = [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    reference = [(1.0, 0.0, 0.0), (2.2, 0.0, 0.0)]  # 10% off at the second sample
    report = validate_against_oracle(actual, reference, budget=0.05)
    assert not report.passed
    assert math.isclose(report.max_error, 0.2 / 2.2, rel_tol=1e-9)
    assert validate_against_oracle(actual, reference, budget=0.2).passed


def test_validate_against_oracle_absolute_mode() -> None:
    report = validate_against_oracle([(0.0,)], [(0.0,)], budget=0.0, relative=False)
    assert report.passed and report.max_error == 0.0


def test_validate_against_oracle_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="budget must be non-negative"):
        validate_against_oracle([(0.0,)], [(0.0,)], budget=-1.0)
    with pytest.raises(ValueError, match="differ in length"):
        validate_against_oracle([(0.0,)], [(0.0,), (1.0,)], budget=1.0)
    with pytest.raises(ValueError, match="at least one sample"):
        validate_against_oracle([], [], budget=1.0)
