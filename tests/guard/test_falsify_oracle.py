"""The falsification oracle's detection branches + adversary geometry (RM-P1-GUARD-05).

A zero-violation oracle is only trustworthy if it *would* fire on a real breach, so these unit tests
drive synthetic rollout steps that violate each invariant (keep-out entry, a negative core
certificate, a fail-open verdict, an out-of-bounds action) and assert the oracle flags them — the
converse of the full-rollout gate in ``test_falsification.py``.
"""

from __future__ import annotations

import math

import pytest

from astro_mine.guard.falsify import (
    DEFAULT_DT,
    DEFAULT_U_MAX,
    PlantState,
    RolloutStep,
    WorstCaseAdversary,
    anchor_initial_state,
    control_violations,
    keepout_barrier,
    scalar_violations,
    shielded_violations,
)
from astro_mine.guard.falsify.adversary import ANCHOR_SAFE_SIGNALS, _unit_toward_keepout
from astro_mine.guard.models import compile_anchor
from astro_mine.guard.spec.enums import GeometryKind
from astro_mine.guard.spec.ir import CompiledSafetyModel, KeepOutTerm
from tests.guard.conftest import make_verdict

COARSE_PERIOD_S = 120_960.0
_SAFE_POS = (200.0, 200.0, 200.0)  # clear of every anchor keep-out region


@pytest.fixture(scope="module")
def compiled() -> CompiledSafetyModel:
    return compile_anchor(sample_period_s=COARSE_PERIOD_S)


def _term(compiled: CompiledSafetyModel, shape: GeometryKind) -> KeepOutTerm:
    return next(t for t in compiled.keep_out_terms if t.shape == shape)


# --- barriers per geometry -------------------------------------------------------------------


def test_keepout_barrier_all_geometries(compiled: CompiledSafetyModel) -> None:
    sphere = _term(compiled, GeometryKind.SPHERE)  # centre (0,0,0), r=30, margin 3
    box = _term(compiled, GeometryKind.BOX)  # centre (500,-1200,0), half (100,100,20), margin 5
    half = _term(compiled, GeometryKind.HALF_SPACE)  # z >= -1
    # inside → negative; well outside → positive
    assert keepout_barrier(sphere, [0.0, 0.0, 0.0]) == pytest.approx(-33.0)
    assert keepout_barrier(sphere, [100.0, 0.0, 0.0]) == pytest.approx(67.0)
    assert keepout_barrier(box, [500.0, -1200.0, 0.0]) == pytest.approx(-5.0)  # deep inside the box
    assert keepout_barrier(box, [200.0, 200.0, 200.0]) > 0.0
    assert keepout_barrier(half, [0.0, 0.0, 5.0]) == pytest.approx(6.0)  # z=5 → 5+2-1
    assert keepout_barrier(half, [0.0, 0.0, -10.0]) < 0.0  # below the slope edge


def test_scalar_violations_ge_le_and_nan(compiled: CompiledSafetyModel) -> None:
    assert scalar_violations(compiled, dict(ANCHOR_SAFE_SIGNALS)) == []
    # GE floor breached (energy), LE ceiling breached (torque)
    low_energy = {**ANCHOR_SAFE_SIGNALS, "battery_soc_j": 1.0}
    assert "c_energy_floor" in scalar_violations(compiled, low_energy)
    high_torque = {**ANCHOR_SAFE_SIGNALS, "anchor_torque_nm": 999.0}
    assert "c_anchor_torque" in scalar_violations(compiled, high_torque)
    # an unresolvable (missing → NaN) signal counts as violated
    missing = {k: v for k, v in ANCHOR_SAFE_SIGNALS.items() if k != "power_available_w"}
    assert "c_power_floor" in scalar_violations(compiled, missing)


# --- the oracle flags each invariant breach --------------------------------------------------


def _step(index: int, *, position, signals, certified, verdict) -> RolloutStep:  # type: ignore[no-untyped-def]
    return RolloutStep(
        index=index,
        state=PlantState(position=position, velocity=(0.0, 0.0, 0.0), signals=signals),
        certified_action=certified,
        verdict=verdict,
    )


def test_shielded_oracle_flags_keepout_entry(compiled: CompiledSafetyModel) -> None:
    step = _step(
        0,
        position=(0.0, 0.0, 0.0),  # dead centre of the lander-zone sphere
        signals=dict(ANCHOR_SAFE_SIGNALS),
        certified=(0.0, 0.0, 0.0),
        verdict=make_verdict(min_barrier_margin=-33.0),
    )
    kinds = {
        v.kind for v in shielded_violations([step], compiled, u_max=DEFAULT_U_MAX, dt=DEFAULT_DT)
    }
    assert "keep_out" in kinds
    assert "certificate" in kinds  # the core's own margin is negative here too


def test_shielded_oracle_flags_fail_open(compiled: CompiledSafetyModel) -> None:
    # A scalar hard constraint violated, yet the verdict certified the primary action → fail open.
    step = _step(
        0,
        position=_SAFE_POS,
        signals={**ANCHOR_SAFE_SIGNALS, "battery_soc_j": 1.0},
        certified=(0.0, 0.0, 0.0),
        verdict=make_verdict(layer="primary", reason="certified"),
    )
    vs = shielded_violations([step], compiled, u_max=DEFAULT_U_MAX, dt=DEFAULT_DT)
    assert [v.kind for v in vs] == ["fail_open"]


def test_shielded_oracle_accepts_backup_when_scalar_violated(compiled: CompiledSafetyModel) -> None:
    # Same unsafe scalar, but the verdict fell back to a verified backup → not a violation.
    step = _step(
        0,
        position=_SAFE_POS,
        signals={**ANCHOR_SAFE_SIGNALS, "battery_soc_j": 1.0},
        certified=(0.1, 0.0, 0.0),
        verdict=make_verdict(layer="backup", reason="scalar_violated", backup_kind="safe_state"),
    )
    assert shielded_violations([step], compiled, u_max=DEFAULT_U_MAX, dt=DEFAULT_DT) == []


def test_shielded_oracle_flags_unbounded_or_nonfinite_action(compiled: CompiledSafetyModel) -> None:
    over = _step(
        0,
        position=_SAFE_POS,
        signals=dict(ANCHOR_SAFE_SIGNALS),
        certified=(999.0, 0.0, 0.0),
        verdict=make_verdict(),
    )
    nan = _step(
        1,
        position=_SAFE_POS,
        signals=dict(ANCHOR_SAFE_SIGNALS),
        certified=(math.nan, 0.0, 0.0),
        verdict=make_verdict(),
    )
    assert [
        v.kind for v in shielded_violations([over], compiled, u_max=DEFAULT_U_MAX, dt=DEFAULT_DT)
    ] == ["action"]
    assert [
        v.kind for v in shielded_violations([nan], compiled, u_max=DEFAULT_U_MAX, dt=DEFAULT_DT)
    ] == ["action"]


def test_shielded_oracle_clean_step_has_no_violations(compiled: CompiledSafetyModel) -> None:
    step = _step(
        0,
        position=_SAFE_POS,
        signals=dict(ANCHOR_SAFE_SIGNALS),
        certified=(0.5, -0.5, 0.0),
        verdict=make_verdict(layer="primary", reason="certified", min_barrier_margin=10.0),
    )
    assert shielded_violations([step], compiled, u_max=DEFAULT_U_MAX, dt=DEFAULT_DT) == []


def test_control_violations_report_scalar_crossings(compiled: CompiledSafetyModel) -> None:
    step = _step(
        0,
        position=_SAFE_POS,
        signals={**ANCHOR_SAFE_SIGNALS, "chassis_temp_k": 90.0},  # below the 120 K survival floor
        certified=(0.0, 0.0, 0.0),
        verdict=None,
    )
    vs = control_violations([step], compiled, u_max=DEFAULT_U_MAX, dt=DEFAULT_DT)
    assert any(v.kind == "scalar" and "c_thermal_floor" in v.detail for v in vs)


# --- adversary geometry edge cases -----------------------------------------------------------


def test_unit_toward_keepout_half_space_and_degenerate(compiled: CompiledSafetyModel) -> None:
    half = _term(compiled, GeometryKind.HALF_SPACE)  # normal (0,0,1) → toward -z is unsafe
    assert _unit_toward_keepout(half, [0.0, 0.0, 100.0]) == pytest.approx([0.0, 0.0, -1.0])
    sphere = _term(compiled, GeometryKind.SPHERE)
    # position exactly at the sphere centre → degenerate zero-length direction (no NaNs)
    assert _unit_toward_keepout(sphere, [0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_worst_case_action_targets_half_space_when_nearest(compiled: CompiledSafetyModel) -> None:
    adversary = WorstCaseAdversary(compiled)
    # Just above the slope edge and far from the sphere: the half-space is the nearest keep-out, so
    # the worst-case thrusts straight down (-z) at full u_max.
    action = adversary.action(0, [400.0, 0.0, 0.5], [0.0, 0.0, 0.0], 3)
    assert action[2] == pytest.approx(-DEFAULT_U_MAX)


def test_worst_case_without_keepout_terms_is_inert(compiled: CompiledSafetyModel) -> None:
    empty = compiled.model_copy(update={"keep_out_terms": []})
    adversary = WorstCaseAdversary(empty)
    assert adversary.action(0, [10.0, 0.0, 0.0], [0.0, 0.0, 0.0], 3) == [0.0, 0.0, 0.0]


def test_anchor_initial_state_is_inside_the_safe_set(compiled: CompiledSafetyModel) -> None:
    state = anchor_initial_state()
    for term in compiled.keep_out_terms:
        assert keepout_barrier(term, list(state.position)) > 0.0
    assert scalar_violations(compiled, dict(state.signals)) == []
