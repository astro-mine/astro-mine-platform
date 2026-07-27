"""Anchor-scenario acceptance: the lunar-polar safety content, end-to-end (RM-P1-GUARD-04).

Drives the reviewed anchor SafetySpec through the untrusted ``models`` adapter (Fleet SADF budgets +
a Worlds terrain/illumination provider) and the ``PolicyShield`` into the trusted Rust core, and
asserts the GUARD-04 acceptance criteria (issue #4; scenario §10; LUNAR-FR-006):

- the power-floor / thermal monitors fire *before* violation and the arbiter hands control to a
  verified night-survival safe behaviour — fail-safe, never fail-open;
- a night-survival energy breach retreats toward the authored charging pose (the distinct
  ``safe_state`` control law), not the raw policy action;
- slope / keep-out probes are corrected inside the Worlds-derived safe set (or fall back);
- constraint inputs resolve from Worlds/Fleet as Core-typed references in a planetary CRS — an
  Earth/inertial provider is rejected, and an unresolved signal fails safe to the backup.
"""

from __future__ import annotations

import math

import pytest

from astro_mine.core.messages.model import ActionBatch
from astro_mine.core.policy.model import DecisionContext
from astro_mine.core.units import ReferenceFrame
from astro_mine.core.units.enums import FrameClass
from astro_mine.core.world.model import IlluminationState
from astro_mine.guard.audit.sink import CollectingSink
from astro_mine.guard.models import (
    SadfBudgets,
    WorldsTerrain,
    anchor_core_config,
    build_anchor_resolver,
    compile_anchor,
)
from astro_mine.guard.wrap import PolicyShield
from tests.guard.conftest import StubPolicy, make_effort_action, make_observation
from tests.guard.models_fixtures import FakeWorldProvider, sadf_document

AGENT = "rover"
#: A coarse period keeps the two 14-day survival monitors' buffers tiny for the fast checks; the
#: retreat-dynamics test uses the authored 1 Hz period (where the retreat prediction is meaningful).
COARSE_PERIOD_S = 120_960.0

#: Signals the resolver reads off the observation (charging_window_active comes from Worlds).
_SAFE_SENSORS = {
    "anchor_torque_nm": 10.0,
    "battery_soc_j": 500_000.0,
    "chassis_temp_k": 250.0,
    "power_available_w": 50.0,
    "traverse_speed_mps": 0.05,
}


def _budgets() -> SadfBudgets:
    return SadfBudgets.from_document(sadf_document())


def _shield(compiled, *, illumination=IlluminationState.LIT, core_config=None) -> tuple:
    terrain = WorldsTerrain(FakeWorldProvider(illumination=illumination), expected_center="MOON")
    resolver = build_anchor_resolver(terrain=terrain, budgets=_budgets())
    sink = CollectingSink()
    shield = PolicyShield(
        StubPolicy(ActionBatch()),
        compiled,
        signal_resolver=resolver,
        sink=sink,
        core_config=core_config,
    )
    return shield, sink


def _decide(shield, sink, obs, action):
    shield._wrapped = StubPolicy(ActionBatch(actions=[action]))
    out = shield.decide({AGENT: obs}, DecisionContext())
    return out.actions[0], sink.verdicts[-1]


# --- constraint inputs are Core-typed planetary references -----------------------------------


def test_earth_or_inertial_provider_is_rejected() -> None:
    # No implicit Earth/inertial frame may supply a constraint input (LUNAR-TR-001).
    inertial = ReferenceFrame(name="J2000", frame_class=FrameClass.INERTIAL, center="MOON")
    with pytest.raises(ValueError, match="body-fixed"):
        WorldsTerrain(FakeWorldProvider(frame=inertial))


# --- certify / correct / fail-safe on the anchor ---------------------------------------------


def test_safe_far_field_action_is_certified() -> None:
    shield, sink = _shield(compile_anchor(sample_period_s=COARSE_PERIOD_S))
    obs = make_observation(AGENT, position=(100.0, 100.0, 100.0), signals=_SAFE_SENSORS)
    out, v = _decide(shield, sink, obs, make_effort_action(AGENT, [0.0, 0.0, 0.0]))
    assert v.layer in ("primary", "shield")
    assert v.reason in ("certified", "shield_corrected")
    assert all(math.isfinite(x) for x in out.actuator.setpoint)


def test_keepout_probe_is_corrected_or_backed_off() -> None:
    shield, sink = _shield(compile_anchor(sample_period_s=COARSE_PERIOD_S))
    # Just outside the lander sphere (33 m), commanding full thrust inward.
    obs = make_observation(AGENT, position=(34.0, 0.0, 0.0), signals=_SAFE_SENSORS)
    out, v = _decide(shield, sink, obs, make_effort_action(AGENT, [-20.0, 0.0, 0.0]))
    assert v.layer in ("shield", "backup")
    assert out.actuator.setpoint != [-20.0, 0.0, 0.0]  # never the raw inward command
    assert all(math.isfinite(x) and abs(x) <= 20.0 + 1e-6 for x in out.actuator.setpoint)


def test_unresolved_sadf_signal_fails_safe_to_backup() -> None:
    # Drop a required sensor (power_available_w): the resolver returns NaN → the core fails the tick
    # closed to a verified backup — never the raw proposal (fail-safe, never fail-open).
    shield, sink = _shield(compile_anchor(sample_period_s=COARSE_PERIOD_S))
    sensors = {k: v for k, v in _SAFE_SENSORS.items() if k != "power_available_w"}
    obs = make_observation(AGENT, position=(100.0, 100.0, 100.0), signals=sensors)
    out, v = _decide(shield, sink, obs, make_effort_action(AGENT, [9.0, 0.0, 0.0]))
    assert v.layer == "backup"
    assert v.reason == "bad_input"
    assert out.actuator.setpoint != [9.0, 0.0, 0.0]


def test_shadow_psr_zeros_the_charging_window() -> None:
    # In a permanently-shadowed region the Worlds illumination is SHADOW ⇒ charging_window_active
    # resolves to 0.0 (a Core-typed Worlds reference driving a safety signal).
    terrain = WorldsTerrain(FakeWorldProvider(illumination=IlluminationState.SHADOW))
    resolver = build_anchor_resolver(terrain=terrain, budgets=_budgets())
    obs = make_observation(AGENT, position=(40.0, 0.0, 0.0), signals=_SAFE_SENSORS)
    values = resolver.resolve(["charging_window_active"], obs)
    assert values[0] == 0.0


# --- predictive monitor fires before violation -----------------------------------------------


def test_thermal_monitor_fires_before_violation() -> None:
    # Chassis temperature trending down toward the 120 K survival floor: the always-monitor's
    # predictive time-to-violation must trip the backup while the temperature is *still above* the
    # floor (before the hard violation), handing control to a verified safe behaviour.
    shield, sink = _shield(compile_anchor(sample_period_s=COARSE_PERIOD_S))
    obs0 = make_observation(AGENT, position=(100.0, 100.0, 100.0), signals=_SAFE_SENSORS)
    _decide(shield, sink, obs0, make_effort_action(AGENT, [0.0, 0.0, 0.0]))  # prime the trend
    fired_before_violation = False
    for temp in (150.0, 140.0, 132.0, 126.0, 123.0):
        sensors = {**_SAFE_SENSORS, "chassis_temp_k": temp}
        obs = make_observation(AGENT, position=(100.0, 100.0, 100.0), signals=sensors)
        _out, v = _decide(shield, sink, obs, make_effort_action(AGENT, [0.0, 0.0, 0.0]))
        if v.layer == "backup" and v.reason == "monitor_fired":
            assert temp > 120.0  # fired predictively, before the hard floor breach
            assert "c_night_thermal_survival" in v.constraint_ids
            fired_before_violation = True
            break
    assert fired_before_violation, "the survival-thermal monitor never fired predictively"


# --- genuine night-survival retreat (the distinct safe_state control law) ---------------------


def test_energy_floor_retreats_toward_charging_pose() -> None:
    # A night-survival energy breach with an authored charging pose: the arbiter routes on_uncertain
    # safe_state to the *verified retreat* control law — a bounded command toward the pose at
    # (60,0,5) from (40,0,0), i.e. +x and +z — not a brake (which, at rest, would be zero) and never
    # the raw policy action. Uses the authored 1 Hz period so the retreat prediction is meaningful.
    shield, sink = _shield(compile_anchor(), core_config=anchor_core_config())
    sensors = {**_SAFE_SENSORS, "battery_soc_j": 100_000.0}  # below the 180 kJ survival floor
    obs = make_observation(
        AGENT, position=(40.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), signals=sensors
    )
    out, v = _decide(shield, sink, obs, make_effort_action(AGENT, [0.0, 0.0, 0.0]))
    assert v.layer == "backup"
    assert v.reason == "scalar_violated"
    assert v.backup_kind == "safe_state"
    assert v.constraint_ids == ["c_energy_floor"]
    action = out.actuator.setpoint
    assert all(math.isfinite(x) and abs(x) <= 20.0 + 1e-6 for x in action)
    assert action != [0.0, 0.0, 0.0]  # a genuine retreat, distinct from brake-to-stop at rest
    assert action[0] > 0.0 and action[2] > 0.0  # steering toward the pose (+x, +z)


def test_energy_floor_without_charging_still_fails_safe() -> None:
    # Same breach in a PSR (no charging window): the retreat still fires safe (backup), never open.
    shield, sink = _shield(
        compile_anchor(), illumination=IlluminationState.SHADOW, core_config=anchor_core_config()
    )
    sensors = {**_SAFE_SENSORS, "battery_soc_j": 100_000.0}
    obs = make_observation(
        AGENT, position=(40.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), signals=sensors
    )
    out, v = _decide(shield, sink, obs, make_effort_action(AGENT, [15.0, 0.0, 0.0]))
    assert v.layer == "backup"
    assert out.actuator.setpoint != [15.0, 0.0, 0.0]
