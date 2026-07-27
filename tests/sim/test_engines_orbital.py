"""RM-P0-SIM-03 — the orbital engine (reduced-order two-body relay propagation).

A circular lunar orbit must stay circular within a tight tolerance over a step (RK4
conserves energy and angular momentum to high order), the propagation is deterministic, the
engine routes behind the waist, and there are no maneuvers (a mode command is the only
actuation).
"""

from __future__ import annotations

import math

from astro_mine.core.messages.enums import ActionKind, ControlMode
from astro_mine.core.messages.model import Action, ActionBatch, ActuatorCommand, ModeCommand
from astro_mine.core.registry import PluginKind
from astro_mine.core.sadf.enums import DeterminismClass, FidelityTier, Regime
from astro_mine.core.units import INERTIAL_J2000
from astro_mine.sim.engines import RegimeEngine
from astro_mine.sim.engines.orbital import (
    ORBITAL_ENGINE_DESCRIPTOR,
    OrbitalEngine,
    orbital_engine_factory,
)
from astro_mine.sim.runtime import AgentSpec, OrbitalDynamics, RngStreams, Scenario

# A ~100 km circular lunar orbit (Moon radius ≈ 1737.4 km).
_MU = 4.902800118e12
_RADIUS_M = 1_837_400.0
_V_CIRC = math.sqrt(_MU / _RADIUS_M)


def _orbit_scenario(**dyn: object) -> Scenario:
    return Scenario(
        name="relay",
        agents=(
            AgentSpec(
                agent_id="relay",
                initial_position_m=(_RADIUS_M, 0.0, 0.0),
                velocity_mps=(0.0, _V_CIRC, 0.0),
                battery_soc_j=1000.0,
                frame=INERTIAL_J2000,
                dynamics=OrbitalDynamics(**dyn),  # type: ignore[arg-type]
            ),
            # a surface rover the orbital engine must NOT pick up
            AgentSpec(agent_id="rover", velocity_mps=(1.0, 0.0, 0.0)),
        ),
    )


def _radius(engine: OrbitalEngine) -> float:
    t = engine.export_coupling_state().by_agent["relay"].pose.translation_m
    return math.sqrt(t.x**2 + t.y**2 + t.z**2)


def _specific_energy(engine: OrbitalEngine) -> float:
    s = engine.export_coupling_state().by_agent["relay"]
    t, v = s.pose.translation_m, s.linear_velocity_mps
    assert v is not None
    r = math.sqrt(t.x**2 + t.y**2 + t.z**2)
    speed_sq = v.x**2 + v.y**2 + v.z**2
    return speed_sq / 2.0 - _MU / r


# --- the engine owns only its regime's agents ------------------------------------


def test_factory_builds_only_orbital_agents() -> None:
    engine = orbital_engine_factory(_orbit_scenario(), RngStreams(0))
    assert set(engine.export_coupling_state().by_agent) == {"relay"}  # rover skipped


# --- it propagates orbitally ------------------------------------------------------


def test_circular_orbit_conserves_radius_and_energy() -> None:
    engine = orbital_engine_factory(_orbit_scenario(substeps=16), RngStreams(0))
    r0, e0 = _radius(engine), _specific_energy(engine)
    for _ in range(10):  # ~10 minutes of a ~2-hour orbit
        engine.advance(60.0)
    r1, e1 = _radius(engine), _specific_energy(engine)
    assert math.isclose(r1, r0, rel_tol=1e-6)  # stays circular
    assert math.isclose(e1, e0, rel_tol=1e-9)  # energy conserved (RK4, high order)


def test_orbit_actually_moves() -> None:
    engine = orbital_engine_factory(_orbit_scenario(), RngStreams(0))
    before = engine.export_coupling_state().by_agent["relay"].pose.translation_m
    engine.advance(120.0)
    after = engine.export_coupling_state().by_agent["relay"].pose.translation_m
    assert after.y > before.y  # swept along +y from the +x apsis


def test_propagation_is_deterministic() -> None:
    a = orbital_engine_factory(_orbit_scenario(), RngStreams(0))
    b = orbital_engine_factory(_orbit_scenario(), RngStreams(0))
    a.advance(300.0)
    b.advance(300.0)
    assert a.export_coupling_state().by_agent["relay"].pose == (
        b.export_coupling_state().by_agent["relay"].pose
    )


def test_station_keeping_drains_battery() -> None:
    engine = orbital_engine_factory(_orbit_scenario(station_keeping_power_w=2.0), RngStreams(0))
    engine.advance(100.0)
    soc = engine.export_coupling_state().by_agent["relay"].battery_soc_j
    assert soc == 1000.0 - 2.0 * 100.0


# --- descriptor + adapter contract ------------------------------------------------


def test_descriptor_declares_proximity_orbit_massmodel() -> None:
    d = ORBITAL_ENGINE_DESCRIPTOR
    assert d.regimes == (Regime.PROXIMITY_ORBIT,)
    assert d.frames == (INERTIAL_J2000,)
    assert d.fidelity.tier is FidelityTier.MASSMODEL
    assert d.determinism_class is DeterminismClass.TOLERANCE
    manifest = d.to_manifest()
    assert manifest.kind is PluginKind.REGIME_ENGINE
    assert manifest.regimes == [Regime.PROXIMITY_ORBIT]


def test_engine_satisfies_regime_engine_protocol() -> None:
    engine = orbital_engine_factory(_orbit_scenario(), RngStreams(0))
    assert isinstance(engine, RegimeEngine)


# --- actuation: a mode command, but no maneuvers ----------------------------------


def test_mode_command_sets_mode() -> None:
    engine = orbital_engine_factory(_orbit_scenario(), RngStreams(0))
    engine.apply_actions(
        ActionBatch(
            actions=[
                Action(
                    agent_id="relay", kind=ActionKind.MODE, mode=ModeCommand(mode="relay_active")
                )
            ]
        )
    )
    assert engine.export_coupling_state().by_agent["relay"].mode == "relay_active"


def test_actuator_command_does_not_maneuver() -> None:
    engine = orbital_engine_factory(_orbit_scenario(), RngStreams(0))
    before = engine.export_coupling_state().by_agent["relay"].linear_velocity_mps
    engine.apply_actions(
        ActionBatch(
            actions=[
                Action(
                    agent_id="relay",
                    kind=ActionKind.ACTUATOR,
                    actuator=ActuatorCommand(
                        target="thruster", control_mode=ControlMode.VELOCITY, setpoint=[9e9, 0, 0]
                    ),
                )
            ]
        )
    )
    # velocity is untouched by actuation — only gravity (advance) changes it.
    assert engine.export_coupling_state().by_agent["relay"].linear_velocity_mps == before


def test_coupling_state_round_trips() -> None:
    engine = orbital_engine_factory(_orbit_scenario(), RngStreams(0))
    engine.advance(60.0)
    snapshot = engine.export_coupling_state()
    engine.advance(60.0)
    engine.import_coupling_state(snapshot)
    restored = engine.export_coupling_state()
    assert restored.by_agent["relay"].pose == snapshot.by_agent["relay"].pose
    assert restored.sim_time_s == snapshot.sim_time_s
