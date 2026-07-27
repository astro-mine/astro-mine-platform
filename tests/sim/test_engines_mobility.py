"""RM-P0-SIM-03 — the mobility engine (mid-fidelity wheeled rover).

A rover tracks a velocity command or a goto target with a terramechanics-limited
acceleration and a top-speed cap, drains battery with speed, routes behind the waist, and
propagates deterministically.
"""

from __future__ import annotations

import math

from astro_mine.core.messages.enums import ActionKind, ControlMode, TaskKind
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    ActuatorCommand,
    GotoTask,
    ModeCommand,
    Quat,
    TaskDirective,
    Transform,
    Vec3,
)
from astro_mine.core.registry import PluginKind
from astro_mine.core.sadf.enums import DeterminismClass, FidelityTier, Regime
from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.sim.engines import RegimeEngine
from astro_mine.sim.engines.mobility import (
    MOBILITY_ENGINE_DESCRIPTOR,
    MobilityEngine,
    mobility_engine_factory,
)
from astro_mine.sim.runtime import AgentSpec, MobilityDynamics, RngStreams, Scenario


def _scenario(**dyn: object) -> Scenario:
    params: dict[str, object] = {
        "mass_kg": 100.0,
        "max_speed_mps": 20.0,
        "max_traction_n": 50.0,  # a_max = 0.5 m/s²
    }
    params.update(dyn)
    return Scenario(
        name="rovers",
        agents=(
            AgentSpec(
                agent_id="rover",
                battery_soc_j=1000.0,
                dynamics=MobilityDynamics(**params),  # type: ignore[arg-type]
            ),
            AgentSpec(agent_id="other"),  # kinematic default — must be skipped
        ),
    )


def _velocity_cmd(agent_id: str, vx: float, vy: float, vz: float) -> Action:
    return Action(
        agent_id=agent_id,
        kind=ActionKind.ACTUATOR,
        actuator=ActuatorCommand(
            target="base", control_mode=ControlMode.VELOCITY, setpoint=[vx, vy, vz]
        ),
    )


def _goto(agent_id: str, x: float, y: float, z: float) -> Action:
    return Action(
        agent_id=agent_id,
        kind=ActionKind.TASK,
        task=TaskDirective(
            task_kind=TaskKind.GOTO,
            goto=GotoTask(
                target_frame="MOON_ME",
                target_pose=Transform(
                    translation_m=Vec3(x=x, y=y, z=z),
                    rotation_quat_xyzw=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
            ),
        ),
    )


def _speed(engine: MobilityEngine) -> float:
    v = engine.export_coupling_state().by_agent["rover"].linear_velocity_mps
    assert v is not None
    return math.sqrt(v.x**2 + v.y**2 + v.z**2)


def test_factory_builds_only_mobility_agents() -> None:
    engine = mobility_engine_factory(_scenario(), RngStreams(0))
    assert set(engine.export_coupling_state().by_agent) == {"rover"}


def test_traction_caps_acceleration() -> None:
    engine = mobility_engine_factory(_scenario(), RngStreams(0))
    engine.apply_actions(ActionBatch(actions=[_velocity_cmd("rover", 10.0, 0.0, 0.0)]))
    engine.advance(1.0)  # a_max = 50/100 = 0.5 m/s² over 1 s
    assert math.isclose(_speed(engine), 0.5, rel_tol=1e-9)  # not the full 10 m/s


def test_speed_is_capped_at_max_speed() -> None:
    engine = mobility_engine_factory(
        _scenario(max_speed_mps=2.0, max_traction_n=1.0e6), RngStreams(0)
    )
    engine.apply_actions(ActionBatch(actions=[_velocity_cmd("rover", 100.0, 0.0, 0.0)]))
    engine.advance(1.0)
    assert math.isclose(_speed(engine), 2.0, rel_tol=1e-9)  # clamped, despite huge traction


def test_goto_traverses_toward_the_target() -> None:
    engine = mobility_engine_factory(_scenario(max_speed_mps=2.0), RngStreams(0))
    engine.apply_actions(ActionBatch(actions=[_goto("rover", 1000.0, 0.0, 0.0)]))
    for _ in range(20):
        engine.advance(1.0)
    pos = engine.export_coupling_state().by_agent["rover"].pose.translation_m
    assert pos.x > 0.0 and pos.x <= 2.0 * 20  # advanced, never exceeding max speed * time
    assert pos.y == 0.0 and pos.z == 0.0


def test_velocity_command_overrides_goto() -> None:
    engine = mobility_engine_factory(_scenario(max_traction_n=1.0e6), RngStreams(0))
    engine.apply_actions(ActionBatch(actions=[_goto("rover", 0.0, 1000.0, 0.0)]))
    engine.apply_actions(ActionBatch(actions=[_velocity_cmd("rover", 3.0, 0.0, 0.0)]))
    engine.advance(1.0)
    v = engine.export_coupling_state().by_agent["rover"].linear_velocity_mps
    assert v is not None and v.x > 0.0 and v.y == 0.0  # follows the velocity command, not +y


def test_battery_draw_scales_with_speed() -> None:
    engine = mobility_engine_factory(
        _scenario(
            idle_power_w=1.0, drive_power_w_per_mps=2.0, max_traction_n=1.0e6, max_speed_mps=5.0
        ),
        RngStreams(0),
    )
    engine.apply_actions(ActionBatch(actions=[_velocity_cmd("rover", 5.0, 0.0, 0.0)]))
    engine.advance(1.0)  # reaches 5 m/s instantly (huge traction); draw = (1 + 2·5)·1 = 11 J
    soc = engine.export_coupling_state().by_agent["rover"].battery_soc_j
    assert soc is not None and math.isclose(soc, 1000.0 - 11.0, rel_tol=1e-9)


def test_mode_command_sets_mode() -> None:
    engine = mobility_engine_factory(_scenario(), RngStreams(0))
    engine.apply_actions(
        ActionBatch(
            actions=[
                Action(agent_id="rover", kind=ActionKind.MODE, mode=ModeCommand(mode="traverse"))
            ]
        )
    )
    assert engine.export_coupling_state().by_agent["rover"].mode == "traverse"


def test_propagation_is_deterministic() -> None:
    a = mobility_engine_factory(_scenario(), RngStreams(0))
    b = mobility_engine_factory(_scenario(), RngStreams(0))
    for engine in (a, b):
        engine.apply_actions(ActionBatch(actions=[_goto("rover", 50.0, 50.0, 0.0)]))
        for _ in range(5):
            engine.advance(1.0)
    assert a.export_coupling_state().by_agent["rover"].pose == (
        b.export_coupling_state().by_agent["rover"].pose
    )


def test_descriptor_declares_surface_kinematic() -> None:
    d = MOBILITY_ENGINE_DESCRIPTOR
    assert d.regimes == (Regime.SURFACE,)
    assert d.frames == (MOON_BODY_FIXED,)
    assert d.fidelity.tier is FidelityTier.KINEMATIC
    assert d.determinism_class is DeterminismClass.TOLERANCE
    assert d.to_manifest().kind is PluginKind.REGIME_ENGINE


def test_engine_satisfies_regime_engine_protocol() -> None:
    assert isinstance(mobility_engine_factory(_scenario(), RngStreams(0)), RegimeEngine)


def test_coupling_state_round_trips() -> None:
    engine = mobility_engine_factory(_scenario(), RngStreams(0))
    engine.apply_actions(ActionBatch(actions=[_velocity_cmd("rover", 1.0, 0.0, 0.0)]))
    engine.advance(1.0)
    snapshot = engine.export_coupling_state()
    engine.advance(1.0)
    engine.import_coupling_state(snapshot)
    assert engine.export_coupling_state().by_agent["rover"].pose == snapshot.by_agent["rover"].pose


def test_goto_at_current_position_holds_still() -> None:
    engine = mobility_engine_factory(_scenario(), RngStreams(0))
    engine.apply_actions(ActionBatch(actions=[_goto("rover", 0.0, 0.0, 0.0)]))  # already at origin
    engine.advance(1.0)
    assert _speed(engine) == 0.0  # zero distance to target → zero desired velocity
