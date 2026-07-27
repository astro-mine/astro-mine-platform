"""RM-P0-SIM-03 — the manipulation engine (reduced-order articulated linkage).

Forward kinematics maps the joint configuration to the tool-tip pose; joints track a
position setpoint under their rate and joint limits; actuation draws battery only while
moving; the engine routes behind the waist and is deterministic.
"""

from __future__ import annotations

import math

from astro_mine.core.messages.enums import ActionKind, ControlMode
from astro_mine.core.messages.model import Action, ActionBatch, ActuatorCommand, ModeCommand
from astro_mine.core.registry import PluginKind
from astro_mine.core.sadf.enums import DeterminismClass, FidelityTier, JointType, Regime
from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.sim.engines import RegimeEngine
from astro_mine.sim.engines.manipulation import (
    MANIPULATION_ENGINE_DESCRIPTOR,
    ManipulationEngine,
    manipulation_engine_factory,
)
from astro_mine.sim.runtime import (
    AgentSpec,
    JointSpec,
    ManipulationDynamics,
    RngStreams,
    Scenario,
)


def _arm_scenario(joints: tuple[JointSpec, ...], **dyn: object) -> Scenario:
    return Scenario(
        name="arm",
        agents=(
            AgentSpec(
                agent_id="arm",
                battery_soc_j=1000.0,
                dynamics=ManipulationDynamics(joints=joints, **dyn),  # type: ignore[arg-type]
            ),
            AgentSpec(agent_id="other"),  # kinematic default — skipped
        ),
    )


def _revolute(rate_limit: float = 10.0, **kw: object) -> JointSpec:
    return JointSpec(
        name="j", joint_type=JointType.REVOLUTE, link_length_m=1.0, rate_limit=rate_limit, **kw
    )  # type: ignore[arg-type]


def _position_cmd(*targets: float) -> ActionBatch:
    return ActionBatch(
        actions=[
            Action(
                agent_id="arm",
                kind=ActionKind.ACTUATOR,
                actuator=ActuatorCommand(
                    target="arm", control_mode=ControlMode.POSITION, setpoint=list(targets)
                ),
            )
        ]
    )


def _tip(engine: ManipulationEngine) -> tuple[float, float, float]:
    t = engine.export_coupling_state().by_agent["arm"].pose.translation_m
    return (t.x, t.y, t.z)


def test_factory_builds_only_manipulation_agents() -> None:
    engine = manipulation_engine_factory(_arm_scenario((_revolute(),)), RngStreams(0))
    assert set(engine.export_coupling_state().by_agent) == {"arm"}


def test_initial_forward_kinematics_places_the_tip() -> None:
    engine = manipulation_engine_factory(_arm_scenario((_revolute(),)), RngStreams(0))
    x, y, z = _tip(engine)  # one unit link along +x at q=0
    assert math.isclose(x, 1.0) and math.isclose(y, 0.0, abs_tol=1e-12) and z == 0.0


def test_revolute_joint_swings_the_tip() -> None:
    engine = manipulation_engine_factory(_arm_scenario((_revolute(),)), RngStreams(0))
    engine.apply_actions(_position_cmd(math.pi / 2))
    engine.advance(1.0)  # rate_limit 10 ≥ π/2, so the joint reaches the target
    x, y, z = _tip(engine)  # +x link rotated 90° about +z → +y
    assert math.isclose(x, 0.0, abs_tol=1e-9) and math.isclose(y, 1.0) and z == 0.0


def test_prismatic_joint_extends_the_tip() -> None:
    prismatic = JointSpec(
        name="p",
        joint_type=JointType.PRISMATIC,
        axis=(1.0, 0.0, 0.0),
        link_length_m=0.0,
        rate_limit=10.0,
    )
    engine = manipulation_engine_factory(_arm_scenario((prismatic,)), RngStreams(0))
    engine.apply_actions(_position_cmd(2.0))
    engine.advance(1.0)
    assert math.isclose(_tip(engine)[0], 2.0)


def test_rate_limit_bounds_joint_motion() -> None:
    engine = manipulation_engine_factory(_arm_scenario((_revolute(rate_limit=0.1),)), RngStreams(0))
    engine.apply_actions(_position_cmd(math.pi / 2))
    engine.advance(1.0)  # only 0.1 rad of motion is allowed
    assert math.isclose(_tip(engine)[0], math.cos(0.1), rel_tol=1e-9)


def test_joint_limits_clamp_the_setpoint() -> None:
    engine = manipulation_engine_factory(
        _arm_scenario((_revolute(lower=0.0, upper=0.5),)), RngStreams(0)
    )
    engine.apply_actions(_position_cmd(10.0))  # well past the upper limit
    engine.advance(1.0)
    assert math.isclose(_tip(engine)[0], math.cos(0.5), rel_tol=1e-9)  # clamped at 0.5 rad


def test_actuation_draws_battery_only_while_moving() -> None:
    engine = manipulation_engine_factory(
        _arm_scenario((_revolute(),), actuation_power_w=4.0), RngStreams(0)
    )
    engine.advance(1.0)  # setpoint == initial config → idle → no draw
    assert engine.export_coupling_state().by_agent["arm"].battery_soc_j == 1000.0
    engine.apply_actions(_position_cmd(0.2))
    engine.advance(1.0)  # now moving → draw 4 J
    assert engine.export_coupling_state().by_agent["arm"].battery_soc_j == 996.0


def test_mode_command_sets_mode() -> None:
    engine = manipulation_engine_factory(_arm_scenario((_revolute(),)), RngStreams(0))
    engine.apply_actions(
        ActionBatch(
            actions=[Action(agent_id="arm", kind=ActionKind.MODE, mode=ModeCommand(mode="digging"))]
        )
    )
    assert engine.export_coupling_state().by_agent["arm"].mode == "digging"


def test_propagation_is_deterministic() -> None:
    a = manipulation_engine_factory(_arm_scenario((_revolute(rate_limit=0.3),)), RngStreams(0))
    b = manipulation_engine_factory(_arm_scenario((_revolute(rate_limit=0.3),)), RngStreams(0))
    for engine in (a, b):
        engine.apply_actions(_position_cmd(1.0))
        for _ in range(3):
            engine.advance(1.0)
    assert _tip(a) == _tip(b)


def test_descriptor_declares_surface_articulated() -> None:
    d = MANIPULATION_ENGINE_DESCRIPTOR
    assert d.regimes == (Regime.SURFACE,)
    assert d.frames == (MOON_BODY_FIXED,)
    assert d.fidelity.tier is FidelityTier.ARTICULATED
    assert d.determinism_class is DeterminismClass.TOLERANCE
    assert d.to_manifest().kind is PluginKind.REGIME_ENGINE


def test_engine_satisfies_regime_engine_protocol() -> None:
    assert isinstance(
        manipulation_engine_factory(_arm_scenario((_revolute(),)), RngStreams(0)), RegimeEngine
    )


def test_coupling_state_round_trips() -> None:
    engine = manipulation_engine_factory(_arm_scenario((_revolute(),)), RngStreams(0))
    engine.apply_actions(_position_cmd(0.4))
    engine.advance(1.0)
    snapshot = engine.export_coupling_state()
    engine.advance(1.0)
    engine.import_coupling_state(snapshot)
    assert engine.export_coupling_state().by_agent["arm"].pose == snapshot.by_agent["arm"].pose


def test_joint_moves_in_the_negative_direction_under_rate_limit() -> None:
    engine = manipulation_engine_factory(
        _arm_scenario((_revolute(rate_limit=0.1, initial=1.0),)), RngStreams(0)
    )
    engine.apply_actions(_position_cmd(0.0))  # target below the initial 1.0 rad → negative delta
    engine.advance(1.0)  # clamped to -0.1 rad → q = 0.9
    assert math.isclose(_tip(engine)[0], math.cos(0.9), rel_tol=1e-9)
