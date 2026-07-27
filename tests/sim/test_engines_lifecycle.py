"""RM-P0-SIM-03 — the lifecycle contract shared by every concrete engine.

Each engine exposes its descriptor off the live instance, ignores actions/imports addressed
to agents it does not own (a no-op, not an error), and retires agents on request. Run once
per engine over a heterogeneous fleet so each factory picks its own agent.
"""

from __future__ import annotations

import math

import pytest

from astro_mine.core.messages.enums import ActionKind
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    ModeCommand,
    Quat,
    StateSample,
    Transform,
    Vec3,
)
from astro_mine.core.sadf.enums import JointType
from astro_mine.core.units import INERTIAL_J2000
from astro_mine.sim.engines import (
    GRANULAR_ENGINE_DESCRIPTOR,
    MANIPULATION_ENGINE_DESCRIPTOR,
    MOBILITY_ENGINE_DESCRIPTOR,
    ORBITAL_ENGINE_DESCRIPTOR,
    CouplingState,
    EngineDescriptor,
    granular_engine_factory,
    manipulation_engine_factory,
    mobility_engine_factory,
    orbital_engine_factory,
)
from astro_mine.sim.engines.registry import EngineFactory
from astro_mine.sim.runtime import (
    AgentSpec,
    GranularDynamics,
    JointSpec,
    ManipulationDynamics,
    MobilityDynamics,
    OrbitalDynamics,
    RngStreams,
    Scenario,
)

_V_CIRC = math.sqrt(4.902800118e12 / 1_837_400.0)


def _fleet() -> Scenario:
    return Scenario(
        name="anchor-fleet",
        agents=(
            AgentSpec(
                agent_id="relay",
                frame=INERTIAL_J2000,
                initial_position_m=(1_837_400.0, 0.0, 0.0),
                velocity_mps=(0.0, _V_CIRC, 0.0),
                dynamics=OrbitalDynamics(),
            ),
            AgentSpec(
                agent_id="rover",
                dynamics=MobilityDynamics(mass_kg=200.0, max_speed_mps=0.5, max_traction_n=300.0),
            ),
            AgentSpec(
                agent_id="arm",
                dynamics=ManipulationDynamics(
                    joints=(JointSpec(name="j", joint_type=JointType.REVOLUTE, rate_limit=0.5),)
                ),
            ),
            AgentSpec(agent_id="digger", dynamics=GranularDynamics(max_dig_rate_m3_s=0.01)),
        ),
    )


_CASES = [
    (orbital_engine_factory, ORBITAL_ENGINE_DESCRIPTOR, "relay"),
    (mobility_engine_factory, MOBILITY_ENGINE_DESCRIPTOR, "rover"),
    (manipulation_engine_factory, MANIPULATION_ENGINE_DESCRIPTOR, "arm"),
    (granular_engine_factory, GRANULAR_ENGINE_DESCRIPTOR, "digger"),
]


def _unknown_sample() -> StateSample:
    return StateSample(
        agent_id="ghost",
        frame=INERTIAL_J2000,
        pose=Transform(
            translation_m=Vec3(x=1.0, y=2.0, z=3.0),
            rotation_quat_xyzw=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )


@pytest.mark.parametrize(("factory", "descriptor", "agent_id"), _CASES)
def test_descriptor_is_introspectable_off_the_live_engine(
    factory: EngineFactory, descriptor: EngineDescriptor, agent_id: str
) -> None:
    engine = factory(_fleet(), RngStreams(0))
    assert engine.descriptor is descriptor


@pytest.mark.parametrize(("factory", "descriptor", "agent_id"), _CASES)
def test_actions_for_unowned_agents_are_ignored(
    factory: EngineFactory, descriptor: EngineDescriptor, agent_id: str
) -> None:
    engine = factory(_fleet(), RngStreams(0))
    engine.apply_actions(
        ActionBatch(
            actions=[Action(agent_id="ghost", kind=ActionKind.MODE, mode=ModeCommand(mode="x"))]
        )
    )
    assert set(engine.export_coupling_state().by_agent) == {agent_id}  # unchanged


@pytest.mark.parametrize(("factory", "descriptor", "agent_id"), _CASES)
def test_import_ignores_unowned_agents(
    factory: EngineFactory, descriptor: EngineDescriptor, agent_id: str
) -> None:
    engine = factory(_fleet(), RngStreams(0))
    engine.import_coupling_state(CouplingState(sim_time_s=5.0, samples=(_unknown_sample(),)))
    assert "ghost" not in engine.export_coupling_state().by_agent


@pytest.mark.parametrize(("factory", "descriptor", "agent_id"), _CASES)
def test_retire_drops_the_agent_and_tolerates_unknowns(
    factory: EngineFactory, descriptor: EngineDescriptor, agent_id: str
) -> None:
    engine = factory(_fleet(), RngStreams(0))
    engine.retire([agent_id, "never-existed"])  # retiring an unknown id is a no-op
    assert engine.export_coupling_state().by_agent == {}
