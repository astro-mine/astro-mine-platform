"""RM-P0-SIM-03 — the built-in engine set is registered and resolvable behind the waist.

The default registry exposes the reference engine plus the four anchor-scenario engines;
each gates through Core, resolves by name, and instantiates owning only its regime's agents.
"""

from __future__ import annotations

import math

from astro_mine.core.registry import PluginKind
from astro_mine.core.sadf.enums import JointType
from astro_mine.core.units import INERTIAL_J2000
from astro_mine.sim.engines import (
    BUILTIN_ENGINES,
    EngineRegistry,
    RegimeEngine,
    default_engine_registry,
    register_builtin_engines,
)
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
            AgentSpec(agent_id="kin"),  # kinematic default
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


def test_builtin_engine_names_are_unique() -> None:
    # The set grows append-only, so the invariant under test is uniqueness (a duplicate name would
    # silently shadow an engine in the registry), not a head count.
    names = [descriptor.name for descriptor, _ in BUILTIN_ENGINES]
    assert len(set(names)) == len(names)


def test_default_registry_registers_every_builtin() -> None:
    registry = default_engine_registry()
    assert set(registry.names) == {
        "astro-mine.sim.kinematic",
        # The reduced-order tiers — the always-works local fallback (CX-LOCAL).
        "astro-mine.sim.orbital",
        "astro-mine.sim.mobility",
        "astro-mine.sim.manipulation",
        "astro-mine.sim.granular",
        # The high-fidelity / real-backend tiers (RM-P0-SIM-03, RM-P1-SIM-04/06). Every one of these
        # registers **manifest-only** — no numpy, JAX, MuJoCo, or JVM is imported to get here.
        "astro-mine.sim.dem_granular",
        "astro-mine.sim.brax_contact",
        "astro-mine.sim.mjx_contact",
        "astro-mine.sim.orekit_orbital",
        "astro-mine.sim.mujoco_mobility",
    }
    for name in registry.names:
        assert registry.manifest(name).kind is PluginKind.REGIME_ENGINE


def test_register_builtin_engines_is_idempotent_into_a_given_registry() -> None:
    registry = register_builtin_engines(EngineRegistry())
    assert len(registry.names) == len(BUILTIN_ENGINES)


def test_each_specialized_engine_owns_only_its_regime() -> None:
    registry = default_engine_registry()
    scenario = _fleet()
    expected = {
        "astro-mine.sim.orbital": "relay",
        "astro-mine.sim.mobility": "rover",
        "astro-mine.sim.manipulation": "arm",
        "astro-mine.sim.granular": "digger",
    }
    for name, agent_id in expected.items():
        engine = registry.create(name, scenario, RngStreams(0))
        assert isinstance(engine, RegimeEngine)
        assert set(engine.export_coupling_state().by_agent) == {agent_id}
