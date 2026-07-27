"""RM-P0-SIM-02 — the engine-adapter framework (``RegimeEngine`` plugin).

Covers the adapter contract + descriptor introspection, the reference engine driving the
stepping core behind the waist, the coupling-state export/import round-trip, and the
:class:`EngineRegistry` gating engine loads through Core's plugin registry (RM-P0-CORE-05)
— including the dual-use capability gate and the signature gate. The determinism and
Core-Environment-contract guarantees of the refactor are held (unchanged) by
``test_runtime.py``.
"""

from __future__ import annotations

import pytest

from astro_mine.core.messages.enums import ActionKind, ControlMode
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    ActuatorCommand,
    ModeCommand,
    Quat,
    StateSample,
    Transform,
    Vec3,
)
from astro_mine.core.registry import (
    CapabilityTag,
    ManifestValidationError,
    PluginKind,
    RegistryError,
    UnsignedManifest,
)
from astro_mine.core.sadf.enums import (
    DeterminismClass,
    FidelityTier,
    Regime,
    SurrogatePhysicsDomain,
)
from astro_mine.core.units import INERTIAL_J2000, MOON_BODY_FIXED
from astro_mine.sim.engines import (
    KINEMATIC_ENGINE_DESCRIPTOR,
    CouplingState,
    EngineDescriptor,
    EngineRegistry,
    FidelityDescriptor,
    KinematicEngine,
    RegimeEngine,
    actions_by_agent,
    kinematic_engine_factory,
)
from astro_mine.sim.runtime import AgentSpec, RngStreams, Scenario, Simulator

_KINEMATIC = KINEMATIC_ENGINE_DESCRIPTOR.name


def _scenario(**overrides: object) -> Scenario:
    base: dict[str, object] = {
        "name": "engine-unit",
        "agents": (
            AgentSpec(agent_id="a", velocity_mps=(1.0, 0.0, 0.0), battery_soc_j=100.0),
            AgentSpec(
                agent_id="b",
                velocity_mps=(0.0, 2.0, 0.0),
                battery_soc_j=50.0,
                frame=INERTIAL_J2000,
            ),
        ),
        "seed": 7,
        "dt_s": 0.5,
        "horizon_steps": 3,
    }
    base.update(overrides)
    return Scenario(**base)  # type: ignore[arg-type]


def _sample(agent_id: str, x: float, y: float, z: float) -> StateSample:
    return StateSample(
        agent_id=agent_id,
        frame=MOON_BODY_FIXED,
        pose=Transform(
            translation_m=Vec3(x=x, y=y, z=z),
            rotation_quat_xyzw=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
        ),
    )


# --- the adapter contract + descriptor introspection -----------------------------


def test_reference_engine_satisfies_the_regime_engine_protocol() -> None:
    engine = kinematic_engine_factory(_scenario(), RngStreams(7))
    assert isinstance(engine, RegimeEngine)  # runtime-checkable structural match
    assert isinstance(engine, KinematicEngine)


def test_descriptor_declares_frames_determinism_and_fidelity() -> None:
    d = KINEMATIC_ENGINE_DESCRIPTOR
    assert d.frames == (MOON_BODY_FIXED, INERTIAL_J2000)
    assert d.determinism_class is DeterminismClass.BIT_EXACT
    assert d.fidelity.tier is FidelityTier.KINEMATIC
    assert d.regimes == (Regime.SURFACE,)
    # the same declaration is introspectable behind the waist, off the live engine.
    engine = kinematic_engine_factory(_scenario(), RngStreams(7))
    assert engine.descriptor is d


def test_descriptor_renders_a_core_regime_engine_manifest() -> None:
    manifest = KINEMATIC_ENGINE_DESCRIPTOR.to_manifest()
    assert manifest.kind is PluginKind.REGIME_ENGINE
    assert manifest.determinism_class is DeterminismClass.BIT_EXACT
    assert manifest.regimes == [Regime.SURFACE]
    assert manifest.core_interfaces == {"messages": "0.1.0"}
    # the fidelity descriptor + frames ride in the open attributes map (Core does not
    # schematize them — core.md registry attributes; sim.md §11).
    assert manifest.attributes["fidelity"] == {"tier": "kinematic"}
    assert manifest.attributes["frames"][0]["name"] == "MOON_ME"


def test_fidelity_descriptor_carries_a_surrogate_domain() -> None:
    fidelity = FidelityDescriptor(
        tier=FidelityTier.SURROGATE,
        surrogate_domain=SurrogatePhysicsDomain.GRANULAR_EXCAVATION,
    )
    assert fidelity.as_attributes() == {
        "tier": "surrogate",
        "surrogate_domain": "granular_excavation",
    }


# --- the engine drives the stepping core -----------------------------------------


def test_engine_advances_state_deterministically() -> None:
    engine = kinematic_engine_factory(_scenario(), RngStreams(7))
    before = engine.export_coupling_state().by_agent["a"]
    engine.advance(0.5)
    after = engine.export_coupling_state().by_agent["a"]
    assert after.pose.translation_m.x > before.pose.translation_m.x  # moves along +x
    assert after.battery_soc_j is not None and before.battery_soc_j is not None
    assert after.battery_soc_j < before.battery_soc_j  # battery drains
    # same seed -> identical advance (BIT_EXACT determinism).
    twin = kinematic_engine_factory(_scenario(), RngStreams(7))
    twin.advance(0.5)
    assert twin.export_coupling_state().by_agent["a"].pose == after.pose


def test_coupling_state_is_frame_explicit() -> None:
    engine = kinematic_engine_factory(_scenario(), RngStreams(7))
    by_agent = engine.export_coupling_state().by_agent
    assert by_agent["a"].frame == MOON_BODY_FIXED  # scenario default
    assert by_agent["b"].frame == INERTIAL_J2000  # per-agent override


def test_coupling_state_round_trips_through_import() -> None:
    engine = kinematic_engine_factory(_scenario(), RngStreams(7))
    engine.advance(0.5)
    snapshot = engine.export_coupling_state()
    engine.advance(0.5)  # diverge from the snapshot
    assert engine.export_coupling_state().by_agent["a"].pose != snapshot.by_agent["a"].pose
    engine.import_coupling_state(snapshot)  # adopt the boundary state
    restored = engine.export_coupling_state()
    assert restored.by_agent["a"].pose == snapshot.by_agent["a"].pose
    assert restored.by_agent["a"].battery_soc_j == snapshot.by_agent["a"].battery_soc_j
    assert restored.sim_time_s == snapshot.sim_time_s


def test_import_ignores_unknown_agents() -> None:
    engine = kinematic_engine_factory(_scenario(), RngStreams(7))
    engine.import_coupling_state(
        CouplingState(sim_time_s=9.0, samples=(_sample("ghost", 1, 2, 3),))
    )
    assert "ghost" not in engine.export_coupling_state().by_agent


def test_import_of_a_partial_sample_preserves_unspecified_fields() -> None:
    engine = kinematic_engine_factory(_scenario(), RngStreams(7))
    before = engine.export_coupling_state().by_agent["a"]
    engine.import_coupling_state(
        CouplingState(sim_time_s=2.0, samples=(_sample("a", 5.0, 6.0, 7.0),))
    )
    after = engine.export_coupling_state().by_agent["a"]
    t = after.pose.translation_m
    assert (t.x, t.y, t.z) == (5.0, 6.0, 7.0)  # pose overwritten
    assert after.linear_velocity_mps == before.linear_velocity_mps  # velocity preserved
    assert after.battery_soc_j == before.battery_soc_j  # battery preserved


def test_retire_drops_agents_from_the_engine() -> None:
    engine = kinematic_engine_factory(_scenario(), RngStreams(7))
    engine.retire(["a"])
    by_agent = engine.export_coupling_state().by_agent
    assert "a" not in by_agent and "b" in by_agent
    engine.retire(["nonexistent"])  # retiring an unknown id is a no-op


# --- the stepping core routes any engine behind the waist ------------------------


def test_simulator_drives_an_injected_engine_factory() -> None:
    built: list[str] = []

    def factory(scenario: Scenario, rng: RngStreams) -> KinematicEngine:
        built.append(scenario.name)
        return kinematic_engine_factory(scenario, rng)

    sim = Simulator(_scenario(), engine_factory=factory)
    result = sim.reset()
    assert built == ["engine-unit"]  # the injected factory was used
    # observations expose only Core messages — the engine type does not leak, and the
    # coupling-only velocity field is projected away from the observation surface.
    obs = result.observations["a"]
    assert isinstance(obs.self_state, StateSample)
    assert obs.self_state.linear_velocity_mps is None


def test_step_before_reset_raises() -> None:
    sim = Simulator(_scenario())
    with pytest.raises(RuntimeError, match="reset"):
        sim.step(ActionBatch())


# --- the engine registry gates loads through Core (RM-P0-CORE-05) -----------------


def test_registry_registers_resolves_and_creates() -> None:
    registry = EngineRegistry()  # local/dev: signing off
    manifest = registry.register(KINEMATIC_ENGINE_DESCRIPTOR, kinematic_engine_factory)
    assert manifest.kind is PluginKind.REGIME_ENGINE
    assert _KINEMATIC in registry
    assert registry.names == (_KINEMATIC,)
    assert registry.manifest(_KINEMATIC).version == "0.1.0"
    # resolve + instantiate behind the waist, then drive the stepping core with it.
    engine = registry.create(_KINEMATIC, _scenario(), RngStreams(7))
    assert isinstance(engine, RegimeEngine)
    sim = Simulator(_scenario(), engine_factory=lambda s, r: registry.create(_KINEMATIC, s, r))
    assert set(sim.reset().observations) == {"a", "b"}


def test_registry_rejects_an_unregistered_engine() -> None:
    registry = EngineRegistry()
    with pytest.raises(RegistryError):
        registry.create("astro-mine.sim.missing", _scenario(), RngStreams(7))


def test_registry_refuses_a_gated_capability_tag() -> None:
    # The dual-use gate is Core's: an open-commons engine MUST NOT declare a reserved tag.
    leaky = EngineDescriptor(
        name="astro-mine.sim.leaky",
        version="0.1.0",
        regimes=(Regime.SURFACE,),
        frames=(MOON_BODY_FIXED,),
        determinism_class=DeterminismClass.BIT_EXACT,
        fidelity=FidelityDescriptor(tier=FidelityTier.KINEMATIC),
        capability_tags=(CapabilityTag.GROUND_TRUTH_ACCESS,),
    )
    registry = EngineRegistry()
    with pytest.raises(ManifestValidationError, match="gated"):
        registry.register(leaky, kinematic_engine_factory)
    assert "astro-mine.sim.leaky" not in registry  # no partial registration


def test_registry_requires_a_signature_when_hardened() -> None:
    registry = EngineRegistry(require_signature=True)
    with pytest.raises(UnsignedManifest):
        registry.register(KINEMATIC_ENGINE_DESCRIPTOR, kinematic_engine_factory)


# --- actuation: ActionBatch reaches the engine (RM-P0-SIM-03) ---------------------


def _velocity_action(agent_id: str, vx: float, vy: float, vz: float) -> Action:
    return Action(
        agent_id=agent_id,
        kind=ActionKind.ACTUATOR,
        actuator=ActuatorCommand(
            target="base", control_mode=ControlMode.VELOCITY, setpoint=[vx, vy, vz]
        ),
    )


def test_actions_by_agent_indexes_last_write_wins() -> None:
    batch = ActionBatch(actions=[_velocity_action("a", 1, 0, 0), _velocity_action("a", 9, 0, 0)])
    indexed = actions_by_agent(batch)
    assert set(indexed) == {"a"}
    assert indexed["a"].actuator is not None
    assert indexed["a"].actuator.setpoint == [9, 0, 0]  # later entry wins


def test_apply_actions_retargets_velocity() -> None:
    engine = kinematic_engine_factory(_scenario(), RngStreams(7))
    engine.apply_actions(ActionBatch(actions=[_velocity_action("a", 0.0, 5.0, 0.0)]))
    engine.advance(1.0)
    after = engine.export_coupling_state().by_agent["a"]
    assert after.linear_velocity_mps == Vec3(x=0.0, y=5.0, z=0.0)  # retargeted +y
    assert after.pose.translation_m.y > 0.0


def test_apply_actions_sets_mode() -> None:
    engine = kinematic_engine_factory(_scenario(), RngStreams(7))
    engine.apply_actions(
        ActionBatch(
            actions=[Action(agent_id="a", kind=ActionKind.MODE, mode=ModeCommand(mode="prospect"))]
        )
    )
    assert engine.export_coupling_state().by_agent["a"].mode == "prospect"


def test_apply_actions_empty_batch_is_a_no_op() -> None:
    engine = kinematic_engine_factory(_scenario(), RngStreams(7))
    before = engine.export_coupling_state().by_agent["a"]
    engine.apply_actions(ActionBatch())  # what run_episode passes
    after = engine.export_coupling_state().by_agent["a"]
    assert after == before  # unchanged -> the reference trace stays byte-identical


def test_apply_actions_ignores_unknown_agents() -> None:
    engine = kinematic_engine_factory(_scenario(), RngStreams(7))
    before = engine.export_coupling_state().by_agent["a"]
    engine.apply_actions(ActionBatch(actions=[_velocity_action("ghost", 1, 2, 3)]))
    assert engine.export_coupling_state().by_agent["a"] == before


def test_simulator_actuates_each_step() -> None:
    sim = Simulator(_scenario())
    sim.reset(seed=7)
    # drive agent "a" hard along +y; its default velocity is +x.
    sim.step(ActionBatch(actions=[_velocity_action("a", 0.0, 10.0, 0.0)]))
    result = sim.step(ActionBatch(actions=[_velocity_action("a", 0.0, 10.0, 0.0)]))
    assert result.observations["a"].self_state.pose.translation_m.y > 0.0
