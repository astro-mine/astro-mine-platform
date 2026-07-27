"""The executive — ticking, shielded egress, replan triggers, and fallbacks (RM-P1-MIND-01)."""

from __future__ import annotations

from astro_mine.mind.compose import compose
from astro_mine.mind.exec import Executive
from astro_mine.mind.spec.enums import TierRole
from astro_mine.mind.spec.model import (
    FallbackBinding,
    ShieldBinding,
    StackSpec,
    StackSpecDocument,
    TierBinding,
)
from tests.mind.support.harness import (
    TAGGING_SENTINEL,
    RaisingPolicy,
    TaggingShield,
    assert_shielded_egress,
    compose_reference,
    policy_plugin,
    reference_doc_with_shield,
    reference_registry,
    run_reference,
)
from tests.mind.support.toy_env import ToyProspectingEnv


def _records_for(result, role: str):  # type: ignore[no-untyped-def]
    return [rec for tick in result.trace.ticks for rec in tick.tiers if rec.role == role]


def test_run_steps_env_and_moves_agents() -> None:
    result = run_reference(horizon=6, max_ticks=6)
    assert result.ticks_run == 6
    assert result.truncated is True
    # each agent moved toward its assigned region (mission assigns +x regions).
    for obs in result.final_observations.values():
        assert obs.self_state.pose.translation_m.x > 0.0


def test_guard_is_the_only_output_path() -> None:
    # Swap in a shield that stamps a sentinel on every action it emits; every emitted
    # action must bear it — nothing reaches the env without passing through the shield.
    registry = reference_registry()
    registry.register(policy_plugin("test.tagging-shield", lambda params: TaggingShield()))
    graph = compose(reference_doc_with_shield("test.tagging-shield"), registry, seed=7)
    result = Executive(graph).run(ToyProspectingEnv(horizon=5), max_ticks=5, seed=7)
    assert_shielded_egress(result)


def test_adversarial_tier_output_is_still_shielded() -> None:
    # The emitted batch is always the shield's output, never the tier's raw output. Contrast
    # the pass-through shield (leaves the tier's own sim_time_s) with the tagging shield
    # (overrides every action) — a tier cannot get an un-shielded action to the env.
    passthrough = run_reference(horizon=3, max_ticks=3)
    raw = [a for tick in passthrough.trace.ticks for a in tick.action_batch.actions]
    assert raw and all(a.sim_time_s != TAGGING_SENTINEL for a in raw)

    registry = reference_registry()
    registry.register(policy_plugin("test.tagging-shield", lambda params: TaggingShield()))
    graph = compose(reference_doc_with_shield("test.tagging-shield"), registry, seed=1)
    shielded = Executive(graph).run(ToyProspectingEnv(horizon=3), max_ticks=3, seed=1)
    assert_shielded_egress(shielded)


def test_fallback_activates_on_tier_failure() -> None:
    registry = reference_registry()
    registry.register(
        policy_plugin("test.raising-tamp", lambda params: RaisingPolicy(), tier="tamp")
    )
    doc = StackSpecDocument(
        stack_spec_version="0.1",
        stack_spec=StackSpec(
            id="fallback-stack",
            name="fallback",
            tiers=[
                TierBinding(role=TierRole.MISSION, plugin="mind.reference.mission"),
                TierBinding(
                    role=TierRole.TAMP,
                    plugin="test.raising-tamp",
                    fallback=FallbackBinding(plugin="mind.reference.tamp"),
                ),
                TierBinding(role=TierRole.CONTROL, plugin="mind.reference.control"),
            ],
            shield=ShieldBinding(plugin="mind.reference.shield"),
        ),
    )
    result = Executive(compose(doc, registry, seed=7)).run(
        ToyProspectingEnv(horizon=4), max_ticks=4, seed=7
    )
    # the raising tamp degrades to its fallback (never crashes the executive)...
    tamp = _records_for(result, "tamp")
    assert tamp and all(rec.fallback_used for rec in tamp if rec.replanned)
    # ...and agents still make progress via the fallback's GOTOs.
    assert any(
        obs.self_state.pose.translation_m.x > 0.0 for obs in result.final_observations.values()
    )


def test_degrades_to_safe_idle_without_a_fallback() -> None:
    # A failing tier with no fallback and no cache yields a safe-idle empty batch — a
    # defined state, not a crash (degrade-not-collapse).
    registry = reference_registry()
    registry.register(
        policy_plugin("test.raising-control", lambda params: RaisingPolicy(), tier="control")
    )
    doc = StackSpecDocument(
        stack_spec_version="0.1",
        stack_spec=StackSpec(
            id="idle-stack",
            name="idle",
            tiers=[TierBinding(role=TierRole.CONTROL, plugin="test.raising-control")],
            shield=ShieldBinding(plugin="mind.reference.shield"),
        ),
    )
    result = Executive(compose(doc, registry, seed=7)).run(
        ToyProspectingEnv(horizon=3), max_ticks=3, seed=7
    )
    assert result.ticks_run == 3  # completed without crashing
    assert all(not tick.action_batch.actions for tick in result.trace.ticks)  # safe-idle


def test_replan_triggers_fire_as_configured() -> None:
    # Reference stack: mission horizon 5s (plan_expired at t>=5), tamp periodic every 3,
    # control reactive (every tick).
    result = run_reference(horizon=6, max_ticks=6)
    mission = _records_for(result, "mission")
    tamp = _records_for(result, "tamp")
    control = _records_for(result, "control")

    assert mission[0].trigger == "initial"
    assert not mission[1].replanned  # cached within horizon
    assert mission[5].trigger == "plan_expired"  # 5s horizon elapsed at t=5

    assert tamp[3].trigger == "periodic"  # every_ticks=3

    assert all(rec.replanned for rec in control)  # reactive every tick


def test_composition_is_stateful_but_run_is_fresh() -> None:
    # Executive.run builds a fresh strategy each call, so two runs of one Executive match.
    graph = compose_reference(seed=7)
    executive = Executive(graph)
    a = executive.run(ToyProspectingEnv(horizon=5), max_ticks=5, seed=7)
    b = executive.run(ToyProspectingEnv(horizon=5), max_ticks=5, seed=7)
    assert a.trace.to_dict() == b.trace.to_dict()
