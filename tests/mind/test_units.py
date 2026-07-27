"""Unit coverage for the spine's smaller surfaces (RM-P1-MIND-01)."""

from __future__ import annotations

import pytest

from astro_mine.mind.belief import BELIEF_EXTRAS_KEY, assemble_belief
from astro_mine.mind.compose import compose
from astro_mine.mind.exec import Executive
from astro_mine.mind.spec import StackSpecValidationError, load_stack_spec, validate_stack_spec
from astro_mine.mind.spec.enums import ReplanTriggerKind, TierRole
from astro_mine.mind.spec.model import (
    FallbackBinding,
    ReplanTrigger,
    ShieldBinding,
    StackSpec,
    StackSpecDocument,
    TierBinding,
)
from tests.mind.support.harness import RaisingPolicy, policy_plugin, reference_registry
from tests.mind.support.toy_env import ToyProspectingEnv

# --- belief view ------------------------------------------------------------------


def test_belief_view_surfaces_observations_and_comms() -> None:
    observations = ToyProspectingEnv(comms_denied_ticks=[0]).reset().observations
    belief = assemble_belief(observations, tick=0, sim_time_s=0.0)
    agent = belief.agents[0]
    assert BELIEF_EXTRAS_KEY == "astro_mine.mind.belief"
    assert belief.observation(agent) is not None
    assert belief.observation("ghost") is None
    assert belief.is_observable(agent) is True
    assert belief.is_observable("ghost") is False
    assert belief.earth_contact(agent) is False  # tick 0 is comms-denied
    assert (
        assemble_belief(
            ToyProspectingEnv().reset().observations, tick=0, sim_time_s=0.0
        ).earth_contact(agent)
        is True
    )


# --- validate_stack_spec input forms ----------------------------------------------

_VALID_TEXT = """
stack_spec_version: "0.1"
stack_spec:
  id: s
  name: s
  tiers: [{role: control, plugin: p}]
  shield: {plugin: sh}
"""


def test_validate_accepts_text_dict_and_typed_doc() -> None:
    validate_stack_spec(_VALID_TEXT)  # text
    doc = load_stack_spec(_VALID_TEXT)
    validate_stack_spec(doc)  # typed document
    validate_stack_spec(doc.model_dump(mode="json"))  # parsed mapping


def test_validate_rejects_bad_dict_and_wrong_type() -> None:
    with pytest.raises(StackSpecValidationError):
        validate_stack_spec({"stack_spec_version": "0.1", "stack_spec": {"id": "s"}})
    with pytest.raises(StackSpecValidationError):
        validate_stack_spec(12345)


# --- registry ---------------------------------------------------------------------


def test_registry_len_and_from_entry_points_count() -> None:
    registry = reference_registry()
    # 4 RM-P1-MIND-01 reference tiers + constraint shield (MIND-05) + 5 RM-P1-MIND-03 reference
    # backends + 2 RM-P1-MIND-03 native backends (unified-planning mission, OMPL TAMP — both
    # discoverable without their extras; only instantiation needs them) + reference allocator
    # (MIND-04).
    #
    # A floor, not an equality: the entry-point group is open by design, so a co-installed sibling
    # (astro-mine-guard's `guard.shield`, astro-mine-allocate's `allocate.planner` — both present
    # under the [sim] extra) legitimately adds to it. See test_registry.py for the same reasoning.
    assert len(registry) >= 13


# --- reference tier params --------------------------------------------------------


def test_control_and_mission_accept_params() -> None:
    registry = reference_registry()
    control = registry.instantiate("mind.reference.control", {"gain": 0.25, "max_speed_mps": 3.0})
    mission = registry.instantiate("mind.reference.mission", {"spacing_m": 4.0})
    assert hasattr(control, "decide")
    assert hasattr(mission, "decide")


# --- strategy fallback chain + on_fallback reconcile ------------------------------


def _one_tier_stack(
    plugin: str, *, fallback: str | None, triggers: list[ReplanTrigger]
) -> StackSpecDocument:
    binding = TierBinding(
        role=TierRole.CONTROL,
        plugin=plugin,
        replan_triggers=triggers,
        fallback=FallbackBinding(plugin=fallback) if fallback else None,
    )
    return StackSpecDocument(
        stack_spec_version="0.1",
        stack_spec=StackSpec(
            id="one",
            name="one",
            tiers=[binding],
            shield=ShieldBinding(plugin="mind.reference.shield"),
        ),
    )


def test_fallback_that_also_fails_degrades_to_safe_idle() -> None:
    registry = reference_registry()
    registry.register(policy_plugin("test.raise-a", lambda params: RaisingPolicy(), tier="control"))
    registry.register(policy_plugin("test.raise-b", lambda params: RaisingPolicy()))
    doc = _one_tier_stack("test.raise-a", fallback="test.raise-b", triggers=[])
    result = Executive(compose(doc, registry)).run(ToyProspectingEnv(horizon=2), max_ticks=2)
    # primary raises, fallback raises, no cache -> safe-idle empty batch, no crash
    assert all(not tick.action_batch.actions for tick in result.trace.ticks)


def test_on_fallback_trigger_forces_reconcile() -> None:
    registry = reference_registry()
    registry.register(policy_plugin("test.raise-c", lambda params: RaisingPolicy(), tier="control"))
    doc = _one_tier_stack(
        "test.raise-c",
        fallback="mind.reference.control",
        triggers=[ReplanTrigger(kind=ReplanTriggerKind.ON_FALLBACK)],
    )
    result = Executive(compose(doc, registry)).run(ToyProspectingEnv(horizon=3), max_ticks=3)
    control = [rec for tick in result.trace.ticks for rec in tick.tiers if rec.role == "control"]
    assert control[0].fallback_used is True
    assert control[1].trigger == ReplanTriggerKind.ON_FALLBACK.value


# --- executive termination --------------------------------------------------------


def test_run_stops_on_termination() -> None:
    from tests.mind.support.harness import compose_reference

    result = Executive(compose_reference()).run(
        ToyProspectingEnv(horizon=99, terminate_at=2), max_ticks=10
    )
    assert result.terminated is True
    assert result.ticks_run == 2
