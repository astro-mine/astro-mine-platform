"""Composition — stack spec → validated runnable hierarchy (RM-P1-MIND-01)."""

from __future__ import annotations

import pytest

from astro_mine.core.messages.model import ActionBatch
from astro_mine.mind.compose import ComposeError, compose
from astro_mine.mind.reference import load_reference_stack
from astro_mine.mind.registry import TierRegistry
from astro_mine.mind.spec.enums import TierRole
from astro_mine.mind.spec.model import (
    FallbackBinding,
    ShieldBinding,
    StackSpec,
    StackSpecDocument,
    TierBinding,
)
from tests.mind.support.harness import policy_plugin, reference_registry


class _NullPolicy:
    def decide(self, observations, context):  # type: ignore[no-untyped-def]
        return ActionBatch()


def test_compose_reference_stack() -> None:
    graph = compose(load_reference_stack(), reference_registry(), seed=7)
    assert graph.stack_id == "reference-lunar-prospecting"
    assert [t.role for t in graph.tiers] == [TierRole.MISSION, TierRole.TAMP, TierRole.CONTROL]
    assert graph.shield.plugin_name == "mind.reference.shield"
    # provenance carries the pinned plugin set + Core interface versions + seed
    assert graph.provenance.seed == 7
    assert "mind.reference.mission" in graph.provenance.plugin_versions
    assert "policy" in graph.provenance.core_interface_versions


def test_tiers_ordered_canonically_regardless_of_authoring_order() -> None:
    doc = StackSpecDocument(
        stack_spec_version="0.1",
        stack_spec=StackSpec(
            id="scrambled",
            name="scrambled",
            tiers=[
                TierBinding(role=TierRole.CONTROL, plugin="mind.reference.control"),
                TierBinding(role=TierRole.MISSION, plugin="mind.reference.mission"),
                TierBinding(role=TierRole.TAMP, plugin="mind.reference.tamp"),
            ],
            shield=ShieldBinding(plugin="mind.reference.shield"),
        ),
    )
    graph = compose(doc, reference_registry())
    assert [t.role for t in graph.tiers] == [TierRole.MISSION, TierRole.TAMP, TierRole.CONTROL]


def test_unknown_plugin_raises_compose_error() -> None:
    doc = StackSpecDocument(
        stack_spec_version="0.1",
        stack_spec=StackSpec(
            id="s",
            name="s",
            tiers=[TierBinding(role=TierRole.CONTROL, plugin="does-not-exist")],
            shield=ShieldBinding(plugin="mind.reference.shield"),
        ),
    )
    with pytest.raises(ComposeError):
        compose(doc, reference_registry())


def test_tier_attribute_mismatch_is_rejected() -> None:
    # A plugin advertising attributes.tier="control" bound to the mission role must fail.
    registry = TierRegistry()
    registry.register(policy_plugin("p-control-ish", lambda params: _NullPolicy(), tier="control"))
    registry.register(policy_plugin("p-shield", lambda params: _NullPolicy()))
    doc = StackSpecDocument(
        stack_spec_version="0.1",
        stack_spec=StackSpec(
            id="s",
            name="s",
            tiers=[TierBinding(role=TierRole.MISSION, plugin="p-control-ish")],
            shield=ShieldBinding(plugin="p-shield"),
        ),
    )
    with pytest.raises(ComposeError, match="advertises tier"):
        compose(doc, registry)


def test_fallback_plugin_is_instantiated() -> None:
    registry = reference_registry()
    doc = StackSpecDocument(
        stack_spec_version="0.1",
        stack_spec=StackSpec(
            id="s",
            name="s",
            tiers=[
                TierBinding(
                    role=TierRole.CONTROL,
                    plugin="mind.reference.control",
                    fallback=FallbackBinding(plugin="mind.reference.control"),
                ),
            ],
            shield=ShieldBinding(plugin="mind.reference.shield"),
        ),
    )
    graph = compose(doc, registry)
    control = graph.tier(TierRole.CONTROL)
    assert control is not None
    assert control.fallback is not None
    assert control.fallback_name == "mind.reference.control"
