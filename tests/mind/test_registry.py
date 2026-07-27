"""Tier-plugin discovery, gating, and instantiation (RM-P1-MIND-01)."""

from __future__ import annotations

import pytest

from astro_mine.core.messages.model import ActionBatch
from astro_mine.core.registry.enums import DeterminismClass, PluginKind, SignatureScheme
from astro_mine.core.registry.model import PluginManifest, Signature
from astro_mine.core.registry.registry import IncompatibleManifest, UnsignedManifest
from astro_mine.mind.registry import (
    NotAPolicyPlugin,
    PluginNotRegistered,
    TierPlugin,
    TierRegistry,
)

_REFERENCE_PLUGINS = {
    "mind.reference.mission",
    "mind.reference.tamp",
    "mind.reference.control",
    "mind.reference.shield",
}
#: The full advertised plugin set: RM-P1-MIND-01 reference tiers + the enforcing constraint
#: shield (MIND-05) + the RM-P1-MIND-03 backends + the reference allocator (MIND-04). The two
#: native backends (`mind.mission.up`, `mind.tamp.ompl`) are advertised in a *base* install too:
#: their providers defer the heavy unified-planning / OMPL imports into the factory, so
#: discovery works without the [pddl] / [native] extras — only instantiation needs them.
_ALL_PLUGINS = _REFERENCE_PLUGINS | {
    "mind.reference.constraint_shield",
    "mind.mission.pddl",
    "mind.mission.up",
    "mind.tamp.sampling",
    "mind.tamp.ompl",
    "mind.control.pid",
    "mind.control.mpc",
    "mind.control.onnx",
    "mind.allocate.greedy",
}


class _NullPolicy:
    def decide(self, observations, context):  # type: ignore[no-untyped-def]
        return ActionBatch()


def _manifest(
    name: str, *, kind: PluginKind = PluginKind.POLICY, signed: bool = False
) -> PluginManifest:
    return PluginManifest(
        name=name,
        version="0.1.0",
        kind=kind,
        core_interfaces={"policy": "0.1.0", "messages": "0.1.0"},
        determinism_class=DeterminismClass.BIT_EXACT,
        signature=None if signed else Signature(scheme=SignatureScheme.UNSIGNED),
    )


def test_entry_points_discover_reference_plugins() -> None:
    """Every plugin Mind ships is discoverable through the entry-point group.

    A *subset* assertion, not an equality one: the group is deliberately open, and a SIBLING
    package co-installed with Mind registers into it (astro-mine-guard contributes `guard.shield`,
    astro-mine-allocate contributes `allocate.planner`). That is the mechanism by which the real
    shield and the real solver bind with no mind->guard/allocate dependency (mind.md §6, §7) — so
    an installation that has more than Mind's own plugins is the design working, not a failure.
    """
    registry = TierRegistry.from_entry_points()
    assert {m.name for m in registry.manifests} >= _ALL_PLUGINS
    for name in _ALL_PLUGINS:
        assert name in registry


def test_instantiate_returns_a_policy() -> None:
    registry = TierRegistry.from_entry_points()
    policy = registry.instantiate("mind.reference.control", {"max_speed_mps": 1.0})
    assert hasattr(policy, "decide")


def test_register_rejects_non_policy_kind() -> None:
    registry = TierRegistry()
    with pytest.raises(NotAPolicyPlugin):
        registry.register(
            TierPlugin(_manifest("m", kind=PluginKind.METRIC), lambda params: _NullPolicy())
        )


def test_instantiate_unknown_plugin_raises() -> None:
    registry = TierRegistry()
    with pytest.raises(PluginNotRegistered):
        registry.instantiate("nope")


def test_factory_returning_non_policy_is_rejected() -> None:
    registry = TierRegistry()
    registry.register(TierPlugin(_manifest("rogue"), lambda params: object()))  # type: ignore[arg-type,return-value]
    with pytest.raises(NotAPolicyPlugin):
        registry.instantiate("rogue")


def test_signature_gate_can_be_required() -> None:
    registry = TierRegistry(require_signature=True)
    with pytest.raises(UnsignedManifest):
        registry.register(TierPlugin(_manifest("unsigned"), lambda params: _NullPolicy()))


def test_incompatible_core_interface_is_rejected() -> None:
    registry = TierRegistry()
    bad = PluginManifest(
        name="future",
        version="0.1.0",
        kind=PluginKind.POLICY,
        core_interfaces={"policy": "2.0.0"},  # unsupported major at this Core
        signature=Signature(scheme=SignatureScheme.UNSIGNED),
    )
    with pytest.raises(IncompatibleManifest):
        registry.register(TierPlugin(bad, lambda params: _NullPolicy()))
