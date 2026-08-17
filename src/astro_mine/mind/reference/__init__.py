# SPDX-License-Identifier: Apache-2.0
"""Reference tier/shield plugins — replaceable examples for local dev and tests.

Deterministic, minimal implementations of the three tiers plus a pass-through shield,
bundled with Core manifests and advertised as entry points (group
``astro_mine.mind.tier_plugins``) so :meth:`TierRegistry.from_entry_points
<astro_mine.mind.registry.TierRegistry.from_entry_points>` discovers them, plus the shipped
reference stack spec that wires them (:func:`load_reference_stack`). Not privileged
internals — the heavyweight backends (RM-P1-MIND-03) and the real Guard shield
(RM-P1-MIND-05) replace these through the same registry.
"""

from __future__ import annotations

from collections.abc import Iterator
from importlib import resources

from astro_mine.mind.bt.model import BehaviorTree
from astro_mine.mind.bt.xml import parse_behavior_tree
from astro_mine.mind.reference.tiers import (
    ConstraintShield,
    PassthroughShield,
    ScriptedController,
    ScriptedMissionPlanner,
    ScriptedTampPlanner,
    constraint_shield_plugin,
    control_plugin,
    mission_plugin,
    shield_plugin,
    tamp_plugin,
)
from astro_mine.mind.spec.loader import load_stack_spec
from astro_mine.mind.spec.model import StackSpecDocument

__all__ = [
    "REFERENCE_BT_RESOURCE",
    "REFERENCE_BT_STACK_RESOURCE",
    "REFERENCE_STACK_RESOURCE",
    "ConstraintShield",
    "PassthroughShield",
    "ScriptedController",
    "ScriptedMissionPlanner",
    "ScriptedTampPlanner",
    "constraint_shield_plugin",
    "control_plugin",
    "iter_manifest_resources",
    "iter_stack_resources",
    "load_reference_bt",
    "load_reference_stack",
    "load_stack_resource",
    "mission_plugin",
    "shield_plugin",
    "tamp_plugin",
]

#: The shipped reference stack spec resource (under ``reference/stacks/``).
REFERENCE_STACK_RESOURCE = "stacks/lunar_prospecting.yaml"
#: The shipped reference behavior-tree stack spec + tree (RM-P1-MIND-02).
REFERENCE_BT_STACK_RESOURCE = "stacks/lunar_prospecting_bt.yaml"
REFERENCE_BT_RESOURCE = "trees/lunar_prospecting.bt.xml"


def load_reference_bt() -> BehaviorTree:
    """Load and parse the shipped reference lunar-prospecting behavior tree."""
    text = (
        resources.files("astro_mine.mind.reference")
        .joinpath("trees", "lunar_prospecting.bt.xml")
        .read_text(encoding="utf-8")
    )
    return parse_behavior_tree(text)


def load_stack_resource(resource: str) -> StackSpecDocument:
    """Load and validate a shipped reference stack spec by ``stacks/``-relative name."""
    text = (
        resources.files("astro_mine.mind.reference")
        .joinpath("stacks", resource)
        .read_text(encoding="utf-8")
    )
    return load_stack_spec(text)


def load_reference_stack() -> StackSpecDocument:
    """Load and validate the shipped reference lunar-prospecting stack spec."""
    return load_stack_resource("lunar_prospecting.yaml")


# --- shipped reference package data -------------------------------------------------------
#
# These moved here from `astro_mine.mind.cli` when the CLI surface left the platform
# (astro-mine-platform#1). They were public API in that module -- `astro-mine mind stacks`
# lists what they yield -- and they are package-data readers, not argv handling, so the CLI
# was never their right home.

_REFERENCE = "astro_mine.mind.reference"


def iter_stack_resources() -> Iterator[str]:
    """The shipped reference stack-spec filenames, sorted (from package data, wheel-safe)."""
    yield from sorted(
        entry.name
        for entry in resources.files(_REFERENCE).joinpath("stacks").iterdir()
        if entry.name.endswith(".yaml")
    )


def iter_manifest_resources() -> Iterator[str]:
    """The shipped reference plugin-manifest filenames, sorted (from package data, wheel-safe)."""
    yield from sorted(
        entry.name
        for entry in resources.files(_REFERENCE).joinpath("manifests").iterdir()
        if entry.name.endswith(".yaml")
    )
