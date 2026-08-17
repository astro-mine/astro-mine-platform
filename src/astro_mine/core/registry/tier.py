# SPDX-License-Identifier: Apache-2.0
"""A policy plugin's self-declaration: the manifest Core gates, plus the factory that builds it.

Core resolves plugin **manifests** and runs the three load gates (validity → Core-interface-version
negotiation → signature), but it deliberately never instantiates plugin code (core.md §9). A host
that *does* instantiate — today :class:`~astro_mine.mind.registry.TierRegistry` — needs one more
thing from each plugin: a callable that turns the stack spec's ``params`` into a live
:class:`~astro_mine.core.policy.protocol.Policy`. :class:`TierPlugin` is that pair, and it is what
every entry point in the ``astro_mine.mind.tier_plugins`` group returns.

**Why this lives at the waist.** §3.3 puts a Protocol at the waist "when two or more components
share it", and this one had two implementors outside its host before the rule was written: Allocate
registers its CP-SAT planner as a tier, Guard registers its ``PolicyShield`` as the mandatory shield
stage. Both had to import Mind to name the type they implement — the plugin depending on the host's
abstraction, which is correct, with the abstraction in a component rather than at the waist, which
is not. Moving the declaration here dissolves both edges and leaves the registry — the gating,
discovery and instantiation machinery, which is genuinely Mind's — where it was.

This is a Python contract, not a wire one: no schema, no Core interface-version bump. The manifest
it carries is the schema'd half, and that has not changed.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from astro_mine.core.policy.protocol import Policy
from astro_mine.core.registry.model import PluginManifest

__all__ = ["TierFactory", "TierPlugin"]

#: Builds a tier's policy from its stack-spec ``params``. Kept parameter-driven (not zero-arg) so
#: one plugin can be instantiated differently per stack.
TierFactory = Callable[[Mapping[str, Any]], Policy]


@dataclass(frozen=True)
class TierPlugin:
    """A tier/shield plugin's self-declaration: the Core ``manifest`` a registry gates, plus the
    ``factory`` that builds its :class:`~astro_mine.core.policy.protocol.Policy`. Returned by each
    plugin's entry-point provider."""

    manifest: PluginManifest
    factory: TierFactory
