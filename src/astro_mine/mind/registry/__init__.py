# SPDX-License-Identifier: Apache-2.0
"""Tier-plugin discovery via the Core manifest + entry-point instantiation.

See :mod:`astro_mine.mind.registry.registry`: :class:`TierRegistry` gates plugin manifests
through Core and instantiates their :class:`~astro_mine.core.policy.protocol.Policy`
backends via Python entry points.

The plugin *declaration* a provider returns is Core's
:class:`~astro_mine.core.registry.TierPlugin`, not Mind's — Allocate and Guard both implement it
from outside, which by conventions.md §3.3 makes it the waist's to own. Import it from
:mod:`astro_mine.core.registry`.
"""

from __future__ import annotations

from astro_mine.mind.registry.registry import (
    ENTRY_POINT_GROUP,
    NotAPolicyPlugin,
    PluginNotRegistered,
    TierRegistry,
    TierRegistryError,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "NotAPolicyPlugin",
    "PluginNotRegistered",
    "TierRegistry",
    "TierRegistryError",
]
