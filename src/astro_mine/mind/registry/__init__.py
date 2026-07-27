"""Tier-plugin discovery via the Core manifest + entry-point instantiation.

See :mod:`astro_mine.mind.registry.registry`: :class:`TierRegistry` gates plugin manifests
through Core and instantiates their :class:`~astro_mine.core.policy.protocol.Policy`
backends via Python entry points.
"""

from __future__ import annotations

from astro_mine.mind.registry.registry import (
    ENTRY_POINT_GROUP,
    NotAPolicyPlugin,
    PluginNotRegistered,
    TierFactory,
    TierPlugin,
    TierRegistry,
    TierRegistryError,
)

__all__ = [
    "ENTRY_POINT_GROUP",
    "NotAPolicyPlugin",
    "PluginNotRegistered",
    "TierFactory",
    "TierPlugin",
    "TierRegistry",
    "TierRegistryError",
]
