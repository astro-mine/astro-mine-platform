# SPDX-License-Identifier: Apache-2.0
"""The composer — builds and validates a runnable hierarchy graph from a stack spec.

See :func:`astro_mine.mind.compose.composer.compose`: it resolves each tier's and the
shield's plugin through the :class:`~astro_mine.mind.registry.TierRegistry`, validates the
wiring, and returns a :class:`~astro_mine.mind.compose.graph.HierarchyGraph`.
"""

from __future__ import annotations

from astro_mine.mind.compose.composer import TIER_ATTRIBUTE, ComposeError, compose
from astro_mine.mind.compose.graph import HierarchyGraph, ShieldNode, TierNode

__all__ = [
    "TIER_ATTRIBUTE",
    "ComposeError",
    "HierarchyGraph",
    "ShieldNode",
    "TierNode",
    "compose",
]
