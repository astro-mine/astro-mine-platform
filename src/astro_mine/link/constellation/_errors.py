# SPDX-License-Identifier: Apache-2.0
"""Constellation-layer errors."""

from __future__ import annotations

from astro_mine.link.geometry import LinkGeometryError

__all__ = ["LinkConstellationError"]


class LinkConstellationError(LinkGeometryError):
    """A constellation could not be modeled — a duplicate/undeclared node, an empty node
    set, or a malformed reachability query. Subclasses :class:`LinkGeometryError` so a
    caller can catch every Link geometry failure uniformly; like the rest of Link it
    degrades *loudly* rather than assuming connectivity (link.md §2.9)."""
