"""Errors for the Link geometry layer."""

from __future__ import annotations

__all__ = ["LinkGeometryError"]


class LinkGeometryError(RuntimeError):
    """A geometric line-of-sight query cannot be evaluated.

    Raised when a required input is missing — no world provider, no ephemeris provider
    for an ephemeris node, or a frame with no centre body. Link **degrades loudly**:
    rather than assume connectivity on missing data, it raises. A link is never defaulted
    to "connected" (link.md §2.9).
    """
