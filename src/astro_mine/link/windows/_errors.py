# SPDX-License-Identifier: Apache-2.0
"""Errors for the Link contact-window layer."""

from __future__ import annotations

__all__ = ["LinkWindowError"]


class LinkWindowError(RuntimeError):
    """A contact-window search cannot be evaluated.

    Raised on an invalid search configuration — a non-positive sampling step, a
    refinement tolerance coarser than the step, or a degenerate epoch window. A
    *visibility* failure (a missing kernel, a missing world/ephemeris provider) is the
    geometry layer's :class:`~astro_mine.link.geometry.LinkGeometryError`, which
    propagates through the search unchanged: Link **degrades loudly** rather than
    folding a failed query into an empty "no contact" result (link.md §2.9).
    """
