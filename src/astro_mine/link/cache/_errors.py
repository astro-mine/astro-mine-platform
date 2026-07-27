"""Errors for the Link determinism / caching / oracle layer."""

from __future__ import annotations

__all__ = ["LinkCacheError", "PassTimeBudgetError"]


class LinkCacheError(RuntimeError):
    """A cache key or oracle comparison cannot be evaluated.

    Raised when a pinned cache input cannot be hashed (a missing kernel/DEM file), a pass
    interval cannot be read, or a comparison is given a negative tolerance — Link **degrades
    loudly** rather than caching against a silently-defaulted key (link.md §2.9).
    """


class PassTimeBudgetError(LinkCacheError):
    """Computed pass times disagree with the oracle beyond the stated tolerance.

    Raised by :func:`~astro_mine.link.cache.assert_within_budget` when the rise/set deltas
    against the external oracle (GMAT/STK/Skyfield) exceed the error budget, or the pass
    counts differ — the reproducibility/validation gate failing (link.md §5, §10).
    """
