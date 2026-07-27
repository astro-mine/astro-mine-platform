"""Determinism + content-addressed caching + oracle cross-checks (RM-P0-LINK-05).

Results are cached by a content key over kernels / DEM / node-set / epoch / config
(:class:`CacheKey`, :func:`cache_key`, :func:`build_cache_key`) so plans reproduce from
pinned inputs; the :class:`PlanCache` persists Core's byte-stable wire form so a re-run is a
byte-for-byte hit. Pass times are cross-checked against an external oracle (GMAT/STK/Skyfield)
within an explicit budget (:func:`cross_check_pass_times`, :func:`assert_within_budget`); the
live GMAT run driving that check lives in the test suite. :func:`plan_digest` /
:func:`hash_file` / :func:`canonical_digest` are the content-address primitives.

Backlog: RM-P0-LINK-05 -- https://github.com/astro-mine/astro-mine-link/issues/5
"""

from __future__ import annotations

from astro_mine.link.cache._digest import (
    CacheKey,
    build_cache_key,
    cache_key,
    canonical_digest,
    hash_file,
    plan_digest,
)
from astro_mine.link.cache._errors import LinkCacheError, PassTimeBudgetError
from astro_mine.link.cache._oracle import (
    PassTimeReport,
    assert_within_budget,
    cross_check_pass_times,
)
from astro_mine.link.cache._store import PlanCache

__all__ = [
    "CacheKey",
    "LinkCacheError",
    "PassTimeBudgetError",
    "PassTimeReport",
    "PlanCache",
    "assert_within_budget",
    "build_cache_key",
    "cache_key",
    "canonical_digest",
    "cross_check_pass_times",
    "hash_file",
    "plan_digest",
]
