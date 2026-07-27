"""The content-addressed ContactPlan cache (RM-P0-LINK-05).

A store keyed by :class:`~astro_mine.link.cache.CacheKey` digest: identical pinned inputs
(kernels / DEM / node-set / epoch / config) return the exact prior plan instead of recomputing
it, so a comms-denied benchmark reproduces from pinned inputs (link.md §5; conventions.md
§1.5). Plans persist as Core's byte-stable wire form, so a **re-run** — a fresh process
pointed at the same cache root — is a hit that reproduces the plan byte-for-byte, not a
lookalike.

Backlog: RM-P0-LINK-05 -- https://github.com/astro-mine/astro-mine-link/issues/5
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from astro_mine.core.messages import ContactPlan, contact_plan_from_wire, contact_plan_to_wire
from astro_mine.link.cache._digest import CacheKey

__all__ = ["PlanCache"]


class PlanCache:
    """A content-addressed cache of :class:`ContactPlan`\\ s keyed by :class:`CacheKey`.

    In-memory by default; pass ``root`` to also persist plans under
    ``<root>/<digest>.pb`` (the wire form), so the cache survives across runs. Lookups check
    memory first, then disk; a disk hit is deserialized through Core's ``contact_plan_from_wire``,
    reproducing the stored plan exactly.
    """

    def __init__(self, root: str | Path | None = None) -> None:
        self._root: Path | None = Path(root) if root is not None else None
        if self._root is not None:
            self._root.mkdir(parents=True, exist_ok=True)
        self._memory: dict[str, bytes] = {}

    def _path(self, digest: str) -> Path:
        assert self._root is not None
        return self._root / f"{digest}.pb"

    def _load_bytes(self, key: CacheKey) -> bytes | None:
        digest = key.digest
        data = self._memory.get(digest)
        if data is None and self._root is not None:
            path = self._path(digest)
            if path.is_file():
                data = path.read_bytes()
                self._memory[digest] = data
        return data

    def __contains__(self, key: CacheKey) -> bool:
        return self._load_bytes(key) is not None

    def get(self, key: CacheKey) -> ContactPlan | None:
        """The cached plan for ``key``, or ``None`` on a miss."""
        data = self._load_bytes(key)
        return None if data is None else contact_plan_from_wire(data)

    def put(self, key: CacheKey, plan: ContactPlan) -> str:
        """Store ``plan`` under ``key`` (memory + disk if rooted); returns the content digest."""
        digest = key.digest
        data = contact_plan_to_wire(plan)
        self._memory[digest] = data
        if self._root is not None:
            self._path(digest).write_bytes(data)
        return digest

    def resolve(self, key: CacheKey, compute: Callable[[], ContactPlan]) -> ContactPlan:
        """Return the cached plan for ``key``, else run ``compute`` once, store, and return it."""
        cached = self.get(key)
        if cached is not None:
            return cached
        plan = compute()
        self.put(key, plan)
        return plan
