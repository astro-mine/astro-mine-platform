"""Locality hints -- pre-warm a shared cache and co-schedule to warm nodes."""

from __future__ import annotations

from pathlib import Path

from astro_mine.cloud.data.cache import PullThroughCache
from astro_mine.cloud.data.chunks import ChunkRef
from astro_mine.cloud.data.locality import (
    CACHE_LABEL,
    ZONE_LABEL,
    co_schedule_affinity,
    prewarm,
    zone_affinity,
)


class _Store:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = objects

    def get(self, key: str) -> bytes:
        return self._objects[key]

    def get_range(self, key: str, start: int, length: int) -> bytes:
        return self._objects[key][start : start + length]

    def size(self, key: str) -> int:
        return len(self._objects[key])


def test_prewarm_counts_cold_fetches(tmp_path: Path) -> None:
    cache = PullThroughCache(_Store({"a": b"1", "b": b"2"}), tmp_path)
    refs = [ChunkRef(key="a"), ChunkRef(key="b")]
    assert prewarm(cache, refs) == 2  # both cold
    assert prewarm(cache, refs) == 0  # now warm, nothing fetched


def test_co_schedule_affinity_prefers_warm_nodes() -> None:
    affinity = co_schedule_affinity("lunar-dem", nodes=["node-1"])
    pref = affinity["nodeAffinity"]["preferredDuringSchedulingIgnoredDuringExecution"][0]
    keys = {e["key"] for e in pref["preference"]["matchExpressions"]}
    assert CACHE_LABEL in keys
    assert "kubernetes.io/hostname" in keys
    assert pref["weight"] == 100


def test_co_schedule_affinity_without_nodes() -> None:
    affinity = co_schedule_affinity("lunar-dem")
    pref = affinity["nodeAffinity"]["preferredDuringSchedulingIgnoredDuringExecution"][0]
    exprs = pref["preference"]["matchExpressions"]
    assert len(exprs) == 1 and exprs[0]["values"] == ["lunar-dem"]


def test_zone_affinity_is_required() -> None:
    affinity = zone_affinity("us-east-1a")
    term = affinity["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][
        "nodeSelectorTerms"
    ][0]
    assert term["matchExpressions"][0]["key"] == ZONE_LABEL
    assert term["matchExpressions"][0]["values"] == ["us-east-1a"]
