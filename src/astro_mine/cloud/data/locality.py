"""Locality hints -- pre-warm shared datasets and co-schedule to warm nodes.

For sweeps that re-read the same slices, Cloud pre-warms a shared cache and co-schedules jobs
onto cache-warm nodes, and co-locates the cluster with the object store to avoid cross-zone
egress (``cloud.md`` §5, §8). :func:`prewarm` fills a
:class:`~astro_mine.cloud.data.cache.PullThroughCache` for a set of chunks;
:func:`co_schedule_affinity` and :func:`zone_affinity` emit the K8s affinity that steers pods
to warm nodes / the store's zone.

Backlog: RM-P1-CLOUD-04 -- astro-mine-cloud#15
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_mine.cloud.k8s import Manifest

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from astro_mine.cloud.data.cache import PullThroughCache
    from astro_mine.cloud.data.chunks import ChunkRef

__all__ = ["CACHE_LABEL", "ZONE_LABEL", "co_schedule_affinity", "prewarm", "zone_affinity"]

#: Node label marking which shared dataset a node has warmed in local scratch.
CACHE_LABEL = "astro-mine.org/warm-dataset"
#: The standard topology label the object store's zone is expressed in.
ZONE_LABEL = "topology.kubernetes.io/zone"


def prewarm(cache: PullThroughCache, refs: Iterable[ChunkRef]) -> int:
    """Pull *refs* into *cache*; return how many were fetched from remote (were cold)."""
    before = cache.misses
    for ref in refs:
        cache.read_chunk(ref)
    return cache.misses - before


def co_schedule_affinity(dataset: str, *, nodes: Sequence[str] | None = None) -> Manifest:
    """Preferred node-affinity steering a pod onto nodes warm for *dataset*.

    ``preferred`` (not ``required``): locality is an optimization, so a pod still schedules on a
    cold node under contention -- degrade, don't collapse (``cloud.md`` §2 principle 9).
    """
    match_expressions: list[Manifest] = [
        {"key": CACHE_LABEL, "operator": "In", "values": [dataset]}
    ]
    if nodes:
        match_expressions.append(
            {"key": "kubernetes.io/hostname", "operator": "In", "values": list(nodes)}
        )
    return {
        "nodeAffinity": {
            "preferredDuringSchedulingIgnoredDuringExecution": [
                {"weight": 100, "preference": {"matchExpressions": match_expressions}}
            ]
        }
    }


def zone_affinity(zone: str) -> Manifest:
    """Required node-affinity pinning a pod to the object store's *zone* (egress avoidance)."""
    return {
        "nodeAffinity": {
            "requiredDuringSchedulingIgnoredDuringExecution": {
                "nodeSelectorTerms": [
                    {"matchExpressions": [{"key": ZONE_LABEL, "operator": "In", "values": [zone]}]}
                ]
            }
        }
    }
