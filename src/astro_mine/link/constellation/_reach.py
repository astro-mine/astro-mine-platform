# SPDX-License-Identifier: Apache-2.0
"""Multi-hop *contemporaneous* reachability over the constellation graph (RM-P1-LINK-10).

Given a :class:`~astro_mine.link.products.ConnectivitySampler` built from the constellation
contact plan, this derives **reachable paths** — surface → relay → … → Earth — from the
time-varying contact graph: at an epoch, the set of *simultaneously open* links forms a
graph, and a breadth-first walk finds the fewest-hop path from a source to any target (an
Earth ground station). A surface agent inside a PSR with no direct Earth line-of-sight is
reachable via a relay chain exactly when such a path exists — the acceptance property of
RM-P1-LINK-10.

This is the **instantaneous** multi-hop reach (all hops open at once). Delay-tolerant
*store-and-forward* delivery — a bundle held at a relay until its next contact opens, so a
message can arrive even when no contemporaneous path ever exists — is the separate
:class:`~astro_mine.link.network.DeliveryModel` (RM-P1-LINK-11).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Collection

from astro_mine.core.messages import ContactPlan, Route
from astro_mine.core.messages.enums import NodeRole
from astro_mine.core.units import Epoch, EpochWindow
from astro_mine.link.constellation._errors import LinkConstellationError
from astro_mine.link.products import ConnectivitySampler
from astro_mine.link.windows import ContactWindow, search_windows

__all__ = [
    "build_routes",
    "ground_node_ids",
    "reachability_windows",
    "reachable_route",
    "route_exists",
]


def ground_node_ids(plan: ContactPlan) -> frozenset[str]:
    """The ids of every ``GROUND`` (Earth/DSN) node in ``plan`` — the default reach targets."""
    return frozenset(node.id for node in plan.nodes if node.role is NodeRole.GROUND)


def _adjacency(sampler: ConnectivitySampler, epoch: Epoch) -> dict[str, list[tuple[str, float]]]:
    """The undirected graph of links *open at* ``epoch``: node → sorted (neighbour, latency)."""
    adjacency: dict[str, list[tuple[str, float]]] = {}
    for (a, b), state in sampler.connectivity(epoch).items():
        if not state.reachable:
            continue
        latency = state.latency_s if state.latency_s is not None else 0.0
        adjacency.setdefault(a, []).append((b, latency))
        adjacency.setdefault(b, []).append((a, latency))
    for neighbours in adjacency.values():
        neighbours.sort()  # deterministic BFS expansion order
    return adjacency


def reachable_route(
    sampler: ConnectivitySampler,
    source: str,
    targets: Collection[str],
    epoch: Epoch,
) -> Route | None:
    """The fewest-hop :class:`~astro_mine.core.messages.Route` from ``source`` to any
    ``targets`` over the links open at ``epoch``, or ``None`` if none is reachable.

    Breadth-first over the open-link graph (deterministic: neighbours are expanded in sorted
    order, so the returned path is stable for a given plan + epoch). ``total_latency_s`` sums
    the per-hop latencies along the path (unknown hop latencies count as zero — a floor), and
    ``earliest_delivery_tdb_s`` is ``epoch + total_latency_s``. Raises
    :class:`LinkConstellationError` if ``source`` is not a node of the plan.
    """
    if source not in sampler.nodes:
        raise LinkConstellationError(f"source {source!r} is not a node of the contact plan")
    target_set = set(targets)
    if source in target_set:
        return Route(
            source=source,
            dest=source,
            hops=[source],
            total_latency_s=0.0,
            earliest_delivery_tdb_s=epoch.tdb_seconds,
            store_and_forward=False,
        )
    adjacency = _adjacency(sampler, epoch)
    # BFS carrying the path + accumulated latency; first target dequeued is fewest-hop.
    queue: deque[tuple[str, list[str], float]] = deque([(source, [source], 0.0)])
    seen: set[str] = {source}
    while queue:
        node, path, latency = queue.popleft()
        for neighbour, hop_latency in adjacency.get(node, ()):
            if neighbour in seen:
                continue
            next_path = [*path, neighbour]
            next_latency = latency + hop_latency
            if neighbour in target_set:
                return Route(
                    source=source,
                    dest=neighbour,
                    hops=next_path,
                    total_latency_s=next_latency,
                    earliest_delivery_tdb_s=epoch.tdb_seconds + next_latency,
                    store_and_forward=False,
                )
            seen.add(neighbour)
            queue.append((neighbour, next_path, next_latency))
    return None


def route_exists(
    sampler: ConnectivitySampler, source: str, targets: Collection[str], epoch: Epoch
) -> bool:
    """Whether any ``targets`` node is reachable from ``source`` at ``epoch`` (multi-hop)."""
    return reachable_route(sampler, source, targets, epoch) is not None


def reachability_windows(
    sampler: ConnectivitySampler,
    source: str,
    targets: Collection[str],
    window: EpochWindow,
    step_s: float,
    *,
    target_label: str = "earth",
    refine_s: float | None = None,
) -> list[ContactWindow]:
    """The intervals over ``window`` during which ``source`` can reach ``targets`` multi-hop.

    Reuses the shared rise/set :func:`~astro_mine.link.windows.search_windows`, thresholding
    the multi-hop :func:`route_exists` predicate instead of a single-pair LOS — so a surface
    agent's *end-to-end* Earth reachability (through whatever relay chain is open) is itself a
    first-class time-varying product. ``target_label`` names the aggregate target endpoint on
    the emitted windows (default ``"earth"``)."""
    return search_windows(
        (source, target_label),
        lambda epoch: route_exists(sampler, source, targets, epoch),
        window,
        step_s,
        refine_s=refine_s,
    )


def build_routes(
    sampler: ConnectivitySampler,
    sources: Collection[str],
    targets: Collection[str],
    epoch: Epoch,
) -> list[Route]:
    """A :class:`~astro_mine.core.messages.Route` per reachable source at ``epoch``.

    Sources with no path to any target at ``epoch`` are omitted. Deterministic in ``sources``
    order; suitable for populating :attr:`ContactPlan.routes` at a representative epoch."""
    routes: list[Route] = []
    for source in sources:
        route = reachable_route(sampler, source, targets, epoch)
        if route is not None:
            routes.append(route)
    return routes
