# SPDX-License-Identifier: Apache-2.0
"""The abstract store-and-forward delivery model (RM-P1-LINK-11).

Given the CGR-style :class:`~astro_mine.link.network.ContactGraph` and a message
(size, source→dest, send epoch), :class:`DeliveryModel` returns a **modeled delivery time**
(or non-delivery) under **store-and-forward**: a bundle may be held at a relay until its next
contact opens, so it can arrive even when no *contemporaneous* end-to-end path ever exists —
the delay-tolerant behavior that makes a comms-denied PSR scenario realistic (link.md §1,
§11). This is the Bundle-Protocol *concepts* abstraction, not a live DTN agent: Link models
delivery and supplies the contact plan an external DTN would route over; it never ships
application traffic at runtime (that is the Ops/Bridge data plane, link.md §1).

**Multi-fidelity dial** (link.md §11 "delivery models" plugin axis): the same query runs in
either fidelity without an API change — ``"instantaneous"`` (a hop is usable only if its
contact is open *now*; equivalent to contemporaneous multi-hop, so no delivery when the path
is broken at send time) or ``"store_and_forward"`` (a hop may wait for a future contact).
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import Literal

from astro_mine.core.messages import Route
from astro_mine.core.units import Epoch
from astro_mine.link.network._contactgraph import Contact, ContactGraph
from astro_mine.link.network._errors import LinkNetworkError

__all__ = ["Delivery", "DeliveryFidelity", "DeliveryModel"]

#: The two delivery fidelities the dial selects between (link.md §11).
DeliveryFidelity = Literal["instantaneous", "store_and_forward"]


@dataclass(frozen=True, slots=True)
class Delivery:
    """The modeled outcome of one message delivery query.

    ``delivered`` is whether the message reaches ``dest`` within the horizon; when it does,
    ``arrival`` is the earliest arrival :class:`~astro_mine.core.units.Epoch` (the epoch carried
    with its scale, not a scale-by-naming ``float`` — RFC-0007; conventions.md §5), ``latency_s``
    the end-to-end delay (``arrival - send`` in SI seconds, including any store-and-forward wait),
    and ``hops`` the node path. ``route`` is the equivalent Core
    :class:`~astro_mine.core.messages.Route`. A non-delivery is an honest modeled result
    (``delivered=False``, everything else ``None``/empty), not an error (link.md §2.9)."""

    delivered: bool
    source: str
    dest: str
    arrival: Epoch | None = None
    latency_s: float | None = None
    hops: tuple[str, ...] = ()
    route: Route | None = None


@dataclass(frozen=True)
class DeliveryModel:
    """Store-and-forward delivery-time modeling over a :class:`ContactGraph`.

    ``turnaround_s`` is a fixed per-hop processing/turnaround delay added at each relay.
    Reduced-order: a bundle must be *fully* transmitted within a single contact (no
    fragmentation across contacts); infinite buffers; FIFO forwarding. These are the Phase-1
    modeling assumptions — a full Bundle-Protocol-fidelity plugin is deferred (link.md §11).
    """

    graph: ContactGraph
    turnaround_s: float = 0.0
    _known: frozenset[str] = field(init=False, repr=False, compare=False, default=frozenset())

    def __post_init__(self) -> None:
        object.__setattr__(self, "_known", self.graph.node_ids | self.graph.nodes)

    def deliver(
        self,
        source: str,
        dest: str,
        *,
        send: Epoch,
        size_bits: float = 0.0,
        fidelity: DeliveryFidelity = "store_and_forward",
        horizon_s: float | None = None,
    ) -> Delivery:
        """Model delivery of a ``size_bits`` message ``source → dest`` sent at ``send``.

        ``send`` is a typed :class:`~astro_mine.core.units.Epoch` (RFC-0007; conventions.md §5),
        carried through to the returned ``arrival`` in the same scale. Runs an earliest-arrival
        search over the contact graph. In ``store_and_forward`` a hop may wait for a future
        contact; in ``instantaneous`` a hop's contact must already be open. ``horizon_s`` bounds
        how far ahead delivery is considered (a bundle that would arrive later is a non-delivery).
        ``size_bits`` with a contact's ``rate_bps`` gives the transmission time; a bundle that
        cannot fit inside a single contact window skips that contact. Raises
        :class:`LinkNetworkError` only for an **unknown** endpoint id.
        """
        for endpoint, label in ((source, "source"), (dest, "dest")):
            if self._known and endpoint not in self._known:
                raise LinkNetworkError(f"unknown {label} node {endpoint!r} for delivery query")
        if size_bits < 0.0:
            raise LinkNetworkError(f"size_bits must be non-negative, got {size_bits}")
        # The earliest-arrival search is a bare-float numeric kernel (a Dijkstra over a heap of
        # TDB-second keys); the typed Epoch is unwrapped here and re-wrapped on the results.
        send_tdb_s = send.tdb_seconds
        if source == dest:
            return Delivery(
                delivered=True,
                source=source,
                dest=dest,
                arrival=send,
                latency_s=0.0,
                hops=(source,),
                route=Route(
                    source=source,
                    dest=dest,
                    hops=[source],
                    total_latency_s=0.0,
                    earliest_delivery_tdb_s=send_tdb_s,
                    earliest_delivery=send,
                    store_and_forward=False,
                ),
            )

        best: dict[str, float] = {source: send_tdb_s}
        prev: dict[str, str] = {}
        heap: list[tuple[float, str]] = [(send_tdb_s, source)]
        while heap:
            t_u, u = heapq.heappop(heap)
            if t_u > best.get(u, float("inf")):
                continue  # a stale heap entry superseded by an earlier arrival
            if u == dest:
                break
            for contact in self.graph.contacts_from(u):
                arrival = self._hop_arrival(contact, t_u, size_bits, fidelity)
                if arrival is None:
                    continue
                if horizon_s is not None and arrival > send_tdb_s + horizon_s:
                    continue
                v = contact.other(u)
                if arrival < best.get(v, float("inf")):
                    best[v] = arrival
                    prev[v] = u
                    heapq.heappush(heap, (arrival, v))

        if dest not in best:
            return Delivery(delivered=False, source=source, dest=dest)
        hops = _reconstruct(prev, source, dest)
        arrival_s = best[dest]
        latency = arrival_s - send_tdb_s
        arrival_epoch = Epoch(tdb_seconds=arrival_s, scale=send.scale)
        return Delivery(
            delivered=True,
            source=source,
            dest=dest,
            arrival=arrival_epoch,
            latency_s=latency,
            hops=tuple(hops),
            route=Route(
                source=source,
                dest=dest,
                hops=hops,
                total_latency_s=latency,
                earliest_delivery_tdb_s=arrival_s,
                earliest_delivery=arrival_epoch,
                store_and_forward=(fidelity == "store_and_forward"),
            ),
        )

    def _hop_arrival(
        self, contact: Contact, t_u: float, size_bits: float, fidelity: DeliveryFidelity
    ) -> float | None:
        """The arrival time at ``contact``'s far endpoint if usable from ``t_u``, else ``None``.

        ``t_u`` and the return are bare TDB seconds — the kernel's numeric currency; the contact's
        typed :class:`~astro_mine.core.units.EpochWindow` is read down to its float bounds here."""
        start_s = contact.window.start.tdb_seconds
        end_s = contact.window.end.tdb_seconds
        if fidelity == "instantaneous":
            if not (start_s <= t_u < end_s):
                return None
            usable_start = t_u
        else:
            usable_start = max(start_s, t_u)
            if usable_start >= end_s:
                return None  # the contact has already closed relative to t_u
        tx = 0.0
        if size_bits > 0.0 and contact.rate_bps is not None and contact.rate_bps > 0.0:
            tx = size_bits / contact.rate_bps
        if usable_start + tx > end_s:
            return None  # the bundle does not fit inside this contact window
        latency = contact.latency_s if contact.latency_s is not None else 0.0
        return usable_start + tx + latency + self.turnaround_s


def _reconstruct(prev: dict[str, str], source: str, dest: str) -> list[str]:
    """The node path ``source → … → dest`` from the predecessor map."""
    path = [dest]
    node = dest
    while node != source:
        node = prev[node]
        path.append(node)
    path.reverse()
    return path
