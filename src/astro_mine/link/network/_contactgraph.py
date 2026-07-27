"""The CGR-style contact graph derived from a ContactPlan (RM-P1-LINK-11).

A contact-graph-routing (CGR) view of the constellation: the plan's per-pair contact
intervals become a set of time-ordered, undirected :class:`Contact`\\ s, indexed per node so
the store-and-forward :class:`~astro_mine.link.network.DeliveryModel` can walk "which contacts
does this node have, and when". This is the *modeling* substrate an external DTN (Bundle
Protocol) would route over — Link supplies the contact plan, it does not ship bundles at
runtime (link.md §1, §11 "abstract store-and-forward over a CGR-style contact graph").
"""

from __future__ import annotations

from dataclasses import dataclass

from astro_mine.core.messages import ContactInterval, ContactPlan
from astro_mine.core.units import Epoch, EpochWindow, TimeScale
from astro_mine.link.network._errors import LinkNetworkError

__all__ = ["Contact", "ContactGraph"]


def _interval_window(interval: ContactInterval) -> EpochWindow:
    """The interval's contact span as a typed :class:`EpochWindow` (RFC-0007).

    Prefers the interval's typed ``window`` field when the producer populated it (RFC-0007:
    consumers MUST prefer the typed field), and reconstructs one from the ``*_tdb_s`` primitives
    (tagged :data:`TimeScale.TDB` by the field's naming contract) only as a fallback.
    """
    if interval.window is not None:
        return interval.window
    return EpochWindow(
        start=Epoch(tdb_seconds=interval.start_tdb_s, scale=TimeScale.TDB),
        end=Epoch(tdb_seconds=interval.end_tdb_s, scale=TimeScale.TDB),
    )


@dataclass(frozen=True, slots=True)
class Contact:
    """One undirected contact interval between two nodes, with its modeled link quality.

    ``window`` is the contact's typed :class:`~astro_mine.core.units.EpochWindow` span — the
    frame/time vocabulary carried explicitly rather than as a scale-by-naming ``float`` (RFC-0007;
    conventions.md §5). ``rate_bps`` / ``latency_s`` are the interval's achievable rate and
    one-way latency (``None`` when the plan left them unset — geometry-only)."""

    node_a: str
    node_b: str
    window: EpochWindow
    rate_bps: float | None = None
    latency_s: float | None = None

    def other(self, node: str) -> str:
        """The endpoint of this contact that is not ``node``."""
        if node == self.node_a:
            return self.node_b
        if node == self.node_b:
            return self.node_a
        raise LinkNetworkError(
            f"node {node!r} is not an endpoint of contact {self.node_a}-{self.node_b}"
        )

    def open_at(self, epoch: Epoch) -> bool:
        """Whether the contact is open at ``epoch`` (half-open ``[start, end)``)."""
        t = epoch.tdb_seconds
        return self.window.start.tdb_seconds <= t < self.window.end.tdb_seconds


class ContactGraph:
    """A per-node index of time-ordered contacts, built once from a :class:`ContactPlan`.

    ``contacts_from(node)`` returns that node's contacts sorted by start time — the adjacency
    the delivery model iterates. Immutable and cheap to query; the same graph serves every
    delivery query, mirroring how :class:`~astro_mine.link.products.ConnectivitySampler` serves
    every per-tick connectivity query.
    """

    def __init__(self, contacts: list[Contact], *, node_ids: frozenset[str] | None = None) -> None:
        by_node: dict[str, list[Contact]] = {}
        for contact in contacts:
            by_node.setdefault(contact.node_a, []).append(contact)
            by_node.setdefault(contact.node_b, []).append(contact)
        for node_contacts in by_node.values():
            node_contacts.sort(key=lambda c: (c.window.start.tdb_seconds, c.window.end.tdb_seconds))
        self._by_node = by_node
        self._contacts = tuple(contacts)
        # Declared nodes: the plan's full node set when known (so a *declared but isolated*
        # node is distinguishable from an *unknown* id), else just those carrying contacts.
        self._node_ids = node_ids if node_ids is not None else frozenset(by_node)

    @classmethod
    def from_plan(cls, plan: ContactPlan) -> ContactGraph:
        """Derive the contact graph from a Core :class:`ContactPlan`'s intervals."""
        contacts = [
            Contact(
                node_a=interval.node_a,
                node_b=interval.node_b,
                window=_interval_window(interval),
                rate_bps=interval.max_rate_bps,
                latency_s=(
                    interval.mean_latency_s
                    if interval.mean_latency_s is not None
                    else interval.min_latency_s
                ),
            )
            for interval in plan.intervals
        ]
        return cls(contacts, node_ids=frozenset(node.id for node in plan.nodes) or None)

    @property
    def nodes(self) -> frozenset[str]:
        """Every node that has at least one contact."""
        return frozenset(self._by_node)

    @property
    def node_ids(self) -> frozenset[str]:
        """Every *declared* node (the plan's full node set when built via :meth:`from_plan`)."""
        return self._node_ids

    @property
    def contacts(self) -> tuple[Contact, ...]:
        """Every contact in the graph, in construction order."""
        return self._contacts

    def contacts_from(self, node: str) -> list[Contact]:
        """``node``'s contacts, sorted by start time (empty if it has none)."""
        return self._by_node.get(node, [])
