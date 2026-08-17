# SPDX-License-Identifier: Apache-2.0
"""On-demand connectivity + per-tick comms masks over a ContactPlan (RM-P0-LINK-04).

The two agent-facing products Sim drives each tick: a :class:`ConnectivitySampler` that
answers ``connectivity(epoch)`` from a precomputed
:class:`~astro_mine.core.messages.ContactPlan` (indexed for sub-millisecond lookup, so it
never bottlenecks the step loop — link.md §8), and the per-agent
:class:`~astro_mine.core.messages.CommsObservationMask` it emits through the Core
Environment API so a policy literally cannot observe or message an unreachable peer
(link.md §3; LUNAR-TR-003).

Connectivity is a function of epoch by construction (link.md §2.3): the plan's intervals are
the ground truth, and a boolean/rate snapshot is derived from them, never the other way
around. Reachability is symmetric — an ``a<->b`` line of sight is one link regardless of the
order it was stored in.

Backlog: RM-P0-LINK-04 -- astro-mine-link#4
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from hashlib import sha256

from astro_mine.core.messages import (
    CommsObservationMask,
    ContactInterval,
    ContactPlan,
    PeerLink,
    contact_plan_to_wire,
)
from astro_mine.core.messages.enums import NodeRole
from astro_mine.core.units import Epoch
from astro_mine.link.products._errors import LinkProductsError

__all__ = ["ConnectivitySampler", "LinkState"]


@dataclass(frozen=True, slots=True)
class LinkState:
    """Connectivity of one node pair at one epoch — the snapshot ``connectivity`` returns.

    ``reachable`` is whether a contact interval is open at the queried epoch; ``rate_bps`` /
    ``latency_s`` / ``margin_db`` carry that interval's annotated link quality (``None`` when
    the pair is unreachable, or the plan left the field unset).
    """

    reachable: bool
    rate_bps: float | None = None
    latency_s: float | None = None
    margin_db: float | None = None


def _pair_key(a: str, b: str) -> tuple[str, str]:
    """The canonical, order-independent key for an ``a<->b`` link."""
    return (a, b) if a <= b else (b, a)


def _latency(interval: ContactInterval) -> float | None:
    """The representative latency of an interval — mean if present, else the light-time floor."""
    if interval.mean_latency_s is not None:
        return interval.mean_latency_s
    return interval.min_latency_s


class ConnectivitySampler:
    """Answers per-epoch connectivity + comms masks from a :class:`ContactPlan`.

    Built once from a plan; every query is read-only, so the same sampler serves every Sim
    tick without re-deriving geometry. Intervals are grouped per link and indexed by start
    time, so a query bisects to the single candidate interval (windows for a pair are
    disjoint) rather than scanning them all.
    """

    def __init__(self, plan: ContactPlan) -> None:
        self._roles: dict[str, NodeRole] = {node.id: node.role for node in plan.nodes}
        self._node_ids: tuple[str, ...] = tuple(self._roles)
        by_pair: dict[tuple[str, str], list[ContactInterval]] = {}
        for interval in plan.intervals:
            by_pair.setdefault(_pair_key(interval.node_a, interval.node_b), []).append(interval)
        for intervals in by_pair.values():
            intervals.sort(key=lambda iv: iv.start_tdb_s)
        self._by_pair = by_pair
        self._starts = {key: [iv.start_tdb_s for iv in ivs] for key, ivs in by_pair.items()}
        self._content_hash = f"sha256:{sha256(contact_plan_to_wire(plan)).hexdigest()}"

    @property
    def nodes(self) -> tuple[str, ...]:
        """Every node id in the contact graph, in declaration order."""
        return self._node_ids

    @property
    def content_hash(self) -> str:
        """The content address of the source plan — its input identity in a Sim run's provenance.

        The same digest :func:`~astro_mine.link.cache.plan_digest` computes (SHA-256 of Core's
        byte-stable ``contact_plan_to_wire``), in the platform ``sha256:<hex>`` form, and the same
        value the plan's published Hub manifest carries as ``provenance.digest``. Sim's episode
        provenance duck-types this attribute off the injected connectivity source
        (``source_content_hashes["contact_plan"]``, RM-P0-SIM-09), so a run driven by a Link plan
        records *which* plan it was driven by — the comms half of a reproducible benchmark
        (link.md §5; conventions.md §1.5). Computed inline rather than via
        :mod:`astro_mine.link.cache` to keep ``products`` free of a cache import cycle.
        """
        return self._content_hash

    def reachable(self, node_a: str, node_b: str, epoch: Epoch) -> LinkState:
        """The :class:`LinkState` of the ``node_a<->node_b`` link at ``epoch``.

        Returns an unreachable state (all quality fields ``None``) when the pair has no
        contact interval open at ``epoch`` — including a pair with no interval at all.
        """
        key = _pair_key(node_a, node_b)
        intervals = self._by_pair.get(key)
        if intervals is None:
            return LinkState(reachable=False)
        t = epoch.tdb_seconds
        idx = bisect.bisect_right(self._starts[key], t) - 1
        if idx < 0:
            return LinkState(reachable=False)
        interval = intervals[idx]
        if interval.end_tdb_s <= t:  # the nearest-starting window already closed
            return LinkState(reachable=False)
        return LinkState(
            reachable=True,
            rate_bps=interval.max_rate_bps,
            latency_s=_latency(interval),
            margin_db=interval.margin_db,
        )

    def connectivity(self, epoch: Epoch) -> dict[tuple[str, str], LinkState]:
        """Every known link's :class:`LinkState` at ``epoch``, keyed by canonical node pair.

        Includes links that exist in the plan but are currently closed (``reachable=False``),
        so a planner sees the whole comms graph, not only what is open this tick.
        """
        return {key: self.reachable(key[0], key[1], epoch) for key in self._by_pair}

    def comms_mask(self, agent_id: str, epoch: Epoch) -> CommsObservationMask:
        """The Core :class:`CommsObservationMask` for ``agent_id`` at ``epoch``.

        Lists every other node as a :class:`~astro_mine.core.messages.PeerLink` with its
        current ``reachable`` flag and link quality — so the agent knows a peer is denied,
        not merely absent. ``earth_contact`` is true iff any reachable peer is a ground
        (Earth/DSN) node. Raises :class:`LinkProductsError` for an unknown ``agent_id``.
        """
        if agent_id not in self._roles:
            raise LinkProductsError(
                f"unknown agent {agent_id!r}; it is not a node of the contact plan"
            )
        links: list[PeerLink] = []
        earth_contact = False
        for peer in self._node_ids:
            if peer == agent_id:
                continue
            state = self.reachable(agent_id, peer, epoch)
            links.append(
                PeerLink(
                    peer=peer,
                    reachable=state.reachable,
                    rate_bps=state.rate_bps,
                    latency_s=state.latency_s,
                    margin_db=state.margin_db,
                )
            )
            if state.reachable and self._roles[peer] == NodeRole.GROUND:
                earth_contact = True
        return CommsObservationMask(agent_id=agent_id, links=links, earth_contact=earth_contact)

    def comms_masks(
        self, epoch: Epoch, *, agents: tuple[str, ...] | None = None
    ) -> dict[str, CommsObservationMask]:
        """A :class:`CommsObservationMask` per agent at ``epoch`` (all nodes by default)."""
        agent_ids = self._node_ids if agents is None else agents
        return {agent_id: self.comms_mask(agent_id, epoch) for agent_id in agent_ids}
