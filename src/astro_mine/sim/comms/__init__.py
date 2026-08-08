"""Comms masking -- consume a Core ContactPlan into per-tick masks (RM-P0-SIM-08).

Turns Link connectivity into per-tick, per-agent observation/comms masks applied through the
Core Environment API, so a policy literally cannot observe or message an unreachable peer --
making PSR comms-denial and relay-window gaps real in simulation (sim.md §3; LUNAR-TR-003).

Sim consumes the **Core** contract, not the Link package: the input is a Core
:class:`~astro_mine.core.messages.model.ContactPlan` and the output a Core
:class:`~astro_mine.core.messages.model.CommsObservationMask`, exactly as Sim consumes Worlds
and Prospect through their Core contracts (``WorldProvider`` / ``ResourceField``) and ships its
own reference realizations. :class:`ReferenceConnectivitySampler` is that always-works local
tier. The :class:`ConnectivitySource` seam is *structurally identical* to Link's
``ConnectivitySampler`` (RM-P0-LINK-04), so a run that has Link installed can inject Link's own
(index-optimized) sampler with no import and no Sim change.

Backlog: RM-P0-SIM-08 -- astro-mine-sim#8
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol, runtime_checkable

from astro_mine.core.messages.enums import NodeRole
from astro_mine.core.messages.model import (
    CommsObservationMask,
    ContactInterval,
    ContactPlan,
    Observation,
    PeerLink,
)
from astro_mine.core.units import Epoch

__all__ = [
    "ConnectivitySource",
    "ReferenceConnectivitySampler",
    "apply_comms_mask",
]


def _content_digest(data: Any) -> str:
    """The SHA-256 of ``data``'s canonical JSON form -- the ContactPlan's content identity.

    Same canonical form (``sort_keys=True``, compact separators) as
    :meth:`~astro_mine.sim.runtime.episode.Trace.to_canonical_json` and the run-provenance digest
    (RM-P0-SIM-09), kept leaf-local so ``comms`` depends only on Core; all three fold into Core's
    shared content-hash helper (core#19) when it ships.
    """
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


@runtime_checkable
class ConnectivitySource(Protocol):
    """The per-tick connectivity seam the Simulator drives for comms masking (RM-P0-SIM-08).

    Structurally identical to Link's ``ConnectivitySampler`` (RM-P0-LINK-04): a caller that has
    Link installed can inject Link's optimized sampler directly, since Sim depends only on this
    shape, never on the Link package. :meth:`nodes` names the contact graph so the Simulator can
    mask exactly the agents that are comms nodes; :meth:`comms_mask` answers the per-epoch mask.
    """

    @property
    def nodes(self) -> tuple[str, ...]:
        """Every node id in the contact graph (surface agents, relays, and ground stations)."""
        ...

    def comms_mask(self, agent_id: str, epoch: Epoch) -> CommsObservationMask:
        """The per-tick :class:`CommsObservationMask` for ``agent_id`` at ``epoch``."""
        ...


def _pair_key(node_a: str, node_b: str) -> tuple[str, str]:
    """The canonical, order-independent key for an ``a<->b`` link (reachability is symmetric)."""
    return (node_a, node_b) if node_a <= node_b else (node_b, node_a)


def _representative_latency(interval: ContactInterval) -> float | None:
    """The interval's representative latency -- its mean if annotated, else the light-time floor."""
    if interval.mean_latency_s is not None:
        return interval.mean_latency_s
    return interval.min_latency_s


class ReferenceConnectivitySampler:
    """A deterministic :class:`ConnectivitySource` over a Core ContactPlan -- the local tier.

    The plan's intervals are ground truth; a per-epoch boolean/rate snapshot is *derived* from
    them (link.md §2.3), never the other way around. A pair is reachable at ``t`` iff a contact
    interval is open over the half-open span ``[start_tdb_s, end_tdb_s)``; windows for a pair are
    disjoint by contract. The mask lists every other node as a
    :class:`~astro_mine.core.messages.model.PeerLink` -- so an agent knows a peer is *denied*, not
    merely absent -- and ``earth_contact`` is true iff any reachable peer is a ground node.

    Reimplements the ContactPlan reading so Sim's offline tier needs no Link install; a run that
    has Link can inject Link's own ``ConnectivitySampler`` instead (both read the same plan).
    """

    def __init__(self, plan: ContactPlan) -> None:
        self._roles: dict[str, NodeRole] = {node.id: node.role for node in plan.nodes}
        self._node_ids: tuple[str, ...] = tuple(self._roles)
        by_pair: dict[tuple[str, str], list[ContactInterval]] = {}
        for interval in plan.intervals:
            by_pair.setdefault(_pair_key(interval.node_a, interval.node_b), []).append(interval)
        self._by_pair = by_pair
        self._content_hash = _content_digest(plan.model_dump(mode="json"))

    @property
    def nodes(self) -> tuple[str, ...]:
        """Every node id in the contact graph, in declaration order."""
        return self._node_ids

    @property
    def content_hash(self) -> str:
        """The content hash of the source ContactPlan -- its input identity in run provenance."""
        return self._content_hash

    def _open_interval(self, node_a: str, node_b: str, epoch: Epoch) -> ContactInterval | None:
        """The contact interval open for the pair at ``epoch``, or ``None`` when denied."""
        t = epoch.tdb_seconds
        for interval in self._by_pair.get(_pair_key(node_a, node_b), ()):
            if interval.start_tdb_s <= t < interval.end_tdb_s:
                return interval
        return None

    def comms_mask(self, agent_id: str, epoch: Epoch) -> CommsObservationMask:
        """The Core :class:`CommsObservationMask` for ``agent_id`` at ``epoch``.

        Raises :class:`KeyError` for an ``agent_id`` that is not a node of the contact plan --
        the Simulator only asks for agents in :attr:`nodes`, so this guards misuse, not the loop.
        """
        if agent_id not in self._roles:
            raise KeyError(f"unknown agent {agent_id!r}: not a node of the contact plan")
        links: list[PeerLink] = []
        earth_contact = False
        for peer in self._node_ids:
            if peer == agent_id:
                continue
            interval = self._open_interval(agent_id, peer, epoch)
            reachable = interval is not None
            links.append(
                PeerLink(
                    peer=peer,
                    reachable=reachable,
                    rate_bps=interval.max_rate_bps if interval is not None else None,
                    latency_s=_representative_latency(interval) if interval is not None else None,
                    margin_db=interval.margin_db if interval is not None else None,
                )
            )
            if reachable and self._roles[peer] == NodeRole.GROUND:
                earth_contact = True
        return CommsObservationMask(agent_id=agent_id, links=links, earth_contact=earth_contact)


def apply_comms_mask(observation: Observation, mask: CommsObservationMask) -> Observation:
    """Return ``observation`` with its per-tick comms mask applied (RM-P0-SIM-08).

    The one primitive that puts a :class:`CommsObservationMask` onto the Core Environment surface
    (``Observation.comms``), so a policy cannot observe or message an unreachable peer. Pure:
    returns a copy and leaves the input untouched.
    """
    return observation.model_copy(update={"comms": mask})
