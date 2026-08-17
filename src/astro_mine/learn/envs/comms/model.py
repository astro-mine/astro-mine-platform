# SPDX-License-Identifier: Apache-2.0
"""``CommsModel`` — the declarative comms-regime channel over the SwarmEnv stream.

The model is a *pure function of (config, observation stream, seed)* applied inside the
environment, so every algorithm — independent, CTDE, or comms-learning — sees the identical
masking/drop/delay and comms-stress results stay comparable across algorithms (learn.md §2,
§3). It never imports :mod:`astro_mine.link`: when Link is present it reads the reachable
set / rate / latency off the Core :class:`~astro_mine.core.messages.model.CommsObservationMask`
already on each observation; otherwise it synthesizes the same structure from neighbour
geometry — the *same* wrapper API either way.

Each tick, for each agent, a candidate peer link crosses four stages — gate → budget →
drop → delay — and the agent observes a peer through its most recently *delivered* (possibly
stale) message. The single RNG stream is seeded from the run seed mixed with a fixed salt,
so the drop/delay realization is reproducible and independent of how far the environment's
own RNG has advanced.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.model import (
    CommsObservationMask,
    Observation,
    PeerLink,
    StateSample,
)
from astro_mine.learn.envs.comms.config import CommsModelConfig
from astro_mine.learn.envs.comms.ledger import CommsLedger, LinkTally

__all__ = ["CommsModel"]

_INF = float("inf")


class _Buffered:
    """One in-flight/last-known message about a peer: the release tick and its payload."""

    __slots__ = ("link", "release_tick", "state")

    def __init__(self, release_tick: int, state: StateSample | None, link: PeerLink) -> None:
        self.release_tick = release_tick
        self.state = state
        self.link = link


class CommsModel:
    """A declarative, composable comms regime applied to a SwarmEnv observation stream.

    Construct from a :class:`~astro_mine.learn.envs.comms.config.CommsModelConfig`, call
    :meth:`reset` with the episode seed, then :meth:`apply` to each per-tick observation
    map. :meth:`ledger` exposes the comms-budget accounting for the RM-P1-LEARN-06 curves;
    :meth:`provenance` yields the JSON-serializable declared comms assumption for the
    ``PolicyPackage`` metadata sidecar (RM-P1-LEARN-05)."""

    def __init__(self, config: CommsModelConfig | None = None) -> None:
        self.config = config if config is not None else CommsModelConfig()
        self._rng: np.random.Generator = np.random.default_rng()
        self._buffers: dict[tuple[AgentId, str], list[_Buffered]] = {}
        self._ledger = CommsLedger()

    # --- lifecycle -----------------------------------------------------------------

    def reset(self, seed: int | None = None) -> None:
        """Re-seed the comms RNG deterministically and clear all channel state.

        Mixing the run ``seed`` with the config's fixed ``seed_salt`` gives a stream
        independent of the environment's own RNG, so the same config + seed reproduce the
        identical drop/delay realization (the determinism gate)."""
        base = 0 if seed is None else int(seed)
        self._rng = np.random.default_rng(np.random.SeedSequence([base, self.config.seed_salt]))
        self._buffers = {}
        self._ledger.reset()

    @property
    def ledger(self) -> CommsLedger:
        """The per-agent comms-budget accounting accumulated since :meth:`reset`."""
        return self._ledger

    def provenance(self) -> dict[str, object]:
        """The declared comms/observability assumption for ``PolicyPackage`` metadata —
        honest provenance for Guard (learn.md §3; RM-P1-LEARN-05)."""
        return {
            "kind": "comms_model",
            "schema_version": self.config.schema_version,
            "config": self.config.model_dump(mode="json"),
        }

    # --- per-tick channel ----------------------------------------------------------

    def apply(self, observations: Mapping[AgentId, Observation]) -> dict[AgentId, Observation]:
        """Return the observation map degraded by the comms regime.

        The returned observations carry a rebuilt ``comms`` mask (peers whose message
        *arrives this tick*) and ``neighbors`` list (each peer's last-known delivered
        state). An identity config returns the observations unchanged."""
        if self.config.is_identity:
            return dict(observations)
        return {agent: self._apply_agent(agent, obs) for agent, obs in sorted(observations.items())}

    def _apply_agent(self, agent: AgentId, obs: Observation) -> Observation:
        tick = obs.tick
        tally = self._ledger.tally(agent)
        neighbors_by_id = {n.agent_id: n for n in obs.neighbors}
        candidates = self._candidates(obs, neighbors_by_id)

        # Stage 1 — gate (LOS/range/margin/latency). Gated-out links deliver nothing new.
        passed: list[PeerLink] = []
        for link in candidates:
            tally.offered += 1
            tally.bits_offered += self.config.bandwidth.message_bits
            if self._gate(link, obs.self_state, neighbors_by_id.get(link.peer)):
                passed.append(link)
            else:
                tally.gated_out += 1

        # Stage 2 — budget: admit by priority until the per-agent bit budget is exhausted.
        admitted = self._admit(passed, tally)

        # Stage 3 & 4 — drop then delay, in a fixed peer-sorted order so the RNG stream is
        # independent of priority ordering and policy behaviour.
        for link in sorted(admitted, key=lambda link_: link_.peer):
            if self.config.drop.probability > 0.0 and self._rng.random() < (
                self.config.drop.probability
            ):
                tally.loss_dropped += 1
                continue
            tally.delivered += 1
            tally.bits_delivered += self.config.bandwidth.message_bits
            delay = self._sample_delay()
            if delay > 0:
                tally.delayed += 1
            self._enqueue(agent, link, neighbors_by_id.get(link.peer), tick + delay)

        return self._render(obs, candidates, tick)

    # --- stages --------------------------------------------------------------------

    def _candidates(
        self, obs: Observation, neighbors_by_id: Mapping[str, StateSample]
    ) -> list[PeerLink]:
        """The peer links entering the channel this tick.

        Link-driven when a Core comms mask is present (honoring its ``reachable`` LOS
        verdict); otherwise a synthetic all-reachable link per observed neighbour, gated
        by range downstream — the same API either way."""
        if obs.comms is not None:
            links = obs.comms.links
            if self.config.range_gate.honor_reachable:
                return [link for link in links if link.reachable]
            return list(links)
        return [PeerLink(peer=peer, reachable=True) for peer in sorted(neighbors_by_id)]

    def _gate(self, link: PeerLink, self_state: StateSample, neighbor: StateSample | None) -> bool:
        gate = self.config.range_gate
        if gate.min_margin_db is not None and (
            link.margin_db is None or link.margin_db < gate.min_margin_db
        ):
            return False
        if gate.max_latency_s is not None and (
            link.latency_s is not None and link.latency_s > gate.max_latency_s
        ):
            return False
        if gate.max_range_m is not None:
            if neighbor is None:
                return False
            if _range_m(self_state, neighbor) > gate.max_range_m:
                return False
        return True

    def _admit(self, passed: list[PeerLink], tally: LinkTally) -> list[PeerLink]:
        budget = self.config.bandwidth.per_agent_bits_per_tick
        if budget is None:
            return passed
        cost = self.config.bandwidth.message_bits
        capacity = int(budget // cost) if cost > 0 else len(passed)
        ordered = sorted(passed, key=self._priority_key)
        admitted = ordered[:capacity]
        tally.budget_dropped += len(ordered) - len(admitted)
        return admitted

    def _priority_key(self, link: PeerLink) -> tuple[float, str]:
        """Lower sorts first (admitted first). Ties break on peer id for determinism."""
        policy = self.config.bandwidth.priority
        if policy == "peer_order":
            return (0.0, link.peer)
        if policy == "margin_db":
            return (-(link.margin_db if link.margin_db is not None else -_INF), link.peer)
        if policy == "rate_bps":
            return (-(link.rate_bps if link.rate_bps is not None else -_INF), link.peer)
        # latency_s: lower latency is higher priority; unknown latency sorts last.
        return (link.latency_s if link.latency_s is not None else _INF, link.peer)

    def _sample_delay(self) -> int:
        delay = self.config.delay
        if delay.kind == "none":
            return 0
        if delay.kind == "fixed":
            return delay.ticks
        if delay.kind == "uniform":
            return int(self._rng.integers(delay.low, delay.high + 1))
        # geometric: mean extra ticks = mean_ticks; numpy geometric is >= 1, so subtract 1.
        p = 1.0 / (1.0 + delay.mean_ticks) if delay.mean_ticks > 0 else 1.0
        return min(int(self._rng.geometric(p)) - 1, delay.max_ticks)

    def _enqueue(
        self, agent: AgentId, link: PeerLink, state: StateSample | None, release_tick: int
    ) -> None:
        buf = self._buffers.setdefault((agent, link.peer), [])
        buf.append(_Buffered(release_tick, state, link))
        # Bound memory: keep only entries that can still be the latest-released now or later.
        if len(buf) > self.config.delay.max_ticks + 2:
            buf.sort(key=lambda b: b.release_tick)
            del buf[: len(buf) - (self.config.delay.max_ticks + 2)]

    def _render(self, obs: Observation, candidates: list[PeerLink], tick: int) -> Observation:
        """Rebuild the observation's ``comms`` mask and ``neighbors`` from the buffers.

        A peer is ``reachable`` iff a message is *released this tick*; its neighbour entry
        is the most recently released (last-known) state."""
        agent = obs.agent_id
        links_out: list[PeerLink] = []
        neighbors_out: list[StateSample] = []
        for cand in candidates:
            buf = self._buffers.get((agent, cand.peer))
            latest = _latest_released(buf, tick) if buf else None
            arrives_now = bool(buf and any(b.release_tick == tick for b in buf))
            base = latest.link if latest is not None else cand
            links_out.append(base.model_copy(update={"reachable": arrives_now}))
            if latest is not None and latest.state is not None:
                neighbors_out.append(latest.state)

        comms_out = None
        if obs.comms is not None:
            comms_out = obs.comms.model_copy(update={"links": links_out})
        elif links_out:
            comms_out = CommsObservationMask(agent_id=agent, links=links_out, earth_contact=False)
        return obs.model_copy(update={"comms": comms_out, "neighbors": neighbors_out})


def _range_m(a: StateSample, b: StateSample) -> float:
    pa, pb = a.pose.translation_m, b.pose.translation_m
    return float(np.hypot(np.hypot(pa.x - pb.x, pa.y - pb.y), pa.z - pb.z))


def _latest_released(buf: list[_Buffered], tick: int) -> _Buffered | None:
    released = [b for b in buf if b.release_tick <= tick]
    return max(released, key=lambda b: b.release_tick) if released else None
