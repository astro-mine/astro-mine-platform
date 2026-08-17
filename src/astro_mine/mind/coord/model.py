# SPDX-License-Identifier: Apache-2.0
"""Decentralized neighbor coordination (RM-P1-MIND-06).

The ``coord/`` substrate for acting coherently when the global (mission-tier) view is stale:
neighbors gossip their task claims and resolve conflicts by a deterministic local rule, with no
central coordinator and no synchronous exchange (mind.md §11, decentralized/hybrid). When two
agents claim the same task, a stable tiebreak (lowest agent id) keeps the claim and the others
yield — so a comms-denied swarm does not double-book a region while it acts on cached intent.
Pure and deterministic: the same intents always resolve the same way, so a seeded run reproduces.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from astro_mine.core.env.model import AgentId

__all__ = ["GossipCoordinator", "Intent", "Resolution"]


@dataclass(frozen=True, slots=True)
class Intent:
    """One agent's gossiped claim: the ``task_id`` it intends to work (e.g. a prospect region)."""

    agent_id: AgentId
    task_id: str


@dataclass(frozen=True, slots=True)
class Resolution:
    """The coordinated outcome: each agent's kept ``task_id`` (``None`` if it yielded), and how
    many conflicts the local rule resolved this round."""

    assignments: dict[AgentId, str | None]
    conflicts_resolved: int = 0
    yielded: tuple[AgentId, ...] = field(default_factory=tuple)


class GossipCoordinator:
    """Resolves conflicting task claims among neighbors by a deterministic local tiebreak."""

    def resolve(self, intents: Iterable[Intent]) -> Resolution:
        """Keep one claimant per task (lowest agent id wins); the rest yield to ``None``."""
        claimants: dict[str, list[AgentId]] = {}
        for intent in intents:
            claimants.setdefault(intent.task_id, []).append(intent.agent_id)
        assignments: dict[AgentId, str | None] = {}
        yielded: list[AgentId] = []
        conflicts = 0
        for task_id, agents in claimants.items():
            winner = min(agents)
            if len(agents) > 1:
                conflicts += 1
            for agent_id in agents:
                if agent_id == winner:
                    assignments[agent_id] = task_id
                else:
                    assignments[agent_id] = None
                    yielded.append(agent_id)
        return Resolution(
            assignments=assignments,
            conflicts_resolved=conflicts,
            yielded=tuple(sorted(yielded)),
        )
