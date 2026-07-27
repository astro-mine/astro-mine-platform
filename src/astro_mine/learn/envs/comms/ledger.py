"""Comms-budget accounting (RM-P1-LEARN-02 → RM-P1-LEARN-06).

The ``CommsModel`` degrades the channel; the ledger *counts* the degradation so the
comms-stress evaluation curves (performance vs. drop/delay/budget, RM-P1-LEARN-06) have a
denominator. Every tick, for every agent, each candidate peer link falls into exactly one
outcome, so ``delivered + gated_out + budget_dropped + loss_dropped == offered`` — an
invariant the Hypothesis tests assert.

The ledger is plain, JSON-serializable, and reset per episode. It is *accounting*, not
control: the model's decisions do not read back from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["CommsLedger", "LinkTally"]


@dataclass
class LinkTally:
    """One agent's cumulative link outcomes over an episode, plus this-tick deltas."""

    offered: int = 0
    delivered: int = 0
    gated_out: int = 0
    budget_dropped: int = 0
    loss_dropped: int = 0
    delayed: int = 0
    bits_offered: float = 0.0
    bits_delivered: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "offered": self.offered,
            "delivered": self.delivered,
            "gated_out": self.gated_out,
            "budget_dropped": self.budget_dropped,
            "loss_dropped": self.loss_dropped,
            "delayed": self.delayed,
            "bits_offered": self.bits_offered,
            "bits_delivered": self.bits_delivered,
        }


@dataclass
class CommsLedger:
    """Per-agent :class:`LinkTally` accumulation across an episode."""

    per_agent: dict[str, LinkTally] = field(default_factory=dict)

    def reset(self) -> None:
        self.per_agent.clear()

    def tally(self, agent: str) -> LinkTally:
        return self.per_agent.setdefault(agent, LinkTally())

    def snapshot(self) -> dict[str, dict[str, float]]:
        """A JSON-serializable copy of the cumulative accounting (for run provenance /
        the RM-P1-LEARN-06 curves)."""
        return {agent: tally.as_dict() for agent, tally in self.per_agent.items()}

    def delivery_ratio(self) -> float:
        """Fraction of offered messages delivered across the whole swarm (1.0 when nothing
        was offered) — the headline comms-stress scalar."""
        offered = sum(t.offered for t in self.per_agent.values())
        delivered = sum(t.delivered for t in self.per_agent.values())
        return 1.0 if offered == 0 else delivered / offered
