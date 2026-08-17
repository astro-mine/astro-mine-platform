# SPDX-License-Identifier: Apache-2.0
"""The belief view — a tier's partial-observability-aware input (RM-P1-MIND-01).

Assembled by the executive from the per-agent :class:`~astro_mine.core.messages.Observation`
the Environment API yields, and passed to the tiers via
``DecisionContext.extras[BELIEF_EXTRAS_KEY]``. A lightweight, frozen in-memory container
(like the Environment API's result containers) — **not** a wire document; the serializable
loop types stay the Core messages.

v0.1 surfaces exactly what the spine needs: per-agent observations, observability, and the
:class:`~astro_mine.core.messages.CommsObservationMask` (the input the RM-P1-MIND-06
comms-regime switch keys on). Explicit uncertainty / information-value handles (principle 6;
co-designed with Prospect) extend this container additively later.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.model import CommsObservationMask, Observation

__all__ = ["BELIEF_EXTRAS_KEY", "BeliefView", "assemble_belief"]

#: The ``DecisionContext.extras`` key under which the executive publishes the tick's
#: :class:`BeliefView` to every tier. Stable, documented convention (Core keeps ``extras``
#: untyped by design); tier backends read the belief view from here.
BELIEF_EXTRAS_KEY = "astro_mine.mind.belief"


@dataclass(frozen=True, slots=True)
class BeliefView:
    """The fused, partial-observability-aware tier input for one tick.

    ``observations`` is the per-agent map from the Environment API; ``comms`` is the
    per-agent connectivity mask extracted from each observation (``None`` when the agent
    reported no mask this tick). ``agents`` is the observed set in a stable order.
    """

    tick: int
    sim_time_s: float
    agents: tuple[AgentId, ...]
    observations: Mapping[AgentId, Observation]
    comms: Mapping[AgentId, CommsObservationMask | None]

    def observation(self, agent_id: AgentId) -> Observation | None:
        """The agent's observation this tick, or ``None`` if it was not observed."""
        return self.observations.get(agent_id)

    def is_observable(self, agent_id: AgentId) -> bool:
        """Whether the agent is observable this tick (``Observation.observable``)."""
        obs = self.observations.get(agent_id)
        return obs is not None and obs.observable

    def earth_contact(self, agent_id: AgentId) -> bool:
        """Whether the agent can reach an Earth/DSN gateway this tick (comms mask)."""
        mask = self.comms.get(agent_id)
        return mask is not None and mask.earth_contact


def assemble_belief(
    observations: Mapping[AgentId, Observation], *, tick: int, sim_time_s: float
) -> BeliefView:
    """Build the tick's :class:`BeliefView` from the Environment API observations."""
    agents = tuple(observations.keys())
    comms = {agent_id: obs.comms for agent_id, obs in observations.items()}
    return BeliefView(
        tick=tick,
        sim_time_s=sim_time_s,
        agents=agents,
        observations=dict(observations),
        comms=comms,
    )
