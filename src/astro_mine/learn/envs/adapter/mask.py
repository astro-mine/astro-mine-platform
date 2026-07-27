"""First-class observation-mask and comms-channel surfaces (RM-P1-LEARN-01).

Partial observability and intermittent comms are what make swarm coordination hard
(learn.md §3; LUNAR-TR-003), so the adapter surfaces them as first-class structure —
not buried in the observation tensor. Both are read straight off the Core
:class:`~astro_mine.core.messages.model.Observation`:

- :func:`observation_mask` — per agent, whether it is observable this tick
  (``Observation.observable``); the adapter echoes it into ``infos[agent]``.
- :func:`comms_channel` — per agent, the *reachable* peer links this tick
  (``Observation.comms.links`` filtered to ``reachable``); echoed into ``infos[agent]``.

Comms drop/delay/budget **dynamics** are out of scope here (RM-P1-LEARN-02); this only
exposes the per-tick connectivity *structure* Link already computed.
"""

from __future__ import annotations

from collections.abc import Mapping

from astro_mine.core.env.model import AgentId, ResetResult, StepResult
from astro_mine.core.messages.model import Observation, PeerLink

__all__ = ["comms_channel", "observation_mask"]

_Result = ResetResult | StepResult


def _observations(result: _Result | Mapping[AgentId, Observation]) -> Mapping[AgentId, Observation]:
    if isinstance(result, ResetResult | StepResult):
        return result.observations
    return result


def observation_mask(
    result: _Result | Mapping[AgentId, Observation],
) -> dict[AgentId, bool]:
    """Per-agent observability this tick, from each observation's ``observable`` flag.

    Only agents that produced an observation appear; a departed agent is simply absent
    (it yields no observation)."""
    return {agent: bool(obs.observable) for agent, obs in _observations(result).items()}


def comms_channel(
    result: _Result | Mapping[AgentId, Observation],
) -> dict[AgentId, list[PeerLink]]:
    """Per-agent reachable peer links this tick (the comms channel structure).

    Filters each agent's :class:`~astro_mine.core.messages.model.CommsObservationMask`
    to the links marked ``reachable``; an agent with no comms mask gets an empty list."""
    channel: dict[AgentId, list[PeerLink]] = {}
    for agent, obs in _observations(result).items():
        if obs.comms is None:
            channel[agent] = []
        else:
            channel[agent] = [link for link in obs.comms.links if link.reachable]
    return channel
