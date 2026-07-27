"""Environment API ⇄ Gymnasium / PettingZoo mapping (RM-P0-CORE-02).

Pure structural adapters that view a Core :class:`ResetResult`/:class:`StepResult` as
the Gymnasium (single-agent) and PettingZoo ``ParallelEnv`` (multi-agent) return tuples
(core.md §3; conventions.md §3). They import **neither** library — the Core contract
*maps* onto those APIs; the concrete ``gymnasium.Env`` / ``pettingzoo.ParallelEnv``
wrapper classes are Learn's (learn.md §3, Phase 1).

- PettingZoo parallel: ``reset -> (observations, infos)``;
  ``step -> (observations, rewards, terminations, truncations, infos)`` — per-agent maps.
- Gymnasium single-agent: ``reset -> (observation, info)``;
  ``step -> (observation, reward, terminated, truncated, info)`` for one selected agent.
"""

from __future__ import annotations

from collections.abc import Mapping

from astro_mine.core.env.model import AgentId, Info, ResetResult, StepResult
from astro_mine.core.messages.model import Observation

__all__ = [
    "as_gymnasium_reset",
    "as_gymnasium_step",
    "as_pettingzoo_reset",
    "as_pettingzoo_step",
]


def as_pettingzoo_reset(
    result: ResetResult,
) -> tuple[Mapping[AgentId, Observation], Mapping[AgentId, Info]]:
    """View a reset as the PettingZoo parallel ``(observations, infos)``."""
    return result.observations, result.infos


def as_pettingzoo_step(
    result: StepResult,
) -> tuple[
    Mapping[AgentId, Observation],
    Mapping[AgentId, float],
    Mapping[AgentId, bool],
    Mapping[AgentId, bool],
    Mapping[AgentId, Info],
]:
    """View a step as the PettingZoo parallel five-tuple.

    Rewards/terminations/truncations are made total over the observed agents (filling
    ``0.0`` / ``False`` for any the env omitted), since PettingZoo expects an entry per
    agent.
    """
    agents = result.observations.keys()
    rewards = {a: result.rewards.get(a, 0.0) for a in agents}
    terminations = {a: result.terminations.get(a, False) for a in agents}
    truncations = {a: result.truncations.get(a, False) for a in agents}
    return result.observations, rewards, terminations, truncations, result.infos


def as_gymnasium_reset(result: ResetResult, agent_id: AgentId) -> tuple[Observation, Info]:
    """View a single agent's reset as the Gymnasium ``(observation, info)``."""
    return _require_agent(result.observations, agent_id), result.infos.get(agent_id, {})


def as_gymnasium_step(
    result: StepResult, agent_id: AgentId
) -> tuple[Observation, float, bool, bool, Info]:
    """View a single agent's step as the Gymnasium
    ``(observation, reward, terminated, truncated, info)``."""
    obs = _require_agent(result.observations, agent_id)
    return (
        obs,
        result.rewards.get(agent_id, 0.0),
        result.terminations.get(agent_id, False),
        result.truncations.get(agent_id, False),
        result.infos.get(agent_id, {}),
    )


def _require_agent(observations: Mapping[AgentId, Observation], agent_id: AgentId) -> Observation:
    try:
        return observations[agent_id]
    except KeyError:
        raise KeyError(
            f"agent {agent_id!r} has no observation this step "
            f"(present: {sorted(observations)}); it may be unobservable this tick"
        ) from None
