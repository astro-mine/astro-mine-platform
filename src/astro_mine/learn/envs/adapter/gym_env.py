# SPDX-License-Identifier: Apache-2.0
"""Gymnasium views over a Core world: single-agent and centralized (RM-P1-LEARN-01).

A multi-agent :class:`~astro_mine.learn.envs.adapter.parallel_env.SwarmEnv` also exposes
the two classic single-agent Gymnasium reductions (learn.md §3):

- :class:`SingleAgentEnv` controls exactly one agent (the others hold), reshaping the
  Core reset/step via :func:`~astro_mine.core.env.adapter.as_gymnasium_reset` /
  :func:`~astro_mine.core.env.adapter.as_gymnasium_step`. This is the view Gymnasium's
  ``check_env`` targets.
- :class:`CentralizedEnv` is one super-agent whose Dict observation/action space is
  keyed by agent id (concatenating every agent's per-agent Dict) — the joint-space
  reduction a centralized controller trains against.

Both are constructed through :meth:`SwarmEnv.single_agent` / :meth:`SwarmEnv.centralized`.
Determinism follows the wrapped Core world: ``reset(seed=...)`` re-seeds it, and ``step``
never advances the Gymnasium RNG.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import gymnasium as gym
from gymnasium import spaces

from astro_mine.core.env.adapter import as_gymnasium_reset, as_gymnasium_step
from astro_mine.core.env.model import AgentId, Info, ResetResult, StepResult
from astro_mine.core.env.protocol import Environment
from astro_mine.core.messages.model import ActionBatch, Observation
from astro_mine.learn.envs.adapter.encode import (
    ObsSample,
    decode_action,
    encode_action_batch,
    encode_observation,
    zero_observation,
)
from astro_mine.learn.envs.adapter.spaces import AgentSpaceSpec

__all__ = ["CentralizedEnv", "SingleAgentEnv"]


def _mask_info(obs: Observation | None, core_info: Info) -> dict[str, Any]:
    info = dict(core_info)
    info["observation_mask"] = bool(obs is not None and obs.observable)
    info["comms"] = (
        [link for link in obs.comms.links if link.reachable]
        if (obs is not None and obs.comms is not None)
        else []
    )
    info["earth_contact"] = bool(
        obs is not None and obs.comms is not None and obs.comms.earth_contact
    )
    return info


def _encode_or_zero(obs: Observation | None, spec: AgentSpaceSpec) -> ObsSample:
    # An absent or masked agent yields only the neutral, in-space observation (no leak).
    if obs is not None and obs.observable:
        return encode_observation(obs, spec)
    return zero_observation(spec)


class SingleAgentEnv(gym.Env[Any, Any]):
    """A Gymnasium view controlling one agent of a Core world."""

    def __init__(self, env: Environment, spec: AgentSpaceSpec) -> None:
        self._env = env
        self._spec = spec
        self._agent_id: AgentId = spec.agent_id
        self.observation_space: spaces.Space[Any] = spec.observation_space
        self.action_space: spaces.Space[Any] = spec.action_space
        self.render_mode: str | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsSample, dict[str, Any]]:
        super().reset(seed=seed)  # seeds gym's own RNG deterministically; step never touches it
        result: ResetResult = self._env.reset(seed=seed, options=options)
        if self._agent_id in result.observations:
            obs, core_info = as_gymnasium_reset(result, self._agent_id)
            return _encode_or_zero(obs, self._spec), _mask_info(obs, core_info)
        return zero_observation(self._spec), _mask_info(None, {})

    def step(
        self, action: Mapping[str, Any]
    ) -> tuple[ObsSample, float, bool, bool, dict[str, Any]]:
        batch = ActionBatch(actions=[decode_action(action, self._spec)])
        result: StepResult = self._env.step(batch)
        if self._agent_id in result.observations:
            obs, reward, term, trunc, core_info = as_gymnasium_step(result, self._agent_id)
            return (
                _encode_or_zero(obs, self._spec),
                float(reward),
                bool(term),
                bool(trunc),
                _mask_info(obs, core_info),
            )
        # The controlled agent departed (terminated) or is unobservable: terminal here.
        return zero_observation(self._spec), 0.0, True, False, _mask_info(None, {})

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None


class CentralizedEnv(gym.Env[Any, Any]):
    """A single super-agent Gymnasium view over the whole swarm (joint Dict spaces)."""

    def __init__(
        self,
        env: Environment,
        specs: Mapping[AgentId, AgentSpaceSpec],
        possible_agents: tuple[AgentId, ...],
    ) -> None:
        self._env = env
        self._specs = dict(specs)
        self._possible_agents = possible_agents
        self.observation_space: spaces.Space[Any] = spaces.Dict(
            {agent: specs[agent].observation_space for agent in possible_agents}
        )
        self.action_space: spaces.Space[Any] = spaces.Dict(
            {agent: specs[agent].action_space for agent in possible_agents}
        )
        self.render_mode: str | None = None

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[AgentId, ObsSample], dict[str, Any]]:
        super().reset(seed=seed)
        result = self._env.reset(seed=seed, options=options)
        obs = {
            agent: _encode_or_zero(result.observations.get(agent), self._specs[agent])
            for agent in self._possible_agents
        }
        return obs, {"agents": list(self._env.agents)}

    def step(
        self, action: Mapping[AgentId, Mapping[str, Any]]
    ) -> tuple[dict[AgentId, ObsSample], float, bool, bool, dict[str, Any]]:
        active = set(self._env.agents)
        batch = encode_action_batch(
            {agent: sample for agent, sample in action.items() if agent in active},
            self._specs,
        )
        result = self._env.step(batch)
        obs = {
            agent: _encode_or_zero(result.observations.get(agent), self._specs[agent])
            for agent in self._possible_agents
        }
        reward = float(sum(result.rewards.values()))
        terminated = not self._env.agents  # every agent has left the active set
        truncated = any(result.truncations.values())
        return obs, reward, terminated, truncated, {"agents": list(self._env.agents)}

    def render(self) -> None:
        return None

    def close(self) -> None:
        return None
