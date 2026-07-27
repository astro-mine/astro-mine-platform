"""``SwarmEnv`` — a Core Environment world as a PettingZoo ``ParallelEnv`` (RM-P1-LEARN-01).

Wraps any object satisfying the Core :class:`~astro_mine.core.env.protocol.Environment`
protocol as a multi-agent PettingZoo parallel environment, reaching the world *only*
through ``astro_mine.core`` (no Sim/Worlds/Fleet import). Because the Core protocol
exposes neither spaces nor SADF assets, the adapter also takes the per-agent
:class:`~astro_mine.core.sadf.model.Asset` documents (Core types) it derives the
capability-keyed spaces from — keeping the narrow waist intact (learn.md §3).

Structural reshaping of the Core reset/step returns onto the PettingZoo tuples reuses
Core's own :mod:`astro_mine.core.env.adapter`; on top of it ``SwarmEnv`` adds the
capability-keyed space encoding, the first-class observation-mask / comms-channel infos,
and PettingZoo's live-agent bookkeeping (an agent that terminates *or* truncates leaves
``agents`` and is never revived). Single-agent and centralized Gymnasium views are
available via :meth:`SwarmEnv.single_agent` / :meth:`SwarmEnv.centralized`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from pettingzoo.utils.env import ParallelEnv

from astro_mine.core.env.adapter import as_pettingzoo_step
from astro_mine.core.env.model import AgentId, Info
from astro_mine.core.env.protocol import Environment
from astro_mine.core.messages.model import Observation
from astro_mine.core.sadf.model import Asset
from astro_mine.learn.envs.adapter.encode import (
    ObsSample,
    encode_action_batch,
    encode_observation,
    zero_observation,
)
from astro_mine.learn.envs.adapter.mask import comms_channel, observation_mask
from astro_mine.learn.envs.adapter.spaces import AgentSpaceSpec, build_agent_spec
from astro_mine.learn.envs.comms import CommsModel

if TYPE_CHECKING:  # static proof SwarmEnv honors the Core Environment contract it wraps
    from gymnasium import spaces

    from astro_mine.learn.envs.adapter.gym_env import CentralizedEnv, SingleAgentEnv

__all__ = ["SwarmEnv", "make_swarm_env"]


class SwarmEnv(ParallelEnv):  # type: ignore[misc]  # pettingzoo ships no py.typed → base is Any
    """A Core Environment world presented as a PettingZoo ``ParallelEnv``.

    Construct with the world (any Core :class:`~astro_mine.core.env.protocol.Environment`)
    and the per-agent SADF :class:`~astro_mine.core.sadf.model.Asset` documents the
    capability-keyed spaces are derived from. An optional
    :class:`~astro_mine.learn.envs.comms.CommsModel` (RM-P1-LEARN-02) degrades the comms
    regime — LOS/range gating, bandwidth budget, stochastic drop, and fixed/sampled delay —
    *inside* the environment, so the constraint is identical across every algorithm; its
    per-agent comms-budget accounting is exposed via :meth:`comms_report`."""

    metadata: ClassVar[dict[str, Any]] = {
        "name": "swarm_env_v0",
        "is_parallelizable": True,
        "render_modes": [],
    }

    def __init__(
        self,
        env: Environment,
        assets: Mapping[AgentId, Asset],
        *,
        comms_model: CommsModel | None = None,
    ) -> None:
        missing = set(env.possible_agents) - set(assets)
        if missing:
            raise ValueError(f"missing SADF asset(s) for agent(s): {sorted(missing)}")
        self._env = env
        self._assets = dict(assets)
        self._comms_model = comms_model
        self.possible_agents: list[AgentId] = list(env.possible_agents)
        self.agents: list[AgentId] = []
        possible = tuple(env.possible_agents)
        self._specs: dict[AgentId, AgentSpaceSpec] = {
            agent: build_agent_spec(agent, self._assets[agent], possible) for agent in possible
        }
        self.observation_spaces: dict[AgentId, spaces.Space[Any]] = {
            agent: spec.observation_space for agent, spec in self._specs.items()
        }
        self.action_spaces: dict[AgentId, spaces.Space[Any]] = {
            agent: spec.action_space for agent, spec in self._specs.items()
        }
        self._last_obs: dict[AgentId, ObsSample] = {}

    # --- PettingZoo ParallelEnv API ------------------------------------------------

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[AgentId, ObsSample], dict[AgentId, Info]]:
        if self._comms_model is not None:
            self._comms_model.reset(seed)
        result = self._env.reset(seed=seed, options=options)
        self.agents = list(self._env.agents)
        obs, infos = self._render(result.observations, result.infos, self.agents)
        self._last_obs = obs
        return obs, infos

    def step(
        self, actions: Mapping[AgentId, Mapping[str, Any]]
    ) -> tuple[
        dict[AgentId, ObsSample],
        dict[AgentId, float],
        dict[AgentId, bool],
        dict[AgentId, bool],
        dict[AgentId, Info],
    ]:
        pre_agents = list(self.agents)
        batch = encode_action_batch(
            {agent: sample for agent, sample in actions.items() if agent in self._specs},
            self._specs,
        )
        result = self._env.step(batch)
        core_obs, rewards, terms, truncs, core_infos = as_pettingzoo_step(result)

        obs, infos = self._render(core_obs, core_infos, pre_agents)
        rewards_out = {agent: float(rewards.get(agent, 0.0)) for agent in pre_agents}
        terms_out = {agent: bool(terms.get(agent, False)) for agent in pre_agents}
        truncs_out = {agent: bool(truncs.get(agent, False)) for agent in pre_agents}

        # PettingZoo attrition: an agent that terminated OR truncated leaves the live set
        # (Core removes only terminated agents from ``env.agents``; a horizon truncation is
        # a departure here too). No agent is ever revived.
        env_active = set(self._env.agents)
        self.agents = [
            agent
            for agent in pre_agents
            if agent in env_active and not (terms_out[agent] or truncs_out[agent])
        ]
        self._last_obs = obs
        return obs, rewards_out, terms_out, truncs_out, infos

    def observation_space(self, agent: AgentId) -> spaces.Space[Any]:
        return self._specs[agent].observation_space

    def action_space(self, agent: AgentId) -> spaces.Space[Any]:
        return self._specs[agent].action_space

    def state(self) -> NDArray[np.float32]:
        """A global view for CTDE: every possible agent's ``self_state`` block, in order
        (zeros for an agent with no current observation)."""
        blocks: list[NDArray[np.float32]] = []
        for agent in self.possible_agents:
            block = self._last_obs.get(agent, {}).get("self_state")
            if block is None:
                shape = zero_observation(self._specs[agent])["self_state"].shape
                block = np.zeros(shape, dtype=np.float32)
            blocks.append(block)
        return np.concatenate(blocks) if blocks else np.zeros((0,), dtype=np.float32)

    def render(self) -> None:
        return None

    def close(self) -> None:
        self._last_obs = {}

    # --- single-agent / centralized Gymnasium views --------------------------------

    def single_agent(self, agent_id: AgentId) -> SingleAgentEnv:
        """A Gymnasium :class:`~gymnasium.Env` view controlling one agent."""
        from astro_mine.learn.envs.adapter.gym_env import SingleAgentEnv

        return SingleAgentEnv(self._env, self._specs[agent_id])

    def centralized(self) -> CentralizedEnv:
        """A Gymnasium :class:`~gymnasium.Env` super-agent over the whole swarm."""
        from astro_mine.learn.envs.adapter.gym_env import CentralizedEnv

        return CentralizedEnv(self._env, self._specs, tuple(self.possible_agents))

    @property
    def agent_specs(self) -> dict[AgentId, AgentSpaceSpec]:
        """The per-agent capability-keyed :class:`AgentSpaceSpec`\\ s (RM-P1-LEARN-03).

        The full space spec — capability-keyed observation/action spaces plus the
        encode/decode metadata (modalities, sensor names, tool/method) — that the baselines
        derive their per-agent policy nets from and that a produced
        :class:`~astro_mine.core.policy.Policy` uses to encode Core observations and decode
        Core actions. A read-only copy; mutating it does not affect the env."""
        return dict(self._specs)

    # --- internals -----------------------------------------------------------------

    def comms_report(self) -> dict[AgentId, dict[str, float]]:
        """Per-agent comms-budget accounting since the last reset (offered / delivered /
        gated / dropped / delayed), for the RM-P1-LEARN-06 comms-stress curves. Empty when
        no :class:`~astro_mine.learn.envs.comms.CommsModel` is attached."""
        return {} if self._comms_model is None else self._comms_model.ledger.snapshot()

    def comms_provenance(self) -> dict[str, Any] | None:
        """The declared comms/observability assumption recorded in ``PolicyPackage``
        metadata (honest provenance for Guard, RM-P1-LEARN-05); ``None`` if unconstrained."""
        return None if self._comms_model is None else self._comms_model.provenance()

    def _render(
        self,
        core_obs: Mapping[AgentId, Observation],
        core_infos: Mapping[AgentId, Info],
        agents: list[AgentId],
    ) -> tuple[dict[AgentId, ObsSample], dict[AgentId, Info]]:
        if self._comms_model is not None:
            core_obs = self._comms_model.apply(core_obs)
        masks = observation_mask(core_obs)
        channel = comms_channel(core_obs)
        obs: dict[AgentId, ObsSample] = {}
        infos: dict[AgentId, Info] = {}
        for agent in agents:
            spec = self._specs[agent]
            o = core_obs.get(agent)
            # A masked (observable=False) or absent agent yields only the neutral, in-space
            # observation — masking must leak no sensor content — but stays a dict key so
            # the PettingZoo live-agent contract holds.
            if o is not None and masks.get(agent, False):
                obs[agent] = encode_observation(o, spec)
            else:
                obs[agent] = zero_observation(spec)
            info: dict[str, Any] = dict(core_infos.get(agent, {}))
            info["observation_mask"] = masks.get(agent, False)
            info["comms"] = channel.get(agent, [])
            info["earth_contact"] = bool(
                o is not None and o.comms is not None and o.comms.earth_contact
            )
            infos[agent] = info
        return obs, infos


def make_swarm_env(
    env: Environment,
    assets: Mapping[AgentId, Asset],
    *,
    comms_model: CommsModel | None = None,
) -> SwarmEnv:
    """Build a :class:`SwarmEnv` over a Core world (the vectorization seam RM-P1-LEARN-04
    consumes)."""
    return SwarmEnv(env, assets, comms_model=comms_model)
