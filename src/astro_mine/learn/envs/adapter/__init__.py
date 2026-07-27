"""The SwarmEnv adapter — a Core world as PettingZoo/Gymnasium RL envs (RM-P1-LEARN-01).

Public surface:

- :class:`SwarmEnv` / :func:`make_swarm_env` — the multi-agent PettingZoo ``ParallelEnv``
  wrapper (and the factory the vectorization seam RM-P1-LEARN-04 consumes);
- :class:`SingleAgentEnv` / :class:`CentralizedEnv` — the Gymnasium single-agent and
  centralized views (built via :meth:`SwarmEnv.single_agent` / :meth:`SwarmEnv.centralized`);
- :func:`observation_mask` / :func:`comms_channel` — the first-class partial-observability
  and comms-channel surfaces;
- :class:`AgentSpaceSpec` / :func:`build_agent_spec` — the per-agent, SADF-capability-keyed
  space builder the encoders key off.
"""

from __future__ import annotations

from astro_mine.learn.envs.adapter.gym_env import CentralizedEnv, SingleAgentEnv
from astro_mine.learn.envs.adapter.mask import comms_channel, observation_mask
from astro_mine.learn.envs.adapter.parallel_env import SwarmEnv, make_swarm_env
from astro_mine.learn.envs.adapter.spaces import AgentSpaceSpec, build_agent_spec

__all__ = [
    "AgentSpaceSpec",
    "CentralizedEnv",
    "SingleAgentEnv",
    "SwarmEnv",
    "build_agent_spec",
    "comms_channel",
    "make_swarm_env",
    "observation_mask",
]
