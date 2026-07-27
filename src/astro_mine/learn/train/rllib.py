"""Ray RLlib scale-out path for the PPO baselines (RM-P1-LEARN-03; learn.md §11, LEARN-04).

learn.md §11 recommends **Ray RLlib (PyTorch)** as the default distributed executor and
**KubeRay** for scale-out. This module is that path: it registers a SwarmEnv as an RLlib
multi-agent env (:class:`ParallelPettingZooEnv`) and builds a per-agent-policy
:class:`~ray.rllib.algorithms.ppo.PPOConfig` — the config a KubeRay ``RayJob`` runs in
RM-P1-LEARN-04, "the same code with a different executor" (learn.md §2.1).

**Why the reproducible baseline is in-house, not RLlib.** RLlib's new API stack does not
first-class heterogeneous, capability-keyed **Dict action** spaces, and byte-identical
learning-curve determinism across a Ray run is not guaranteed — both are hard requirements
of the CX-REPRO determinism gate. So the reproducible, CPU-deterministic trainers live in
:mod:`astro_mine.learn.algos` (stepping the SwarmEnv through the
:class:`~astro_mine.learn.train.executor.LocalExecutor`), and *this* module is the scale-out
config seam. Importing it requires the ``[rllib]`` extra (Ray); it is exercised by the
``ray``-marked tests only.
"""

from __future__ import annotations

from collections.abc import Callable

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

from astro_mine.learn.envs import SwarmEnv

__all__ = [
    "build_ppo_config",
    "register_swarm_env",
    "wrap_parallel_pettingzoo",
]


def wrap_parallel_pettingzoo(env: SwarmEnv) -> ParallelPettingZooEnv:
    """Wrap a :class:`~astro_mine.learn.envs.SwarmEnv` as an RLlib multi-agent env."""
    return ParallelPettingZooEnv(env)


def register_swarm_env(name: str, env_factory: Callable[[], SwarmEnv]) -> None:
    """Register a SwarmEnv factory with RLlib's tune registry under ``name``.

    The registered creator wraps each fresh ``env_factory()`` in
    :class:`ParallelPettingZooEnv`, so a :class:`PPOConfig` referencing ``name`` spawns
    per-worker rollout envs — the seam RM-P1-LEARN-04's KubeRay executor scales out."""

    def creator(_config: object) -> ParallelPettingZooEnv:
        return ParallelPettingZooEnv(env_factory())

    register_env(name, creator)


def build_ppo_config(
    env: SwarmEnv,
    *,
    env_name: str,
    seed: int = 0,
    num_env_runners: int = 0,
    rollout_fragment_length: int = 32,
    train_batch_size: int = 64,
) -> PPOConfig:
    """Build an independent-per-agent multi-agent :class:`PPOConfig` for a SwarmEnv.

    One RLlib policy per (heterogeneous) agent, mapped identity — the IPPO-style scale-out
    config. MAPPO's centralized critic is realized by the in-house trainer (learn.md §11);
    wiring a centralized-critic connector on the RLlib path is an RM-P1-LEARN-04 follow-up.
    ``num_env_runners=0`` keeps sampling in the local process (tier 1, deterministic-friendly)."""
    policies = {
        agent: PolicySpec(
            observation_space=env.observation_space(agent),
            action_space=env.action_space(agent),
        )
        for agent in env.possible_agents
    }
    return (
        PPOConfig()
        .environment(env_name)
        .framework("torch")
        .env_runners(
            num_env_runners=num_env_runners,
            rollout_fragment_length=rollout_fragment_length,
        )
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
        )
        .training(train_batch_size=train_batch_size)
        .debugging(seed=seed)
    )
