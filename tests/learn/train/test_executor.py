"""The rollout executor seam (RM-P1-LEARN-03; LEARN-04 seam) — Torch-free.

Exercises :class:`LocalExecutor` with a trivial numpy decision function (no learner needed)
so the seam is covered without the ``[rllib]`` extra, plus the reward-shaping hook. The
distributed :class:`KubeRayExecutor` behaviour (single-worker byte-identity, multi-worker
reproducibility) is exercised under Ray in ``tests/train/test_kuberay.py``; here we only
assert its Torch-free guard that a plain callable is not a shippable
:class:`~astro_mine.learn.train.executor.BroadcastableStep`.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pytest

from astro_mine.core.env.model import AgentId
from astro_mine.learn import make_swarm_env
from astro_mine.learn.envs.adapter.encode import ObsSample
from astro_mine.learn.train import KubeRayExecutor, LocalExecutor, StepDecision
from tests.learn.fakes import FakeSwarmWorld, build_assets


def _mode_zero_step(
    flat_obs: Mapping[AgentId, np.ndarray],
) -> Mapping[AgentId, StepDecision]:
    # kind=0 is always the "mode" modality; a valid, in-space action for every agent.
    return {a: StepDecision(action_sample={"kind": 0, "mode": 0}) for a in flat_obs}


def test_local_executor_collects_a_trajectory() -> None:
    env = make_swarm_env(FakeSwarmWorld(), build_assets())
    rollout = LocalExecutor().rollout(env, _mode_zero_step, steps=16, seed=0)
    assert rollout.steps  # the fake yields steps before truncating at the horizon
    assert rollout.env_steps() > 0
    assert set(rollout.possible_agents) == {"rover", "digger", "relay"}
    assert rollout.state_dim > 0
    # digger terminates early; its sub-trajectory is a strict prefix of rover's.
    assert len(rollout.agent_trajectory("digger")) <= len(rollout.agent_trajectory("rover"))


def test_reward_shaping_hook_overrides_the_reward_free_env() -> None:
    env = make_swarm_env(FakeSwarmWorld(), build_assets())

    def reward_fn(_agent: AgentId, _obs: ObsSample) -> float:
        return 2.5

    rollout = LocalExecutor().rollout(env, _mode_zero_step, steps=4, seed=0, reward_fn=reward_fn)
    assert all(r == 2.5 for step in rollout.steps for r in step.reward.values())


def test_rollout_is_seed_reproducible() -> None:
    def collect(seed: int) -> list[float]:
        env = make_swarm_env(FakeSwarmWorld(), build_assets())
        rollout = LocalExecutor().rollout(env, _mode_zero_step, steps=8, seed=seed)
        return [float(v) for step in rollout.steps for v in step.state.tolist()]

    assert collect(0) == collect(0)


def test_kuberay_rejects_a_non_broadcastable_step() -> None:
    # KubeRay ships weights + a rebuild fn to remote workers, not the (unpicklable) closure —
    # so a bare callable is rejected loudly before any Ray import; the trainer supplies a
    # BroadcastableStep (see tests/train/test_kuberay.py). No cluster / [rllib] extra needed.
    env = make_swarm_env(FakeSwarmWorld(), build_assets())
    executor = KubeRayExecutor(lambda: make_swarm_env(FakeSwarmWorld(), build_assets()))
    with pytest.raises(TypeError, match=r"BroadcastableStep|broadcast"):
        executor.rollout(env, _mode_zero_step, steps=1, seed=0)
