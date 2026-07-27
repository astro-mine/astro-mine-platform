"""The GPU-vectorized rollout executor (RM-P1-LEARN-04) — Torch-free.

:class:`VectorExecutor` is the third rollout-executor choice (after Local and KubeRay). Like
:class:`LocalExecutor` it is Torch-free — it reaches the world only through the Core
Environment contract — so these run in the no-``[rllib]``-extra CI job with a trivial numpy
step: a single vector env reproduces :class:`LocalExecutor` byte-for-byte, and a batch
returns the identical :class:`Rollout` shape with strictly more collected experience.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from astro_mine.core.env.model import AgentId
from astro_mine.learn import make_swarm_env
from astro_mine.learn.envs.vector import VectorExecutor
from astro_mine.learn.train import LocalExecutor, StepDecision
from tests.learn.fakes import FakeSwarmWorld, build_assets


def _mode_zero_step(
    flat_obs: Mapping[AgentId, np.ndarray],
) -> Mapping[AgentId, StepDecision]:
    return {a: StepDecision(action_sample={"kind": 0, "mode": 0}) for a in flat_obs}


def _factory():
    return make_swarm_env(FakeSwarmWorld(), build_assets())


def _signature(rollout) -> list[object]:
    return [
        (sorted(step.obs), step.state.tolist(), [step.reward[a] for a in sorted(step.obs)])
        for step in rollout.steps
    ]


def test_single_env_is_byte_identical_to_local() -> None:
    local = LocalExecutor().rollout(_factory(), _mode_zero_step, steps=8, seed=4)
    vector = VectorExecutor(_factory, num_envs=1).rollout(
        _factory(), _mode_zero_step, steps=8, seed=4
    )
    assert _signature(local) == _signature(vector)


def test_batch_returns_same_rollout_shape_with_more_steps() -> None:
    local = LocalExecutor().rollout(_factory(), _mode_zero_step, steps=8, seed=4)
    vector = VectorExecutor(_factory, num_envs=3).rollout(
        _factory(), _mode_zero_step, steps=8, seed=4
    )
    # Same static shape the trainer builds tensors from ...
    assert vector.possible_agents == local.possible_agents
    assert vector.agent_obs_dim == local.agent_obs_dim
    assert vector.state_dim == local.state_dim
    # ... but three env copies' worth of experience.
    assert vector.env_steps() == 3 * local.env_steps()
    # Env copy 0 uses the run seed verbatim, so its slice is exactly the single-env run.
    assert _signature(vector)[: len(local.steps)] == _signature(local)


def test_fixed_batch_is_reproducible() -> None:
    def run() -> list[object]:
        return _signature(
            VectorExecutor(_factory, num_envs=2).rollout(
                _factory(), _mode_zero_step, steps=6, seed=1
            )
        )

    assert run() == run()


def test_num_envs_must_be_positive() -> None:
    import pytest

    with pytest.raises(ValueError, match="num_envs"):
        VectorExecutor(_factory, num_envs=0)
