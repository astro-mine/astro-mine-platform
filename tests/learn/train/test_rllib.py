"""Ray RLlib scale-out path (RM-P1-LEARN-03; learn.md §11) — needs the [rllib] extra.

Verifies the LEARN-04 executor seam: a SwarmEnv wraps as an RLlib ``ParallelPettingZooEnv``
and steps, registers with the tune registry, and yields a per-agent-policy ``PPOConfig``
(the config a KubeRay ``RayJob`` would run). It intentionally does **not** run RLlib's own
training loop — the reproducible baseline is the in-house trainer (learn.md §11: RLlib's
new-API-stack does not first-class the heterogeneous Dict action space); this proves the
integration + config surface only, which is fast and deterministic in CI.
"""

from __future__ import annotations

import pytest

pytest.importorskip("ray.rllib")

pytestmark = pytest.mark.ray

from astro_mine.learn import make_swarm_env  # noqa: E402
from astro_mine.learn.train.rllib import (  # noqa: E402
    build_ppo_config,
    register_swarm_env,
    wrap_parallel_pettingzoo,
)
from tests.learn.fakes import FakeSwarmWorld, build_assets  # noqa: E402


def _factory():
    return make_swarm_env(FakeSwarmWorld(), build_assets())


def test_parallel_pettingzoo_wrapper_resets_and_steps() -> None:
    wrapped = wrap_parallel_pettingzoo(_factory())
    obs, _ = wrapped.reset(seed=0)
    assert set(obs) == {"rover", "digger", "relay"}
    actions = {agent: {"kind": 0, "mode": 0} for agent in obs}
    _next_obs, rewards, _terms, _truncs, _infos = wrapped.step(actions)
    assert set(rewards) <= {"rover", "digger", "relay"}


def test_register_swarm_env_creator_builds_a_wrapped_env() -> None:
    from ray.tune.registry import ENV_CREATOR, _global_registry

    register_swarm_env("swarm_learn03_creator", _factory)
    creator = _global_registry.get(ENV_CREATOR, "swarm_learn03_creator")
    env = creator({})  # invokes the registered creator (the KubeRay per-worker path)
    obs, _ = env.reset(seed=0)
    assert set(obs) == {"rover", "digger", "relay"}


def test_register_swarm_env_and_build_ppo_config() -> None:
    register_swarm_env("swarm_learn03_test", _factory)
    config = build_ppo_config(
        _factory(),
        env_name="swarm_learn03_test",
        seed=0,
        rollout_fragment_length=8,
        train_batch_size=16,
    )
    assert config.env == "swarm_learn03_test"
    assert config.framework_str == "torch"
    assert set(config.policies) == {"rover", "digger", "relay"}
