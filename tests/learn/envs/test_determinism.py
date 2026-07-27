"""The wrapped rollout is reproducible under a fixed seed (RM-P1-LEARN-01).

Determinism is contractual for the Core Environment (asserted by
:func:`astro_mine.core.env.check_environment`); the adapter must preserve it — the same
fake, seed, and action sequence produce byte-identical wrapped observations, through both
the multi-agent and single-agent views.
"""

from __future__ import annotations

import numpy as np

from astro_mine.learn.envs import make_swarm_env
from tests.learn.fakes import FakeSwarmWorld, build_assets

_FIXED_ACTION = {"kind": 0, "mode": 0}


def _parallel_rollout(seed: int) -> list[dict[str, dict[str, np.ndarray]]]:
    env = make_swarm_env(FakeSwarmWorld(), build_assets())
    obs, _ = env.reset(seed=seed)
    frames = [obs]
    for _ in range(6):
        if not env.agents:
            break
        actions = {a: dict(_FIXED_ACTION) for a in env.agents}
        obs, _, _, _, _ = env.step(actions)
        frames.append(obs)
    return frames


def test_same_seed_yields_identical_parallel_rollout() -> None:
    first = _parallel_rollout(42)
    second = _parallel_rollout(42)
    assert len(first) == len(second)
    for frame_a, frame_b in zip(first, second, strict=True):
        assert frame_a.keys() == frame_b.keys()
        for agent in frame_a:
            assert frame_a[agent].keys() == frame_b[agent].keys()
            for block in frame_a[agent]:
                assert np.array_equal(frame_a[agent][block], frame_b[agent][block])


def test_single_agent_view_is_deterministic() -> None:
    def rollout(seed: int) -> list[dict[str, np.ndarray]]:
        view = make_swarm_env(FakeSwarmWorld(), build_assets()).single_agent("rover")
        obs, _ = view.reset(seed=seed)
        frames = [obs]
        for _ in range(4):
            obs, _, _, _, _ = view.step(dict(_FIXED_ACTION))
            frames.append(obs)
        return frames

    first, second = rollout(7), rollout(7)
    for frame_a, frame_b in zip(first, second, strict=True):
        assert frame_a.keys() == frame_b.keys()
        for block in frame_a:
            assert np.array_equal(frame_a[block], frame_b[block])
