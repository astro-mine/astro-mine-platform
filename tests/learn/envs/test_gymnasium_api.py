"""The Gymnasium single-agent / centralized views honor the Gymnasium API (RM-P1-LEARN-01).

The acceptance check: ``gymnasium.utils.env_checker.check_env`` on the single-agent view
(spaces, seeded-reset determinism, step determinism, in-space observations). The
centralized super-agent view is smoke-checked for a valid reset/step round-trip.
"""

from __future__ import annotations

import warnings

from gymnasium.utils.env_checker import check_env

from astro_mine.learn.envs import make_swarm_env
from tests.learn.fakes import FakeSwarmWorld, build_assets


def test_single_agent_view_passes_check_env() -> None:
    view = make_swarm_env(FakeSwarmWorld(), build_assets()).single_agent("rover")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check_env(view)


def test_centralized_view_round_trips() -> None:
    view = make_swarm_env(FakeSwarmWorld(), build_assets()).centralized()
    obs, info = view.reset(seed=0)
    assert obs in view.observation_space
    action = view.action_space.sample()
    obs2, reward, terminated, truncated, _info2 = view.step(action)
    assert obs2 in view.observation_space
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert set(info["agents"]) == set(FakeSwarmWorld().possible_agents)
