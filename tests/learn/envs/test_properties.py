"""Property-based invariants of the adapter's spaces and mask surfaces (RM-P1-LEARN-01).

Two families of properties, driven by Hypothesis:

(a) **mask / comms consistency** — a masked (or absent) agent never yields real
    observation content, and a comms channel only ever lists *reachable* peers that are
    real ``possible_agents`` (never the agent itself);
(b) **space conformance** — any sampled action decodes to a well-formed Core
    :class:`~astro_mine.core.messages.model.Action`, and every wrapped observation lands
    ``in`` the agent's declared observation space.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from astro_mine.core.messages.model import Action
from astro_mine.learn.envs import SwarmEnv, make_swarm_env
from astro_mine.learn.envs.adapter.encode import ObsSample, decode_action, zero_observation
from astro_mine.learn.envs.adapter.spaces import build_agent_spec
from tests.learn.fakes import AGENTS, FakeSwarmWorld, build_assets


def _obs_equal(a: ObsSample, b: ObsSample) -> bool:
    return a.keys() == b.keys() and all(np.array_equal(a[k], b[k]) for k in a)


def _assert_frame(
    obs: dict[str, ObsSample],
    infos: dict[str, Any],
    zeros: dict[str, ObsSample],
    env: SwarmEnv,
) -> None:
    for agent, sample in obs.items():
        assert agent in env.possible_agents
        # A masked/unobservable agent leaks no sensor content — only the neutral encoding.
        if infos[agent]["observation_mask"] is False:
            assert _obs_equal(sample, zeros[agent])
        # The comms channel lists only reachable, real, non-self peers.
        for link in infos[agent]["comms"]:
            assert link.reachable is True
            assert link.peer in env.possible_agents
            assert link.peer != agent
        assert sample in env.observation_space(agent)


@settings(deadline=None, max_examples=50)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1), steps=st.integers(min_value=0, max_value=15)
)
def test_mask_and_comms_consistency(seed: int, steps: int) -> None:
    assets = build_assets()
    zeros = {a: zero_observation(build_agent_spec(a, assets[a], AGENTS)) for a in AGENTS}
    env = make_swarm_env(FakeSwarmWorld(), assets)
    obs, infos = env.reset(seed=seed)
    _assert_frame(obs, infos, zeros, env)
    for _ in range(steps):
        if not env.agents:
            break
        actions = {a: env.action_space(a).sample() for a in env.agents}
        obs, _, _, _, infos = env.step(actions)
        _assert_frame(obs, infos, zeros, env)


@settings(deadline=None, max_examples=50)
@given(seed=st.integers(min_value=0, max_value=2**31 - 1))
def test_sampled_actions_decode_and_observations_are_in_space(seed: int) -> None:
    assets = build_assets()
    specs = {a: build_agent_spec(a, assets[a], AGENTS) for a in AGENTS}
    env = make_swarm_env(FakeSwarmWorld(), assets)
    obs, _ = env.reset(seed=seed)
    for agent in env.agents:
        space = env.action_space(agent)
        space.seed(seed)
        for _ in range(5):
            action = decode_action(space.sample(), specs[agent])
            assert isinstance(action, Action)
            # A well-formed Core message round-trips through model validation unchanged.
            assert Action.model_validate(action.model_dump()) == action
        assert obs[agent] in env.observation_space(agent)
