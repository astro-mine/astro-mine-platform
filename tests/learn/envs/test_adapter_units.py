"""Focused unit coverage for the adapter's edge branches (RM-P1-LEARN-01).

Covers the paths the API-conformance tests do not exercise directly: missing-asset
guard, ``state()`` / ``render`` / ``close``, the single-agent masked and terminal
branches, the centralized view lifecycle, and the encoder's missing-reading /
absent-comms / absent-neighbor fallbacks.
"""

from __future__ import annotations

import numpy as np
import pytest

from astro_mine.core.env.model import ResetResult, StepResult
from astro_mine.core.messages.model import (
    ActionBatch,
    Observation,
    Quat,
    StateSample,
    Transform,
    Vec3,
)
from astro_mine.core.units import MOON_BODY_FIXED
from astro_mine.learn.envs import SwarmEnv, make_swarm_env
from astro_mine.learn.envs.adapter.encode import encode_observation, zero_observation
from astro_mine.learn.envs.adapter.gym_env import SingleAgentEnv
from astro_mine.learn.envs.adapter.spaces import build_agent_spec
from tests.learn.fakes import AGENTS, FakeSwarmWorld, build_assets


def test_missing_asset_is_rejected() -> None:
    assets = build_assets()
    del assets["relay"]
    with pytest.raises(ValueError, match="missing SADF asset"):
        SwarmEnv(FakeSwarmWorld(), assets)


def test_state_view_and_lifecycle(swarm_env: SwarmEnv) -> None:
    swarm_env.reset(seed=0)
    # state() is the concatenated per-agent self_state (5 floats per agent, 3 agents).
    assert swarm_env.state().shape == (15,)
    # Step past digger's termination so it drops out of the last observation map: state()
    # must fall back to zeros for the departed agent (not raise).
    for _ in range(4):
        actions = {a: swarm_env.action_space(a).sample() for a in swarm_env.agents}
        swarm_env.step(actions)
    assert "digger" not in swarm_env.agents
    assert swarm_env.state().shape == (15,)
    assert swarm_env.render() is None
    swarm_env.close()
    # after close the last-observation cache is cleared → all-zeros state
    assert np.array_equal(swarm_env.state(), np.zeros(15, dtype=np.float32))


def test_single_agent_terminal_branch() -> None:
    view = make_swarm_env(FakeSwarmWorld(terminate_at=2), build_assets()).single_agent("digger")
    view.reset(seed=0)
    terminated = False
    for _ in range(4):
        _obs, _reward, terminated, _trunc, _info = view.step({"kind": 0, "mode": 0})
        if terminated:
            break
    assert terminated is True
    assert view.render() is None
    assert view.close() is None


def test_single_agent_masked_branch() -> None:
    view = make_swarm_env(FakeSwarmWorld(), build_assets()).single_agent("relay")
    spec = build_agent_spec("relay", build_assets()["relay"], AGENTS)
    view.reset(seed=0)  # tick 0 — relay observable
    view.step({"kind": 0, "mode": 0})  # tick 1
    obs, _reward, _term, _trunc, info = view.step({"kind": 0, "mode": 0})  # tick 2 → masked
    assert info["observation_mask"] is False
    # masked → the neutral, in-space observation (no sensor content)
    zero = zero_observation(spec)
    assert obs.keys() == zero.keys()
    for block in obs:
        assert np.array_equal(obs[block], zero[block])


def test_centralized_view_lifecycle() -> None:
    view = make_swarm_env(FakeSwarmWorld(), build_assets()).centralized()
    view.reset(seed=1)
    action = view.action_space.sample()
    view.step(action)
    assert view.render() is None
    assert view.close() is None


def test_encode_handles_missing_readings_comms_and_neighbors() -> None:
    # rover declares imaging + neutron sensors and a comms block, but here we hand it an
    # observation with no readings, no comms mask, and no neighbors → every block must
    # fall back to in-space zeros.
    spec = build_agent_spec("rover", build_assets()["rover"], AGENTS)
    obs = Observation(
        tick=0,
        sim_time_s=0.0,
        agent_id="rover",
        self_state=StateSample(
            agent_id="rover",
            frame=MOON_BODY_FIXED,
            pose=Transform(
                translation_m=Vec3(x=1.0, y=2.0, z=3.0),
                rotation_quat_xyzw=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        ),
    )
    sample = encode_observation(obs, spec)
    assert sample in spec.observation_space
    assert np.array_equal(sample["sensing.imaging"], np.zeros_like(sample["sensing.imaging"]))
    assert np.array_equal(sample["comms"], np.zeros_like(sample["comms"]))
    assert np.array_equal(sample["neighbors"], np.zeros_like(sample["neighbors"]))
    # self_state still carries the real pose (battery/temperature default to 0.0).
    assert sample["self_state"][0] == pytest.approx(1.0)


class _AgentNeverObservedWorld:
    """A minimal Core world that never observes its single agent (edge case)."""

    possible_agents = ("ghost",)

    @property
    def agents(self) -> tuple[str, ...]:
        return ("ghost",)

    def reset(self, *, seed: int | None = None, options: object | None = None) -> ResetResult:
        return ResetResult(observations={})

    def step(self, actions: ActionBatch) -> StepResult:
        return StepResult(observations={}, sim_time_s=0.0)


def test_single_agent_view_handles_never_observed_agent() -> None:
    # The controlled agent is absent from both reset and step observations → the view
    # falls back to the neutral observation (reset) and a terminal step.
    spec = build_agent_spec("ghost", build_assets()["rover"], ("ghost",))
    view = SingleAgentEnv(_AgentNeverObservedWorld(), spec)
    obs, info = view.reset(seed=0)
    assert obs in spec.observation_space
    assert info["observation_mask"] is False
    obs2, reward, terminated, truncated, _info = view.step({"kind": 0, "mode": 0})
    assert obs2 in spec.observation_space
    assert reward == 0.0
    assert terminated is True
    assert truncated is False
