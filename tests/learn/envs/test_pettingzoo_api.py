"""SwarmEnv honors the PettingZoo ParallelEnv API (RM-P1-LEARN-01).

The acceptance check: ``pettingzoo.test.parallel_api_test`` drives the wrapped env for
many cycles and asserts the full parallel contract — reset/step tuple shapes, live-agent
bookkeeping, no revival after termination/truncation, and stable per-agent spaces. The
run also exercises the fake's partial observability, comms mask, agent attrition, and
horizon truncation.
"""

from __future__ import annotations

import warnings

from pettingzoo.test import parallel_api_test

from astro_mine.learn.envs import SwarmEnv, make_swarm_env
from tests.learn.fakes import FakeSwarmWorld, build_assets


def test_parallel_api_contract() -> None:
    env = make_swarm_env(FakeSwarmWorld(), build_assets())
    # The adapter deliberately masks/omits real content for unobservable agents, so the
    # test's optional "live agent was not given observation" note is expected — assert the
    # hard contract holds and ignore the advisory warnings.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parallel_api_test(env, num_cycles=100)


def test_metadata_and_agent_lifecycle() -> None:
    env = make_swarm_env(FakeSwarmWorld(horizon=4, terminate_at=2), build_assets())
    assert env.metadata["name"] == "swarm_env_v0"
    assert env.metadata["is_parallelizable"] is True
    assert isinstance(env, SwarmEnv)

    env.reset(seed=0)
    assert set(env.agents) == set(env.possible_agents)
    # digger terminates at tick 2; step twice to see it leave and never return.
    for _ in range(2):
        actions = {a: env.action_space(a).sample() for a in env.agents}
        env.step(actions)
    assert "digger" not in env.agents
    # horizon truncation empties the swarm.
    while env.agents:
        actions = {a: env.action_space(a).sample() for a in env.agents}
        env.step(actions)
    assert env.agents == []
