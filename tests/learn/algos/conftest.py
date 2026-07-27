"""Shared fixtures for the RM-P1-LEARN-03 baseline tests."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from astro_mine.learn import make_swarm_env
from astro_mine.learn.algos import TrainConfig
from astro_mine.learn.envs import CommsModel, CommsModelConfig, DropConfig, SwarmEnv
from tests.learn.fakes import FakeSwarmWorld, build_assets

EnvFactory = Callable[[], SwarmEnv]


@pytest.fixture
def env_factory() -> EnvFactory:
    """A fresh SwarmEnv over the conformant FakeSwarmWorld + heterogeneous assets."""
    return lambda: make_swarm_env(FakeSwarmWorld(), build_assets())


@pytest.fixture
def comms_env_factory() -> EnvFactory:
    """A SwarmEnv whose comms are degraded by a stochastic-drop CommsModel — exercises the
    comms provenance / comms-stress ledger path."""
    config = CommsModelConfig(drop=DropConfig(probability=0.5))
    return lambda: make_swarm_env(FakeSwarmWorld(), build_assets(), comms_model=CommsModel(config))


@pytest.fixture
def tiny_config() -> TrainConfig:
    """A tier-1, CPU-fast training config for the smoke/determinism gates."""
    return TrainConfig(iterations=2, rollout_steps=8, hidden_sizes=(16, 16))
