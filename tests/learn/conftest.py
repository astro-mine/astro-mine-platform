"""Shared fixtures for the SwarmEnv adapter tests (RM-P1-LEARN-01)."""

from __future__ import annotations

import pytest

from astro_mine.core.sadf.model import Asset
from astro_mine.learn.envs import SwarmEnv, make_swarm_env
from tests.learn.fakes import FakeSwarmWorld, build_assets


@pytest.fixture
def assets() -> dict[str, Asset]:
    return build_assets()


@pytest.fixture
def world() -> FakeSwarmWorld:
    return FakeSwarmWorld()


@pytest.fixture
def swarm_env(world: FakeSwarmWorld, assets: dict[str, Asset]) -> SwarmEnv:
    return make_swarm_env(world, assets)
