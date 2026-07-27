"""The in-repo fake Core world honors the Core Environment contract (RM-P1-LEARN-01).

Learn's whole test suite rides on this fake, so it must itself pass Core's own
consumer-driven conformance check — with **no** ``astro_mine.sim`` import in the loop.
"""

from __future__ import annotations

from astro_mine.core.env import check_environment
from tests.learn.fakes import FakeSwarmWorld


def test_fake_world_passes_core_conformance() -> None:
    check_environment(FakeSwarmWorld())


def test_fake_world_conformance_across_seeds() -> None:
    for seed in (0, 1, 7, 123):
        check_environment(FakeSwarmWorld(), seed=seed)
