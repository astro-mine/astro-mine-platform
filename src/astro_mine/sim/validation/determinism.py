"""Determinism gates — seeded reproducibility + golden traces (RM-P0-SIM-10; sim.md §2.4, §10).

Determinism is a hard requirement, not a hope: same inputs + same seed + same pinned engine
versions must reproduce the same :class:`~astro_mine.sim.runtime.Trace` — and CI fails on
non-reproducibility (sim.md §2.4; conventions.md §11). Two gates:

- :func:`assert_reproducible` — run a scenario repeatedly under one seed and assert byte-identical
  content hashes. The teeth: a non-deterministic engine (an unseeded draw, wall-clock) makes two
  runs disagree and trips this gate.
- :func:`assert_matches_golden` / :func:`golden_hash` — pin the content hash of a reference
  scenario so an *unintended* dynamics change is caught even when each run is internally
  reproducible. Pin golden hashes only for ``BIT_EXACT`` engines (kinematic, granular);
  ``TOLERANCE`` engines (orbital, mobility — RK4/``sqrt``) are gated by the analytic oracles in the
  sibling modules instead, since a bit-exact golden is not portable across builds (sim.md §11).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_mine.sim.runtime.episode import run_episode

if TYPE_CHECKING:
    from astro_mine.core.resource import ResourceField
    from astro_mine.sim.engines import EngineFactory
    from astro_mine.sim.runtime.scenario import Scenario

__all__ = ["DeterminismError", "assert_matches_golden", "assert_reproducible", "golden_hash"]


class DeterminismError(AssertionError):
    """A reproducibility breach: runs of the same pinned inputs disagreed (sim.md §2.4)."""


def golden_hash(
    scenario: Scenario,
    *,
    seed: int | None = None,
    resource_field: ResourceField | None = None,
    engine_factory: EngineFactory | None = None,
) -> str:
    """The content hash of one run of ``scenario`` — the value a golden-trace gate pins.

    ``engine_factory`` selects the physics, defaulting to the same multi-domain coupler
    :func:`~astro_mine.sim.runtime.run_episode` uses. Before #65 these gates could not select an
    engine at all, so they always measured the kinematic reference engine no matter what the
    scenario declared — a determinism gate over a path no real run took.
    """
    return run_episode(
        scenario, seed=seed, resource_field=resource_field, engine_factory=engine_factory
    ).content_hash


def assert_reproducible(
    scenario: Scenario,
    *,
    seed: int | None = None,
    resource_field: ResourceField | None = None,
    engine_factory: EngineFactory | None = None,
    runs: int = 3,
) -> str:
    """Run ``scenario`` ``runs`` times under the same seed and assert byte-identical content hashes.

    The core determinism gate: same inputs + same seed ⇒ identical ``Trace.content_hash``. Raises
    :class:`DeterminismError` on any divergence (CI fails on non-reproducibility). Returns the
    shared hash. Raises ``ValueError`` if asked for fewer than two runs.
    """
    if runs < 2:
        raise ValueError(f"need at least 2 runs to check reproducibility, got {runs}")
    hashes = [
        golden_hash(
            scenario, seed=seed, resource_field=resource_field, engine_factory=engine_factory
        )
        for _ in range(runs)
    ]
    distinct = sorted(set(hashes))
    if len(distinct) != 1:
        raise DeterminismError(
            f"scenario {scenario.name!r} is non-deterministic under seed {seed}: "
            f"{runs} runs produced {len(distinct)} distinct content hashes ({distinct})"
        )
    return distinct[0]


def assert_matches_golden(
    scenario: Scenario,
    expected_hash: str,
    *,
    seed: int | None = None,
    resource_field: ResourceField | None = None,
    engine_factory: EngineFactory | None = None,
) -> str:
    """Assert ``scenario``'s content hash equals the pinned ``expected_hash`` — the golden gate.

    Catches an unintended change to the dynamics (the run is still internally reproducible, but the
    *result* drifted from the pinned reference). Raises :class:`DeterminismError` on a mismatch.
    Returns the actual hash. Use only for ``BIT_EXACT`` reference scenarios.
    """
    actual = golden_hash(
        scenario, seed=seed, resource_field=resource_field, engine_factory=engine_factory
    )
    if actual != expected_hash:
        raise DeterminismError(
            f"scenario {scenario.name!r} drifted from its golden hash: "
            f"expected {expected_hash}, got {actual}"
        )
    return actual
