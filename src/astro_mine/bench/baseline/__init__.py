"""Baseline policy + the local scoring path (RM-P0-BENCH-05).

The reference :class:`BaselinePolicy` (a Core :class:`~astro_mine.core.policy.Policy`) and
:func:`run` — the offline, no-account "clone, run the anchor, score in an afternoon" entry point
(bench.md §7, §12). :func:`run` drives an injected :class:`EpisodeRunner` over the scenario's
seeds and aggregates the per-seed traces into a content-addressed
:class:`~astro_mine.bench.metrics.Scorecard`; :func:`assert_score_reproducible` is the
scoring-path determinism gate.

The runner is injected to keep the base package dep-clean (core + pydantic) and honour the narrow
waist (conventions.md §1.1; bench.md §2.2): :func:`reference_episode_runner` is the always-available
default fixture, and a real [Sim](sim.md)-backed runner lives in ``astro-mine-sim`` (the Sim repo),
injected through the seam — Bench ships no Sim code.

Backlog: RM-P0-BENCH-05 -- https://github.com/astro-mine/astro-mine-bench/issues/5
"""

from __future__ import annotations

from astro_mine.bench.baseline._policy import BaselinePolicy
from astro_mine.bench.baseline._registry import (
    RUNNER_ENTRYPOINT_GROUP,
    BenchRunnerProvider,
    DefaultPolicyProvider,
    RunnerNotAvailableError,
    ScoringRefused,
    default_policy_for,
    fixture_runner_provider,
    load_runner_provider,
)
from astro_mine.bench.baseline._run import assert_score_reproducible, run
from astro_mine.bench.baseline._runner import (
    REFERENCE_EPISODE_RUNNER_ID,
    EpisodeRunner,
    reference_episode_runner,
    resolve_episode_runner_id,
)

__all__ = [
    "REFERENCE_EPISODE_RUNNER_ID",
    "RUNNER_ENTRYPOINT_GROUP",
    "BaselinePolicy",
    "BenchRunnerProvider",
    "DefaultPolicyProvider",
    "EpisodeRunner",
    "RunnerNotAvailableError",
    "ScoringRefused",
    "assert_score_reproducible",
    "default_policy_for",
    "fixture_runner_provider",
    "load_runner_provider",
    "reference_episode_runner",
    "resolve_episode_runner_id",
    "run",
]
