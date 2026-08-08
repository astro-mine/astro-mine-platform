"""The local scoring path — ``run(spec, policy)`` (RM-P0-BENCH-05; bench.md §7, §8, §12).

:func:`run` is the platform's headline promise (charter §13): *clone → run the anchor → score
a baseline in an afternoon*, **offline, no account, no cloud**. It resolves the
:class:`~astro_mine.bench.scenario.ScenarioSpec`, drives an :class:`EpisodeRunner` over the
scenario's public seeds (the held-out seeds stay embargoed until evaluation — bench.md §9), and
aggregates the per-seed traces into a content-addressed
:class:`~astro_mine.bench.metrics.Scorecard` via the metric set the scenario pins — each metric
combined by *its own* :class:`~astro_mine.core.objective.MetricAggregation` rule. The Scorecard
is the authoritative score; the reproducibility harness (BENCH-04) is the separate provenance /
determinism oracle.

The runner is injected (conventions.md §1.1; bench.md §2.2): the dependency-clean
:func:`~astro_mine.bench.baseline.reference_episode_runner` is the default, and a real
[Sim](sim.md)-backed runner lives in ``astro-mine-sim`` (the Sim repo) and slots in through the
seam — Bench ships no Sim code and never imports Sim.

Backlog: RM-P0-BENCH-05 — astro-mine-bench#5
"""

from __future__ import annotations

from collections.abc import Sequence

from astro_mine.bench.baseline._runner import (
    EpisodeRunner,
    reference_episode_runner,
    resolve_episode_runner_id,
)
from astro_mine.bench.harness import DeterminismError
from astro_mine.bench.metrics import Scorecard, resolve_metrics, score
from astro_mine.bench.scenario import ScenarioSpec, resolve_scenario
from astro_mine.core.policy import Policy
from astro_mine.core.scoring import EpisodeTrace

__all__ = ["assert_score_reproducible", "run"]


def run(
    spec: ScenarioSpec,
    policy: Policy,
    *,
    runner: EpisodeRunner = reference_episode_runner,
    runner_id: str | None = None,
    seeds: Sequence[int] | None = None,
) -> Scorecard:
    """Score ``policy`` on ``spec`` locally and return its content-addressed :class:`Scorecard`.

    Resolves the spec (validating Core compatibility + content pins), runs ``policy`` through
    ``runner`` on each seed (the scenario's **public** seeds by default), and aggregates the
    per-seed traces with the scenario's pinned metric set. Pure and offline: no cloud, no
    account, no network — a clean clone scores the anchor baseline on one workstation.

    The runner's identity is recorded on the scorecard: the built-in fixture resolves to
    :data:`~astro_mine.bench.baseline.REFERENCE_EPISODE_RUNNER_ID`, and an injected Sim runner
    passes its own ``runner_id`` (e.g. ``SIM_RUNNER_ID``) so the two are distinguishable by
    provenance (bench.md §11).
    """
    resolved = resolve_scenario(spec)
    metrics = resolve_metrics(spec.metrics)
    run_seeds = tuple(seeds) if seeds is not None else spec.seeds.public
    traces: dict[int, EpisodeTrace] = {seed: runner(resolved, policy, seed) for seed in run_seeds}
    rid = resolve_episode_runner_id(runner, runner_id)
    return score(traces, metrics, scenario_id=spec.scenario_id, runner=rid)


def assert_score_reproducible(
    spec: ScenarioSpec,
    policy: Policy,
    *,
    runner: EpisodeRunner = reference_episode_runner,
    runner_id: str | None = None,
    seeds: Sequence[int] | None = None,
    runs: int = 2,
) -> Scorecard:
    """The scoring-path determinism gate: score ``runs`` times and raise on any drift.

    Same inputs + same seeds + same runner ⇒ the identical content-addressed :class:`Scorecard`
    (conventions.md §1.5). Raises :class:`~astro_mine.bench.harness.DeterminismError` if the
    scorecard hash differs across runs — the reproducibility guarantee a leaderboard entry needs.
    Returns the canonical scorecard on success.
    """
    if runs < 2:
        raise ValueError("assert_score_reproducible needs runs >= 2 to compare")
    cards = [
        run(spec, policy, runner=runner, runner_id=runner_id, seeds=seeds) for _ in range(runs)
    ]
    hashes = {card.content_hash for card in cards}
    if len(hashes) != 1:
        raise DeterminismError(
            f"{spec.scenario_id!r} scoring did not reproduce across {runs} runs: {sorted(hashes)}"
        )
    return cards[0]
