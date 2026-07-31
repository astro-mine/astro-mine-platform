"""Astro-Mine-Bench — benchmark suite, scenario zoo, and reproducibility harness.

The :mod:`~astro_mine.bench.scenario` spec + content-hash resolver (incl. the anchor
scenario), the reference :mod:`~astro_mine.bench.metrics`, the reproducibility
:mod:`~astro_mine.bench.harness`, the :mod:`~astro_mine.bench.baseline` policy and the
local scoring path :func:`~astro_mine.bench.baseline.run`, and the
:mod:`~astro_mine.bench.leaderboard` service.

**Programmatic scoring API (Studio; bench.md §6).** Studio consumes Bench *programmatically* to
score candidate designs during trade studies — a pure Core consumer that never imports a Bench
private schema (studio.md §2). The in-process surface is :func:`score_episode` (score one candidate
episode trace against a scenario's metric set, returning a per-metric :class:`Scorecard` that
carries each metric's cross-seed **uncertainty**, so Studio surfaces bounds not point estimates) and
:func:`~astro_mine.bench.metrics.reference_metric_manifest` (the Core ``metric``-kind manifest
Studio negotiates its objective vocabulary against). Community metric plugins are discovered from
Hub through a :class:`~astro_mine.bench.metrics.MetricRegistry` (bench.md §3), so a candidate can be
scored against measures that are not Bench built-ins — with no Bench code change.

Reproducibility is the product. See ``docs/architecture/bench.md``.
"""

from __future__ import annotations

from astro_mine.bench._version import __version__
from astro_mine.bench.baseline import REFERENCE_EPISODE_RUNNER_ID, run
from astro_mine.bench.metrics import (
    AggregateScore,
    Metric,
    MetricRegistry,
    Scorecard,
    reference_metric_manifest,
    reference_registry,
    resolve_metrics,
    score,
    scored_metric_values,
)
from astro_mine.bench.scenario import ScenarioSpec
from astro_mine.bench.zoo import list_scenarios, load_scenario
from astro_mine.core.scoring import EpisodeTrace

__all__ = [
    "AggregateScore",
    "Metric",
    "MetricRegistry",
    "ScenarioSpec",
    "Scorecard",
    "__version__",
    "list_scenarios",
    "load_scenario",
    "reference_metric_manifest",
    "reference_registry",
    "resolve_metrics",
    "run",
    "score",
    "score_episode",
    "scored_metric_values",
]


def score_episode(
    scenario: ScenarioSpec,
    trace: EpisodeTrace,
    *,
    seed: int = 0,
    registry: MetricRegistry | None = None,
    runner: str = REFERENCE_EPISODE_RUNNER_ID,
) -> Scorecard:
    """Score one candidate episode ``trace`` against ``scenario``'s metric set (bench.md §6).

    The programmatic scoring facade Studio's design loop binds to (studio.md §4 step 7): resolve the
    scenario's pinned :class:`~astro_mine.bench.scenario.MetricRef` set — through ``registry`` when
    given, so **Hub-discovered community metrics** overlaid on it are in scope (bench.md §3),
    otherwise the built-in reference set — and score the single trace, returning a content-addressed
    :class:`Scorecard`. Each :class:`AggregateScore` carries the metric's value, direction, and
    cross-seed ``dispersion`` (its uncertainty), so a trade study shows bounds rather than a point
    estimate presented as truth (studio.md §7). For a multi-seed candidate, call :func:`score`
    directly with the resolved metrics.

    ``runner`` records which runner produced ``trace`` on the scorecard (bench.md §11); it defaults
    to the reference fixture, so a caller holding a Sim trace should pass the Sim runner id.
    """
    catalog = registry if registry is not None else reference_registry()
    metrics = catalog.resolve_all(scenario.metrics)
    return score({seed: trace}, metrics, scenario_id=scenario.scenario_id, runner=runner)
