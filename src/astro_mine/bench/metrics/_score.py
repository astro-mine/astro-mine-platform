"""Scoring + cross-seed aggregation over a metric set (bench.md §3).

:func:`score` runs each metric over every seed's trace, aggregates the per-seed values by that
metric's :class:`~astro_mine.core.objective.MetricAggregation` rule (so a ``lower_better``
latency and a ``higher_better`` mass each combine correctly — not the harness's uniform mean),
and emits a content-addressed :class:`Scorecard`. The scorecard is deterministic and hashable
so the leaderboard's sampled re-execution (bench.md §9) is a hash comparison; it carries
**per-metric** results with direction + uncertainty and no single scalarized score — multi-
objective ranking is a pluggable strategy deferred to Phase 1 (bench.md §11).

Not-applicable episodes (``MetricValue.value is None``) are excluded from the aggregate but
counted in ``n``, so censoring (e.g. seeds that never discovered) stays visible.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from astro_mine.bench.metrics._metric import Metric
from astro_mine.bench.metrics._trace import EpisodeTrace
from astro_mine.bench.scenario._hash import content_hash
from astro_mine.core.objective import MetricAggregation, MetricDirection

__all__ = ["AggregateScore", "Scorecard", "aggregate_scores", "score"]


def _percentile(values: list[float], q: float) -> float:
    """The ``q``-quantile (0..1) by linear interpolation between order statistics."""
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    low = math.floor(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


_AGGREGATORS: dict[MetricAggregation, Callable[[list[float]], float]] = {
    MetricAggregation.MEAN: lambda values: statistics.fmean(values),
    MetricAggregation.MEDIAN: lambda values: float(statistics.median(values)),
    MetricAggregation.MIN: lambda values: min(values),
    MetricAggregation.MAX: lambda values: max(values),
    MetricAggregation.SUM: lambda values: math.fsum(values),
    MetricAggregation.P05: lambda values: _percentile(values, 0.05),
    MetricAggregation.P95: lambda values: _percentile(values, 0.95),
}


class _Model(BaseModel):
    """Frozen base: reject unknown fields, immutable once built."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class AggregateScore(_Model):
    """One metric's score across the scored seeds.

    ``value`` is the per-seed values combined by ``aggregation`` (``None`` if no seed was
    applicable); ``dispersion`` is the sample standard deviation across applicable seeds
    (``None`` for fewer than two); ``n`` is how many seeds were applicable; ``per_seed`` is the
    raw per-seed values aligned with ``seeds`` (``None`` where the metric did not apply).
    """

    metric: str
    version: str
    unit: str
    direction: MetricDirection
    aggregation: MetricAggregation
    value: float | None
    dispersion: float | None
    n: int
    seeds: tuple[int, ...]
    per_seed: tuple[float | None, ...]


class Scorecard(_Model):
    """A per-metric benchmark scorecard, content-addressed for reproducible verification.

    ``runner`` is the identity of the :class:`~astro_mine.bench.baseline.EpisodeRunner` that
    produced the scored traces — the dependency-clean fixture
    (:data:`~astro_mine.bench.baseline.REFERENCE_EPISODE_RUNNER_ID`) or an injected Sim runner.
    It is part of the scorecard's identity and folds into :attr:`content_hash`, so a fixture
    score and a Sim score are distinguishable by *provenance*, not only by value (conventions.md
    §1.5; bench.md §2.1, §11 "signed runner attestation"). A scorecard that does not name its
    runner is not reproducible provenance, so the field is required at construction.
    """

    scenario_id: str | None
    runner: str
    metrics: tuple[AggregateScore, ...]

    @property
    def content_hash(self) -> str:
        """A deterministic ``sha256:`` digest over the scorecard (bench.md §9 re-execution)."""
        return content_hash(self.model_dump(mode="json"))


def aggregate_scores(
    metrics: Sequence[Metric],
    per_seed_by_metric: Mapping[str, Sequence[float | None]],
    seeds: Sequence[int],
    *,
    scenario_id: str | None = None,
    runner: str,
) -> Scorecard:
    """Aggregate **pre-computed** per-seed metric values into a content-addressed Scorecard.

    ``per_seed_by_metric`` maps each metric's name to its per-seed values aligned with ``seeds``
    (already in the caller's order); each metric is combined by *its own* aggregation rule, with
    not-applicable (``None``) seeds excluded from the aggregate but counted in ``n``. ``runner`` is
    the identity of the runner that produced the traces, recorded on the scorecard so a fixture and
    a Sim score are distinguishable by provenance (bench.md §11); it is required, not defaulted, so
    no aggregation path can emit an unattributed score. This is the shared aggregation kernel:
    :func:`score` computes the values from traces in-process and calls it, while the Cloud
    evaluation collector (RM-P1-BENCH-11) reads the values back from the workers' Parquet and calls
    it — so a cluster-collected scorecard is **byte-identical** to the workstation scorecard for the
    same inputs + seeds + runner (bench.md §7, §8).
    """
    seeds_t = tuple(seeds)
    aggregates: list[AggregateScore] = []
    for metric in metrics:
        per_seed = tuple(per_seed_by_metric[metric.name])
        if len(per_seed) != len(seeds_t):
            raise ValueError(
                f"metric {metric.name!r} has {len(per_seed)} per-seed values for "
                f"{len(seeds_t)} seeds"
            )
        applicable = [value for value in per_seed if value is not None]
        aggregate = _AGGREGATORS[metric.aggregation](applicable) if applicable else None
        dispersion = statistics.stdev(applicable) if len(applicable) >= 2 else None
        aggregates.append(
            AggregateScore(
                metric=metric.name,
                version=metric.version,
                unit=metric.unit,
                direction=metric.direction,
                aggregation=metric.aggregation,
                value=aggregate,
                dispersion=dispersion,
                n=len(applicable),
                seeds=seeds_t,
                per_seed=per_seed,
            )
        )
    return Scorecard(scenario_id=scenario_id, runner=runner, metrics=tuple(aggregates))


def score(
    traces_by_seed: Mapping[int, EpisodeTrace],
    metrics: Sequence[Metric],
    *,
    scenario_id: str | None = None,
    runner: str,
) -> Scorecard:
    """Score every seed's trace against ``metrics`` and aggregate into a :class:`Scorecard`.

    ``traces_by_seed`` maps a seed to that episode's trace; ``metrics`` are resolved metric
    plugins (see :func:`~astro_mine.bench.metrics.resolve_metrics`); ``runner`` is the identity of
    the runner that produced those traces, recorded on the scorecard (bench.md §11). Seeds are
    scored in sorted order so the scorecard — and its hash — are independent of insertion order.
    Delegates the aggregation to :func:`aggregate_scores`, the kernel the Cloud eval collector
    also uses.
    """
    seeds = tuple(sorted(traces_by_seed))
    per_seed_by_metric = {
        metric.name: [metric.compute(traces_by_seed[seed]).value for seed in seeds]
        for metric in metrics
    }
    return aggregate_scores(
        metrics, per_seed_by_metric, seeds, scenario_id=scenario_id, runner=runner
    )
