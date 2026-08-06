"""Reference metric set + scoring/aggregation (RM-P0-BENCH-03; bench.md §3).

The plugin-based metric set: seven reference metrics — water mass, energy/kg, information
gain, PSR area characterized, nights survived, comms robustness, discovery latency — each a
:class:`Metric` with SI-consistent units, a direction, and a cross-seed aggregation rule,
computing deterministically from an :class:`EpisodeTrace`. :func:`score` runs a metric set over
per-seed traces and aggregates into a content-addressed :class:`Scorecard`.

The metric input (:class:`EpisodeTrace`) is the Bench-side view of Sim's MCAP recording; the
MCAP decoder + ``Sim → EpisodeTrace`` adapter land with the real Sim runner in Phase 1
(mirroring the harness's ``reference_runner`` deferral). Direction and aggregation reuse Core's
objective→metric vocabulary (:class:`MetricDirection` / :class:`MetricAggregation`).

Public API:

- the contract — :class:`Metric`, :class:`MetricValue`, and the :class:`EpisodeTrace` /
  :class:`ScoringContext` / :class:`BeliefSnapshot` input;
- the reference metrics — :data:`REFERENCE_METRICS` (and each class);
- the registry — :data:`REGISTRY`, :func:`resolve_metric` / :func:`resolve_metrics`;
- scoring — :func:`score`, :class:`Scorecard`, :class:`AggregateScore`;
- errors — :class:`MetricError` / :class:`MetricComputationError` / :class:`UnknownMetric` /
  :class:`IncompatibleMetricVersion`.

Backlog: RM-P0-BENCH-03 — https://github.com/astro-mine/astro-mine-bench/issues/3
"""

from __future__ import annotations

from astro_mine.bench._hub_payload import PayloadRetrievalError, pull_verified_layer
from astro_mine.bench.metrics._manifest import (
    REFERENCE_METRIC_MANIFEST_NAME,
    metric_manifest,
    reference_metric_manifest,
)
from astro_mine.bench.metrics._metric import (
    Metric,
    MetricComputationError,
    MetricError,
    MetricValue,
)
from astro_mine.bench.metrics._plugin import (
    HubRegistry,
    MetricEntrypointError,
    MetricManifestError,
    MetricPluginError,
    MetricPluginLoader,
    ResolvedMetricPlugin,
    discover_metric,
    load_metric,
    open_registry,
    reference_metric_loader,
    resolve_metric_plugin,
    validate_metric_manifest,
)
from astro_mine.bench.metrics._reference import (
    REFERENCE_METRICS,
    CommsRobustness,
    DiscoveryLatency,
    EnergyPerKg,
    InformationGain,
    NightsSurvived,
    PsrAreaCharacterized,
    WaterMass,
)
from astro_mine.bench.metrics._registry import (
    REGISTRY,
    DuplicateMetric,
    IncompatibleMetricVersion,
    MetricRegistry,
    UnknownMetric,
    reference_registry,
    resolve_metric,
    resolve_metrics,
)
from astro_mine.bench.metrics._score import (
    AggregateScore,
    Scorecard,
    aggregate_scores,
    score,
    scored_metric_values,
)
from astro_mine.core.objective import MetricAggregation, MetricDirection

__all__ = [
    "REFERENCE_METRICS",
    "REFERENCE_METRIC_MANIFEST_NAME",
    "REGISTRY",
    "AggregateScore",
    "CommsRobustness",
    "DiscoveryLatency",
    "DuplicateMetric",
    "EnergyPerKg",
    "HubRegistry",
    "IncompatibleMetricVersion",
    "InformationGain",
    "Metric",
    "MetricAggregation",
    "MetricComputationError",
    "MetricDirection",
    "MetricEntrypointError",
    "MetricError",
    "MetricManifestError",
    "MetricPluginError",
    "MetricPluginLoader",
    "MetricRegistry",
    "MetricValue",
    "NightsSurvived",
    "PayloadRetrievalError",
    "PsrAreaCharacterized",
    "ResolvedMetricPlugin",
    "Scorecard",
    "UnknownMetric",
    "WaterMass",
    "aggregate_scores",
    "discover_metric",
    "load_metric",
    "metric_manifest",
    "open_registry",
    "pull_verified_layer",
    "reference_metric_loader",
    "reference_metric_manifest",
    "reference_registry",
    "resolve_metric",
    "resolve_metric_plugin",
    "resolve_metrics",
    "score",
    "scored_metric_values",
    "validate_metric_manifest",
]
