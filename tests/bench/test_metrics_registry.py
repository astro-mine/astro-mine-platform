"""The metric registry resolves a scenario's MetricRefs and negotiates versions (Core's rule)."""

from __future__ import annotations

import pytest

from astro_mine.bench.metrics import (
    REGISTRY,
    IncompatibleMetricVersion,
    UnknownMetric,
    resolve_metric,
    resolve_metrics,
)
from astro_mine.bench.scenario import MetricRef

ANCHOR_METRIC_NAMES = {
    "water_mass",
    "energy_per_kg",
    "information_gain",
    "psr_area_characterized",
    "nights_survived",
    "comms_robustness",
    "discovery_latency",
}


def test_registry_holds_all_reference_metrics() -> None:
    assert set(REGISTRY) == ANCHOR_METRIC_NAMES


def test_resolve_known_metric() -> None:
    metric = resolve_metric(MetricRef(name="water_mass", version="0.1.0"))
    assert metric.name == "water_mass"


def test_resolve_unknown_metric_raises() -> None:
    with pytest.raises(UnknownMetric, match="no metric registered"):
        resolve_metric(MetricRef(name="bogus_metric"))


def test_resolve_accepts_same_minor_line() -> None:
    # Pre-1.0 exact-minor rule (Core's check_compatible): 0.1.x satisfies registry 0.1.0.
    assert resolve_metric(MetricRef(name="water_mass", version="0.1.9")).name == "water_mass"


def test_resolve_rejects_different_minor() -> None:
    with pytest.raises(IncompatibleMetricVersion, match=r"requires 0\.2\.0"):
        resolve_metric(MetricRef(name="water_mass", version="0.2.0"))


def test_resolve_metrics_preserves_order() -> None:
    refs = (MetricRef(name="discovery_latency"), MetricRef(name="water_mass"))
    assert [m.name for m in resolve_metrics(refs)] == ["discovery_latency", "water_mass"]
