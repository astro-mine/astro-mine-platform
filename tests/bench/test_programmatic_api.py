"""The programmatic scoring API Studio consumes (RM-P1-BENCH-12; bench.md §6; studio.md §4).

``score_episode`` is the in-process facade Studio's design loop binds to (its injected ``Scorer``):
score one candidate episode trace against a scenario's metric set and get a per-metric
:class:`Scorecard` carrying each metric's uncertainty. ``reference_metric_manifest`` is the Core
``metric``-kind manifest Studio negotiates its objective vocabulary against
(``MetricVocabulary.from_manifest`` reads ``kind`` + ``outputs``). Neither imports Studio — Bench is
the provider; Studio the consumer.
"""

from __future__ import annotations

import pytest

from astro_mine.bench import (
    reference_metric_manifest,
    reference_registry,
    score_episode,
)
from astro_mine.bench.metrics import UnknownMetric
from astro_mine.bench.scenario import MetricRef
from tests.bench._factories import (
    COMMUNITY_METRIC,
    make_observation,
    make_scenario_spec,
    make_trace,
)


def test_score_episode_scores_a_reference_metric() -> None:
    spec = make_scenario_spec(metrics=(MetricRef(name="water_mass"),))
    trace = make_trace([make_observation(0, 0.0, water_kg=5.0)])
    card = score_episode(spec, trace)

    assert card.scenario_id == spec.scenario_id
    assert [m.metric for m in card.metrics] == ["water_mass"]
    assert card.metrics[0].value == pytest.approx(5.0)
    assert card.metrics[0].direction.value == "higher_better"
    # Uncertainty is exposed (None for a single seed, but the field is present for Studio's bounds).
    assert card.metrics[0].dispersion is None


def test_score_episode_uses_a_community_registry() -> None:
    # A scenario that pins a community metric scores only when the registry carries it — the
    # discovered-plugin path, exercised through the Studio-facing facade.
    spec = make_scenario_spec(metrics=(MetricRef(name="earth_contact_uptime"),))
    registry = reference_registry()
    registry.register(COMMUNITY_METRIC)
    trace = make_trace(
        [make_observation(i, i * 60.0, earth_contact=(i % 2 == 0)) for i in range(4)]
    )
    card = score_episode(spec, trace, registry=registry)
    assert card.metrics[0].metric == "earth_contact_uptime"
    assert card.metrics[0].value == pytest.approx(0.5)


def test_score_episode_rejects_an_unregistered_metric() -> None:
    spec = make_scenario_spec(metrics=(MetricRef(name="earth_contact_uptime"),))
    with pytest.raises(UnknownMetric):
        score_episode(spec, make_trace([make_observation(0, 0.0, earth_contact=True)]))


def test_scorecard_is_deterministic_and_content_addressed() -> None:
    spec = make_scenario_spec(metrics=(MetricRef(name="water_mass"),))
    trace = make_trace([make_observation(0, 0.0, water_kg=3.0)])
    assert score_episode(spec, trace).content_hash == score_episode(spec, trace).content_hash


def test_reference_manifest_is_the_studio_vocabulary() -> None:
    manifest = reference_metric_manifest()
    # The two fields MetricVocabulary.from_manifest reads: kind == 'metric', keys from outputs.
    assert manifest.kind.value == "metric"
    vocabulary = {key: "" for key in manifest.outputs}
    assert {"water_mass", "information_gain", "comms_robustness"} <= set(vocabulary)
