"""score() aggregates per-metric across seeds and emits a content-addressed Scorecard."""

from __future__ import annotations

import dataclasses
import statistics
from collections.abc import Mapping

import pytest

from astro_mine.bench.baseline import REFERENCE_EPISODE_RUNNER_ID
from astro_mine.bench.metrics import (
    EpisodeTrace,
    MetricAggregation,
    MetricDirection,
    MetricValue,
    resolve_metrics,
    score,
)
from astro_mine.bench.scenario import MetricRef
from tests.bench._factories import make_observation, make_trace

#: An arbitrary runner identity for scoring-kernel tests (the kernel records it, never resolves it).
_RUNNER = REFERENCE_EPISODE_RUNNER_ID


@dataclasses.dataclass(frozen=True)
class _ProbeMetric:
    """A controllable metric: reports the first observation's ``sim_time_s`` as its value, or
    ``None`` (not applicable) for an empty trace — so a test fixes each seed's value directly."""

    name: str = "probe"
    version: str = "0.1.0"
    unit: str = "x"
    direction: MetricDirection = MetricDirection.HIGHER_BETTER
    aggregation: MetricAggregation = MetricAggregation.MEAN

    def compute(self, trace: EpisodeTrace) -> MetricValue:
        if not trace.observations:
            return MetricValue(value=None, unit=self.unit)
        return MetricValue(value=trace.observations[0].sim_time_s, unit=self.unit)


def _traces(values: Mapping[int, float | None]) -> dict[int, EpisodeTrace]:
    return {
        seed: make_trace(() if value is None else (make_observation(0, value),))
        for seed, value in values.items()
    }


def test_score_aggregates_mean_with_dispersion_and_provenance() -> None:
    card = score(_traces({1: 2.0, 2: 4.0, 3: 6.0}), [_ProbeMetric()], runner=_RUNNER)
    agg = card.metrics[0]
    assert agg.value == pytest.approx(4.0)
    assert agg.n == 3
    assert agg.seeds == (1, 2, 3)
    assert agg.per_seed == (2.0, 4.0, 6.0)
    assert agg.dispersion == pytest.approx(statistics.stdev([2.0, 4.0, 6.0]))
    assert agg.direction is MetricDirection.HIGHER_BETTER
    assert card.runner == _RUNNER


@pytest.mark.parametrize(
    ("aggregation", "expected"),
    [
        (MetricAggregation.MEAN, 4.0),
        (MetricAggregation.MEDIAN, 2.0),
        (MetricAggregation.MIN, 1.0),
        (MetricAggregation.MAX, 9.0),
        (MetricAggregation.SUM, 12.0),
        (MetricAggregation.P05, 1.1),  # linear interp over [1,2,9] at q=0.05
        (MetricAggregation.P95, 8.3),  # linear interp over [1,2,9] at q=0.95
    ],
)
def test_score_honors_each_aggregation_rule(
    aggregation: MetricAggregation, expected: float
) -> None:
    metric = _ProbeMetric(aggregation=aggregation)
    card = score(_traces({1: 1.0, 2: 2.0, 3: 9.0}), [metric], runner=_RUNNER)
    assert card.metrics[0].value == pytest.approx(expected)


def test_percentile_single_applicable_seed() -> None:
    metric = _ProbeMetric(aggregation=MetricAggregation.P95)
    assert score(_traces({7: 5.0}), [metric], runner=_RUNNER).metrics[0].value == pytest.approx(5.0)


def test_score_excludes_not_applicable_seeds_but_counts_them() -> None:
    card = score(_traces({1: 2.0, 2: None, 3: 4.0}), [_ProbeMetric()], runner=_RUNNER)
    agg = card.metrics[0]
    assert agg.n == 2
    assert agg.value == pytest.approx(3.0)  # mean of the applicable (2, 4)
    assert agg.per_seed == (2.0, None, 4.0)
    assert agg.seeds == (1, 2, 3)


def test_score_all_not_applicable_yields_none() -> None:
    agg = score(_traces({1: None, 2: None}), [_ProbeMetric()], runner=_RUNNER).metrics[0]
    assert agg.value is None
    assert agg.n == 0
    assert agg.dispersion is None


def test_dispersion_none_for_single_applicable_seed() -> None:
    assert score(_traces({1: 5.0}), [_ProbeMetric()], runner=_RUNNER).metrics[0].dispersion is None


def test_scorecard_is_seed_order_independent() -> None:
    kw = {"scenario_id": "s", "runner": _RUNNER}
    unordered = score(_traces({3: 9.0, 1: 1.0, 2: 2.0}), [_ProbeMetric()], **kw)
    ordered = score(_traces({1: 1.0, 2: 2.0, 3: 9.0}), [_ProbeMetric()], **kw)
    assert ordered.metrics[0].seeds == (1, 2, 3)
    assert unordered.content_hash == ordered.content_hash


def test_scorecard_content_hash_is_deterministic_and_sensitive() -> None:
    kw = {"scenario_id": "s", "runner": _RUNNER}
    base = score(_traces({1: 1.0, 2: 2.0}), [_ProbeMetric()], **kw)
    same = score(_traces({1: 1.0, 2: 2.0}), [_ProbeMetric()], **kw)
    changed = score(_traces({1: 1.0, 2: 3.0}), [_ProbeMetric()], **kw)
    assert base.content_hash == same.content_hash
    assert base.content_hash != changed.content_hash
    assert base.content_hash.startswith("sha256:")


def test_runner_is_part_of_scorecard_identity() -> None:
    """A fixture score and a Sim score with identical metric values must not collide (G1.1/G1.8).

    The runner is provenance, not a value, so two scorecards that agree on every metric but were
    produced by different runners hash differently — otherwise a fixture score is indistinguishable
    from a Sim score in a leaderboard or a paper.
    """
    traces = _traces({1: 1.0, 2: 2.0})
    fixture = score(traces, [_ProbeMetric()], scenario_id="s", runner="fixture/0.1.0")
    sim = score(traces, [_ProbeMetric()], scenario_id="s", runner="sim/0.1.0")
    assert fixture.metrics == sim.metrics  # identical metric values...
    assert fixture.runner != sim.runner
    assert fixture.content_hash != sim.content_hash  # ...but distinguishable by provenance


def test_score_end_to_end_with_registry_resolved_metrics() -> None:
    metrics = resolve_metrics((MetricRef(name="water_mass"), MetricRef(name="comms_robustness")))
    traces = {
        1: make_trace([make_observation(0, 0.0, water_kg=2.0, earth_contact=True)]),
        2: make_trace([make_observation(0, 0.0, water_kg=4.0, earth_contact=False)]),
    }
    card = score(traces, metrics, scenario_id="lunar-polar-ice-prospecting-v1", runner=_RUNNER)
    assert card.scenario_id == "lunar-polar-ice-prospecting-v1"
    by_name = {agg.metric: agg for agg in card.metrics}
    assert by_name["water_mass"].value == pytest.approx(3.0)  # mean(2, 4)
    assert by_name["water_mass"].unit == "kg"
    assert by_name["comms_robustness"].value == pytest.approx(0.5)  # mean(1.0, 0.0)
