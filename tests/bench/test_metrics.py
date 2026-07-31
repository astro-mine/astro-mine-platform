"""Each reference metric computes deterministically from a trace with typed units."""

from __future__ import annotations

import math

import pytest

from astro_mine.bench.metrics import (
    REFERENCE_METRICS,
    CommsRobustness,
    DiscoveryLatency,
    EnergyPerKg,
    InformationGain,
    MetricAggregation,
    MetricComputationError,
    MetricDirection,
    MetricValue,
    NightsSurvived,
    PsrAreaCharacterized,
    WaterMass,
)
from astro_mine.core.messages.model import SensorReading
from astro_mine.core.scoring import ScoringContext
from tests.bench._factories import belief, belief_snapshot, make_observation, make_trace

ANCHOR_METRIC_NAMES = (
    "water_mass",
    "energy_per_kg",
    "information_gain",
    "psr_area_characterized",
    "nights_survived",
    "comms_robustness",
    "discovery_latency",
)


def test_reference_set_is_the_seven_anchor_metrics() -> None:
    assert tuple(m.name for m in REFERENCE_METRICS) == ANCHOR_METRIC_NAMES
    assert all(m.version == "0.1.0" for m in REFERENCE_METRICS)


def test_metric_metadata() -> None:
    assert WaterMass().unit == "kg"
    assert WaterMass().direction is MetricDirection.HIGHER_BETTER
    assert EnergyPerKg().direction is MetricDirection.LOWER_BETTER
    assert DiscoveryLatency().direction is MetricDirection.LOWER_BETTER
    assert DiscoveryLatency().aggregation is MetricAggregation.MEDIAN
    assert NightsSurvived().aggregation is MetricAggregation.MIN


def test_water_mass_sums_latest_reading_per_agent() -> None:
    trace = make_trace(
        [
            make_observation(0, 0.0, "rover-a", water_kg=1.0),
            make_observation(1, 10.0, "rover-a", water_kg=3.0),  # later tick wins
            make_observation(0, 0.0, "rover-b", water_kg=2.0),
        ]
    )
    value = WaterMass().compute(trace)
    assert value.value == pytest.approx(5.0)  # 3 (rover-a latest) + 2 (rover-b)
    assert value.unit == "kg"
    assert value.uncertainty is None


def test_water_mass_reads_the_stored_mass_channel_not_the_energy_channel() -> None:
    """An ISRU gauge reports ``[stored_water_kg, extraction_energy_j]`` on one reading (RFC-0003).

    The regression this pins (astro-mine-sim#61): reading ``values[-1]`` scored the cumulative
    extraction *energy* in joules and reported it as kilograms of water — a confidently wrong,
    wildly larger number, on the anchor's headline metric.
    """
    gauge = SensorReading(
        sensor="tank",
        values=[12.5, 2.5e7],  # 12.5 kg stored, 25 MJ spent extracting it
        unit="kg",
        resource_species="water",
    )
    observation = make_observation(0, 0.0, "plant").model_copy(update={"sensors": (gauge,)})

    value = WaterMass().compute(make_trace([observation]))

    assert value.value == pytest.approx(12.5)
    assert value.unit == "kg"


def test_water_mass_is_unchanged_for_a_single_channel_gauge() -> None:
    """A one-value reading is untouched by the channel fix — ``values[0]`` is ``values[-1]``."""
    trace = make_trace([make_observation(0, 0.0, "rover", water_kg=4.0)])
    assert WaterMass().compute(trace).value == pytest.approx(4.0)


def test_water_mass_is_zero_without_readings() -> None:
    assert WaterMass().compute(make_trace([make_observation(0, 0.0)])).value == 0.0


def test_energy_per_kg_divides_discharge_by_water() -> None:
    trace = make_trace(
        [
            make_observation(0, 0.0, "r", battery_soc_j=100.0, water_kg=0.0),
            make_observation(1, 1.0, "r", battery_soc_j=60.0, water_kg=2.0),  # -40 J, 2 kg
        ]
    )
    value = EnergyPerKg().compute(trace)
    assert value.value == pytest.approx(20.0)
    assert value.unit == "J/kg"


def test_energy_per_kg_ignores_recharge() -> None:
    trace = make_trace(
        [
            make_observation(0, 0.0, "r", battery_soc_j=50.0, water_kg=1.0),
            make_observation(1, 1.0, "r", battery_soc_j=80.0, water_kg=1.0),  # recharge, ignored
            make_observation(2, 2.0, "r", battery_soc_j=60.0, water_kg=1.0),  # -20 discharge
        ]
    )
    assert EnergyPerKg().compute(trace).value == pytest.approx(20.0)


def test_energy_per_kg_not_applicable_without_water() -> None:
    trace = make_trace(
        [
            make_observation(0, 0.0, "r", battery_soc_j=100.0),
            make_observation(1, 1.0, "r", battery_soc_j=60.0),
        ]
    )
    assert EnergyPerKg().compute(trace).value is None


def test_comms_robustness_is_earth_contact_fraction() -> None:
    trace = make_trace(
        [
            make_observation(0, 0.0, earth_contact=True),
            make_observation(1, 1.0, earth_contact=False),
            make_observation(2, 2.0, earth_contact=True),
            make_observation(3, 3.0, earth_contact=True),
        ]
    )
    assert CommsRobustness().compute(trace).value == pytest.approx(0.75)


def test_comms_robustness_not_applicable_without_mask() -> None:
    assert CommsRobustness().compute(make_trace([make_observation(0, 0.0)])).value is None


def test_discovery_latency_is_first_detection_time() -> None:
    ctx = ScoringContext(discovery_threshold=0.5)
    trace = make_trace(
        [
            make_observation(0, 0.0, detection=0.2),  # below threshold
            make_observation(2, 9.0, detection=1.0),
            make_observation(1, 5.0, detection=0.9),  # earliest detection, out of tick order
        ],
        ctx,
    )
    value = DiscoveryLatency().compute(trace)
    assert value.value == pytest.approx(5.0)
    assert value.unit == "s"


def test_discovery_latency_not_applicable_when_never_detected() -> None:
    ctx = ScoringContext(discovery_threshold=0.5)
    trace = make_trace([make_observation(0, 0.0, detection=0.1)], ctx)
    assert DiscoveryLatency().compute(trace).value is None


def test_information_gain_is_variance_reduction_in_nats() -> None:
    ctx = ScoringContext(
        prior_belief={"c1": belief(0.0, 4.0), "c2": belief(0.0, 4.0)},
        belief_history=(belief_snapshot(10.0, {"c1": belief(0.0, 1.0), "c2": belief(0.0, 2.0)}),),
    )
    value = InformationGain().compute(make_trace(context=ctx))
    expected = 0.5 * math.log(4.0 / 1.0) + 0.5 * math.log(4.0 / 2.0)
    assert value.value == pytest.approx(expected)
    assert value.unit == "nat"
    assert value.uncertainty is not None and value.uncertainty >= 0.0


def test_information_gain_not_applicable_without_belief() -> None:
    assert InformationGain().compute(make_trace()).value is None


def test_information_gain_requires_positive_variance() -> None:
    ctx = ScoringContext(
        prior_belief={"c1": belief(0.0, 4.0)},
        belief_history=(belief_snapshot(1.0, {"c1": belief(0.0, 0.0)}),),
    )
    with pytest.raises(MetricComputationError, match="positive belief variance"):
        InformationGain().compute(make_trace(context=ctx))


def test_information_gain_not_applicable_when_cells_disjoint() -> None:
    # A prior cell absent from the posterior contributes nothing; with no overlap at all the
    # metric is not applicable rather than zero.
    ctx = ScoringContext(
        prior_belief={"c2": belief(0.0, 4.0)},
        belief_history=(belief_snapshot(1.0, {"c1": belief(0.0, 1.0)}),),
    )
    assert InformationGain().compute(make_trace(context=ctx)).value is None


def test_psr_area_counts_characterized_cells_times_area() -> None:
    ctx = ScoringContext(
        psr_cells=frozenset({"c1", "c2", "c3"}),
        cell_area_m2=10.0,
        characterized_variance_threshold=1.0,
        belief_history=(
            belief_snapshot(
                5.0,
                {"c1": belief(0.0, 0.5), "c2": belief(0.0, 0.9), "c3": belief(0.0, 2.0)},
            ),
        ),
    )
    value = PsrAreaCharacterized().compute(make_trace(context=ctx))
    assert value.value == pytest.approx(20.0)  # c1 & c2 at/under threshold -> 2 cells * 10 m^2
    assert value.unit == "m^2"


def test_psr_area_not_applicable_without_mask_or_belief() -> None:
    assert PsrAreaCharacterized().compute(make_trace()).value is None


def test_nights_survived_counts_intervals_with_power() -> None:
    ctx = ScoringContext(night_intervals=((0.0, 10.0), (20.0, 30.0)))
    trace = make_trace(
        [
            make_observation(0, 5.0, "r", battery_soc_j=50.0),  # night 1: alive
            make_observation(1, 25.0, "r", battery_soc_j=0.0),  # night 2: depleted
        ],
        ctx,
    )
    assert NightsSurvived().compute(trace).value == pytest.approx(1.0)


def test_nights_survived_applies_thermal_floor() -> None:
    ctx = ScoringContext(night_intervals=((0.0, 10.0),), survivable_temperature_k=100.0)
    trace = make_trace(
        [make_observation(0, 5.0, "r", battery_soc_j=50.0, temperature_k=90.0)],  # too cold
        ctx,
    )
    assert NightsSurvived().compute(trace).value == 0.0


def test_nights_survived_not_applicable_without_intervals() -> None:
    trace = make_trace([make_observation(0, 0.0, "r", battery_soc_j=1.0)])
    assert NightsSurvived().compute(trace).value is None


def test_nights_survived_skips_samples_without_battery_state() -> None:
    ctx = ScoringContext(night_intervals=((0.0, 10.0),))
    trace = make_trace(
        [
            make_observation(0, 3.0, "r"),  # in-window but no battery state -> skipped
            make_observation(1, 6.0, "r", battery_soc_j=40.0),  # alive
        ],
        ctx,
    )
    assert NightsSurvived().compute(trace).value == pytest.approx(1.0)


def test_metric_value_rejects_non_finite() -> None:
    with pytest.raises(MetricComputationError):
        MetricValue(value=math.inf, unit="kg")
    with pytest.raises(MetricComputationError):
        MetricValue(value=1.0, unit="kg", uncertainty=-1.0)
    with pytest.raises(MetricComputationError):
        MetricValue(value=1.0, unit="kg", uncertainty=math.nan)


def test_all_metrics_are_deterministic() -> None:
    ctx = ScoringContext(
        prior_belief={"c1": belief(0.0, 4.0)},
        belief_history=(belief_snapshot(1.0, {"c1": belief(0.0, 1.0)}),),
        psr_cells=frozenset({"c1"}),
        characterized_variance_threshold=1.0,
        night_intervals=((0.0, 5.0),),
        discovery_threshold=0.5,
    )
    trace = make_trace(
        [
            make_observation(
                0, 1.0, "r", battery_soc_j=10.0, water_kg=2.0, detection=1.0, earth_contact=True
            )
        ],
        ctx,
    )
    for metric in REFERENCE_METRICS:
        assert metric.compute(trace) == metric.compute(trace)


def _gauge_trace(*, stored_kg: float, extraction_j: float | None, soc: tuple[float, float]):
    """A two-tick trace whose plant carries an ISRU gauge and discharges its battery."""
    values = [stored_kg] if extraction_j is None else [stored_kg, extraction_j]
    gauge = SensorReading(sensor="tank", values=values, unit="kg", resource_species="water")
    first = make_observation(0, 0.0, "plant", battery_soc_j=soc[0])
    last = make_observation(1, 10.0, "plant", battery_soc_j=soc[1]).model_copy(
        update={"sensors": (gauge,)}
    )
    return make_trace([first, last])


def test_energy_per_kg_includes_the_extraction_bus_not_just_the_battery() -> None:
    # Sim tracks ISRU extraction on a dedicated bus that never touches the survival battery, so
    # channel 1 of the storage gauge was emitted every tick and read by no metric — the swarm's
    # extraction cost was free. It is now part of the numerator (astro-mine-sim#64).
    trace = _gauge_trace(stored_kg=2.0, extraction_j=500.0, soc=(1000.0, 900.0))
    value = EnergyPerKg().compute(trace).value
    assert value is not None
    assert value == pytest.approx((100.0 + 500.0) / 2.0)


def test_energy_per_kg_is_unchanged_for_a_single_channel_gauge() -> None:
    # A gauge declaring no extraction energy contributes none, so the fixture path scores exactly
    # as it did — this is a repair, not a redefinition.
    trace = _gauge_trace(stored_kg=2.0, extraction_j=None, soc=(1000.0, 900.0))
    value = EnergyPerKg().compute(trace).value
    assert value is not None
    assert value == pytest.approx(100.0 / 2.0)


def test_energy_per_kg_is_still_not_applicable_without_water() -> None:
    trace = _gauge_trace(stored_kg=0.0, extraction_j=500.0, soc=(1000.0, 900.0))
    assert EnergyPerKg().compute(trace).value is None
