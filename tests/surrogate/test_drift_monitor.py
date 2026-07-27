"""Drift/OOD monitoring + hybrid re-validation triggers (RM-P1-SURR-04; surrogate.md §10).

A drift/OOD monitor raises a re-validation trigger when live queries leave the trust region (the
acceptance criterion), and the hybrid policy also fires on a periodic schedule.
"""

from __future__ import annotations

import numpy as np
import pytest

from astro_mine.surrogate.drift import (
    DriftMonitor,
    DriftReason,
    RevalidationPolicy,
    RevalidationTrigger,
)
from astro_mine.surrogate.model import Prediction


def _prediction(*, in_domain: bool, margin: float, uncertainty: float) -> Prediction:
    return Prediction(
        channels={},
        uncertainty={},
        in_domain=in_domain,
        ood_margin=margin,
        fields={},
        field_uncertainty={"position": np.full((4, 2), uncertainty)},
    )


class _ListSink:
    def __init__(self) -> None:
        self.events: list[RevalidationTrigger] = []

    def publish(self, trigger: RevalidationTrigger) -> None:
        self.events.append(trigger)


def test_healthy_in_domain_queries_raise_no_trigger() -> None:
    monitor = DriftMonitor(
        policy=RevalidationPolicy(schedule_every=1000, min_window=4), baseline_uncertainty=0.1
    )
    triggers = [
        monitor.observe(_prediction(in_domain=True, margin=0.4, uncertainty=0.05))
        for _ in range(20)
    ]
    assert all(trigger is None for trigger in triggers)
    assert monitor.ood_rate == 0.0


def test_out_of_domain_queries_raise_a_revalidation_trigger() -> None:
    sink = _ListSink()
    monitor = DriftMonitor(
        policy=RevalidationPolicy(
            schedule_every=1000, max_ood_rate=0.5, min_margin=0.0, min_window=4
        ),
        sink=sink,
        window=16,
    )
    for _ in range(4):
        monitor.observe(_prediction(in_domain=True, margin=0.3, uncertainty=0.05))
    trigger = None
    for _ in range(12):
        trigger = monitor.observe(_prediction(in_domain=False, margin=-3.0, uncertainty=0.2))
        if trigger is not None:
            break
    assert trigger is not None
    assert trigger.reason in (DriftReason.OOD_RATE, DriftReason.MARGIN_BREACH)
    assert sink.events and sink.events[-1] is trigger


def test_margin_breach_fires_on_a_hard_ood_excursion() -> None:
    # With the OOD-rate signal disabled (max_ood_rate=1.0), a single query whose signed margin is
    # far below the floor trips a margin breach on its own.
    monitor = DriftMonitor(
        policy=RevalidationPolicy(
            schedule_every=1000, max_ood_rate=1.0, min_margin=0.0, min_window=1
        ),
        window=8,
    )
    trigger = monitor.observe(_prediction(in_domain=False, margin=-5.0, uncertainty=0.1))
    assert trigger is not None and trigger.reason == DriftReason.MARGIN_BREACH
    assert trigger.worst_margin == -5.0


def test_uncertainty_drift_above_baseline_multiple_triggers() -> None:
    monitor = DriftMonitor(
        policy=RevalidationPolicy(
            schedule_every=1000,
            max_ood_rate=1.0,
            min_margin=-1e9,
            max_uncertainty_ratio=2.0,
            min_window=4,
        ),
        baseline_uncertainty=0.1,
        window=8,
    )
    # In-domain, healthy margin, but uncertainty is 3x the baseline -> drift.
    trigger = None
    for _ in range(5):
        trigger = monitor.observe(_prediction(in_domain=True, margin=0.5, uncertainty=0.3))
    assert trigger is not None and trigger.reason == DriftReason.UNCERTAINTY_DRIFT


def test_scheduled_trigger_fires_on_cadence() -> None:
    monitor = DriftMonitor(policy=RevalidationPolicy(schedule_every=5, min_window=1000), window=32)
    triggers = [
        monitor.observe(_prediction(in_domain=True, margin=0.5, uncertainty=0.05)) for _ in range(5)
    ]
    assert triggers[-1] is not None and triggers[-1].reason == DriftReason.SCHEDULED
    assert triggers[-1].query_index == 5


def test_monitor_drives_the_served_tier_predictions(served_bundle, served_query, ood_query) -> None:
    """Integration: the monitor consumes real served predictions and trips on an OOD query."""
    from astro_mine.surrogate.serve import OnnxServedSurrogate

    served = OnnxServedSurrogate(served_bundle)
    monitor = DriftMonitor(
        policy=RevalidationPolicy(
            schedule_every=1000, max_ood_rate=0.3, min_margin=0.0, min_window=2
        )
    )
    for _ in range(4):
        assert monitor.observe(served.predict(served_query)) is None
    trigger = None
    for _ in range(6):
        trigger = monitor.observe(served.predict(ood_query))
        if trigger is not None:
            break
    assert trigger is not None


def test_policy_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="schedule_every"):
        RevalidationPolicy(schedule_every=0)
    with pytest.raises(ValueError, match="max_ood_rate"):
        RevalidationPolicy(max_ood_rate=1.5)
    with pytest.raises(ValueError, match="max_uncertainty_ratio"):
        RevalidationPolicy(max_uncertainty_ratio=0.0)
