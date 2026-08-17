# SPDX-License-Identifier: Apache-2.0
"""Drift events — the re-validation trigger a monitor publishes (RM-P1-SURR-04; surrogate.md §10).

When live queries leave the trust region or the surrogate's uncertainty drifts, the monitor raises
a :class:`RevalidationTrigger` — the event that schedules a ground-truth re-validation and active
resampling (surrogate.md §6: "drift events publish to NATS so re-validation can be triggered
automatically"). Here the sink is an in-process :class:`DriftEventSink` protocol — the transport-
agnostic seam; the NATS/JetStream wiring is the Cloud deployment (deferred, surrogate.md §6), and a
consumer supplies whatever sink it wants (a list, a log, a queue publisher).

The trigger carries the window statistics that fired it so the re-validation step is actionable
without re-deriving them, and is a frozen value object (an audit record, surrogate.md §5).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

__all__ = ["DriftEventSink", "DriftReason", "RevalidationTrigger"]


class DriftReason(StrEnum):
    """Why a re-validation fired — the hybrid schedule/drift causes (surrogate.md §11)."""

    #: The periodic re-validation cadence elapsed (the schedule half of the hybrid trigger).
    SCHEDULED = "scheduled"
    #: Too many recent queries fell outside the trust region.
    OOD_RATE = "ood_rate"
    #: A query's signed trust-region margin fell below the floor (a hard OOD excursion).
    MARGIN_BREACH = "margin_breach"
    #: The rolling mean calibrated uncertainty drifted above its baseline multiple.
    UNCERTAINTY_DRIFT = "uncertainty_drift"


@dataclass(frozen=True, slots=True)
class RevalidationTrigger:
    """A request to re-validate the surrogate against the high-fidelity oracle and resample.

    ``reason`` is the cause; ``query_index`` is the live-query count at which it fired; the
    remaining fields snapshot the window statistics (out-of-domain rate, worst signed margin,
    rolling mean uncertainty) that justified it — so the re-validation/resampling step
    (RM-P1-SURR-03 ``datagen``) is actionable from the event alone.
    """

    reason: DriftReason
    query_index: int
    window_ood_rate: float
    worst_margin: float
    mean_uncertainty: float


class DriftEventSink(Protocol):
    """Where a monitor publishes its :class:`RevalidationTrigger`s (in-process; NATS deferred)."""

    def publish(self, trigger: RevalidationTrigger) -> None:
        """Handle one re-validation trigger (append, log, or forward to a message bus)."""
        ...
