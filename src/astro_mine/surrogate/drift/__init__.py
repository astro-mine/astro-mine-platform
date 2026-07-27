"""Drift: OOD/drift monitoring and hybrid re-validation triggers (RM-P1-SURR-04).

The inline **use** loop's watchdog (surrogate.md §3, §10): a :class:`DriftMonitor` accumulates the
per-query trust-region and uncertainty signals of live
:class:`~astro_mine.surrogate.model.Prediction`s and, per a hybrid
:class:`RevalidationPolicy` (schedule + drift), raises a :class:`RevalidationTrigger` that schedules
a ground-truth re-validation and active resampling. Pure numpy — it watches predictions from the
torch surrogate or the served ONNX tier alike, and needs neither the ``[serve]`` nor ``[publish]``
extra.
"""

from __future__ import annotations

from astro_mine.surrogate.drift.events import (
    DriftEventSink,
    DriftReason,
    RevalidationTrigger,
)
from astro_mine.surrogate.drift.monitor import DriftMonitor
from astro_mine.surrogate.drift.policy import RevalidationPolicy

__all__ = [
    "DriftEventSink",
    "DriftMonitor",
    "DriftReason",
    "RevalidationPolicy",
    "RevalidationTrigger",
]
