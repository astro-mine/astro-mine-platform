# SPDX-License-Identifier: Apache-2.0
"""``DriftMonitor`` — OOD/drift monitoring of live surrogate queries (RM-P1-SURR-04).

The inline-loop watchdog of surrogate.md §3/§10: it accumulates the per-query signals a
:class:`~astro_mine.surrogate.model.Prediction` already carries — the ``in_domain`` flag, the signed
``ood_margin`` a drift monitor is documented to accumulate (model.py), and a rolling mean of the
calibrated uncertainty — over a sliding window, and applies a
:class:`~astro_mine.surrogate.drift.policy.RevalidationPolicy` after each query. When the policy
fires, the monitor publishes a :class:`~astro_mine.surrogate.drift.events.RevalidationTrigger` to
its sink and returns it, so live queries leaving the trust region schedule a ground-truth
re-validation and resample (the acceptance criterion).

numpy only — no torch, no onnxruntime; the monitor watches predictions regardless of how they were
produced (the torch surrogate or the served ONNX tier).
"""

from __future__ import annotations

from collections import deque

import numpy as np

from astro_mine.surrogate.drift.events import DriftEventSink, RevalidationTrigger
from astro_mine.surrogate.drift.policy import RevalidationPolicy
from astro_mine.surrogate.model import Prediction

__all__ = ["DriftMonitor"]


def _representative_uncertainty(prediction: Prediction) -> float:
    """A single calibrated-uncertainty scalar per query — mean over channels and fields."""
    values = [float(v) for v in prediction.uncertainty.values()]
    for array in prediction.field_uncertainty.values():
        if np.size(array):
            values.append(float(np.mean(array)))
    return max(values) if values else 0.0


class DriftMonitor:
    """Accumulate live-query drift signals and raise re-validation triggers per a policy.

    ``baseline_uncertainty`` anchors the uncertainty-drift signal (e.g. the mean per-channel RMSE
    from the surrogate's :class:`~astro_mine.surrogate.report.ErrorReport`); leave it ``0`` to
    disable that signal. ``window`` bounds the sliding statistics; an optional ``sink`` receives
    every :class:`RevalidationTrigger`. :meth:`observe` is the one call the inline loop makes per
    query.
    """

    def __init__(
        self,
        *,
        policy: RevalidationPolicy | None = None,
        sink: DriftEventSink | None = None,
        window: int = 32,
        baseline_uncertainty: float = 0.0,
    ) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        self._policy = policy if policy is not None else RevalidationPolicy()
        self._sink = sink
        self._baseline = baseline_uncertainty
        self._in_domain: deque[bool] = deque(maxlen=window)
        self._margins: deque[float] = deque(maxlen=window)
        self._uncertainty: deque[float] = deque(maxlen=window)
        self._count = 0

    def observe(self, prediction: Prediction) -> RevalidationTrigger | None:
        """Record one live prediction; return (and publish) a re-validation trigger if one fires."""
        self._count += 1
        self._in_domain.append(prediction.in_domain)
        if prediction.ood_margin is not None:
            self._margins.append(prediction.ood_margin)
        self._uncertainty.append(_representative_uncertainty(prediction))
        reason = self._policy.evaluate(
            query_index=self._count,
            window_count=len(self._in_domain),
            ood_rate=self.ood_rate,
            worst_margin=self.worst_margin,
            mean_uncertainty=self.mean_uncertainty,
            baseline_uncertainty=self._baseline,
        )
        if reason is None:
            return None
        trigger = RevalidationTrigger(
            reason=reason,
            query_index=self._count,
            window_ood_rate=self.ood_rate,
            worst_margin=self.worst_margin,
            mean_uncertainty=self.mean_uncertainty,
        )
        if self._sink is not None:
            self._sink.publish(trigger)
        return trigger

    @property
    def count(self) -> int:
        """Total live queries observed."""
        return self._count

    @property
    def ood_rate(self) -> float:
        """Fraction of the window whose queries fell outside the trust region."""
        if not self._in_domain:
            return 0.0
        return sum(not ok for ok in self._in_domain) / len(self._in_domain)

    @property
    def worst_margin(self) -> float:
        """The most negative signed trust-region margin in the window (0 if none reported)."""
        return min(self._margins) if self._margins else 0.0

    @property
    def mean_uncertainty(self) -> float:
        """The rolling mean calibrated uncertainty over the window."""
        return float(np.mean(self._uncertainty)) if self._uncertainty else 0.0
