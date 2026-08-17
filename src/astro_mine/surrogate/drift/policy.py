# SPDX-License-Identifier: Apache-2.0
"""``RevalidationPolicy`` — the hybrid schedule-or-drift re-validation rule (RM-P1-SURR-04).

surrogate.md §11 recommends a **hybrid** re-validation trigger: a periodic schedule *plus*
drift-/OOD-triggered re-validation. This is that rule as a pure decision over window statistics —
no monitor state, so it is trivially testable and reused across monitors. :meth:`evaluate` returns
the first :class:`~astro_mine.surrogate.drift.events.DriftReason` that fires (schedule first, then
the drift signals in severity order) or ``None`` when the tier is still trusted.
"""

from __future__ import annotations

from dataclasses import dataclass

from astro_mine.surrogate.drift.events import DriftReason

__all__ = ["RevalidationPolicy"]


@dataclass(frozen=True)
class RevalidationPolicy:
    """Thresholds for the hybrid re-validation trigger.

    ``schedule_every`` is the periodic cadence (in live queries); the drift signals are
    ``max_ood_rate`` (out-of-domain fraction over the window), ``min_margin`` (the signed
    trust-region margin floor — a query below it is a hard OOD excursion), and
    ``max_uncertainty_ratio`` (rolling-mean uncertainty over its baseline). Drift signals are only
    evaluated once the window holds ``min_window`` observations, so a cold monitor does not fire on
    one unlucky query.
    """

    schedule_every: int = 128
    max_ood_rate: float = 0.2
    min_margin: float = 0.0
    max_uncertainty_ratio: float = 2.0
    min_window: int = 8

    def __post_init__(self) -> None:
        if self.schedule_every < 1:
            raise ValueError("schedule_every must be >= 1")
        if not 0.0 <= self.max_ood_rate <= 1.0:
            raise ValueError("max_ood_rate must be in [0, 1]")
        if self.max_uncertainty_ratio <= 0.0:
            raise ValueError("max_uncertainty_ratio must be > 0")
        if self.min_window < 1:
            raise ValueError("min_window must be >= 1")

    def evaluate(
        self,
        *,
        query_index: int,
        window_count: int,
        ood_rate: float,
        worst_margin: float,
        mean_uncertainty: float,
        baseline_uncertainty: float,
    ) -> DriftReason | None:
        """The re-validation reason for the current window, or ``None`` if the tier is trusted.

        ``query_index`` is the 1-based live-query count (drives the schedule); the rest are the
        monitor's current window statistics. Schedule fires first (a due cadence re-validates even
        with no drift); then the drift signals, guarded by ``min_window``.
        """
        if query_index > 0 and query_index % self.schedule_every == 0:
            return DriftReason.SCHEDULED
        if window_count < self.min_window:
            return None
        if ood_rate > self.max_ood_rate:
            return DriftReason.OOD_RATE
        if worst_margin < self.min_margin:
            return DriftReason.MARGIN_BREACH
        if (
            baseline_uncertainty > 0.0
            and mean_uncertainty > self.max_uncertainty_ratio * baseline_uncertainty
        ):
            return DriftReason.UNCERTAINTY_DRIFT
        return None
