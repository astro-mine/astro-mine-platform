# SPDX-License-Identifier: Apache-2.0
"""Split-conformal calibration — distribution-free per-channel bounds (RM-P1-SURR-02).

The deep ensemble gives a *heuristic* spread; split conformal turns it into a bound with a
finite-sample **marginal coverage guarantee** (surrogate.md §11: "deep ensembles + conformal").
On a held-out calibration set we take the normalized-residual quantile per output channel; at
query time the calibrated interval half-width is ``q_channel * ensemble_std`` — so a channel's
90% interval contains truth ~90% of the time (the §10 coverage gate). numpy only; no torch.

Caveat (surrogate.md §11 open question): conformal gives *marginal* coverage; conditional,
long-horizon rollout coverage is unresolved — the per-step bounds here are marginal.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["ConformalCalibration", "calibrate_conformal"]

FloatArray = npt.NDArray[np.float64]

_EPS = 1e-9


@dataclass(frozen=True)
class ConformalCalibration:
    """Per-channel conformal multipliers ``q`` and scale ``floor`` — ``q*(std+floor)`` is the bound.

    The ``floor`` (a per-channel scale from the calibration residuals) keeps the interval
    non-degenerate when the ensemble spread is tiny or zero: coverage is then guaranteed by a
    calibrated *constant* interval, while an informative ensemble ``std`` still varies it per
    particle. This makes the bound robust to a small ensemble.
    """

    channel_names: tuple[str, ...]
    quantiles: FloatArray  # (C,) one multiplier per output channel
    floor: FloatArray  # (C,) per-channel scale floor
    nominal_coverage: float

    def half_widths(self, std: FloatArray) -> FloatArray:
        """Calibrated interval half-widths ``q * (std + floor)`` for per-channel ``std``."""
        return np.asarray((std + self.floor) * self.quantiles, dtype=np.float64)


def calibrate_conformal(
    residuals: FloatArray,
    std: FloatArray,
    channel_names: tuple[str, ...],
    *,
    nominal_coverage: float,
    floor_beta: float = 1.0,
) -> ConformalCalibration:
    """Calibrate per-channel conformal multipliers from held-out residuals and ensemble std.

    ``residuals`` and ``std`` are ``(M, C)`` — the absolute error and ensemble standard deviation
    of ``M`` held-out predictions over ``C`` channels. Each channel's scale ``floor`` is
    ``floor_beta`` times its mean absolute residual (so the interval never collapses); the
    nonconformity score is ``residual / (std + floor)``, and its finite-sample
    ``(1 - alpha)(1 + 1/M)`` quantile is the multiplier — giving marginal coverage
    ``>= nominal_coverage`` per channel.
    """
    floor = floor_beta * residuals.mean(axis=0) + _EPS  # (C,) per-channel scale floor
    scores = residuals / (std + floor)  # (M, C) normalized nonconformity
    m = scores.shape[0]
    level = min((np.ceil((m + 1) * nominal_coverage) / m), 1.0)
    quantiles = np.quantile(scores, level, axis=0, method="higher")
    return ConformalCalibration(
        channel_names=channel_names,
        quantiles=np.asarray(quantiles, dtype=np.float64),
        floor=np.asarray(floor, dtype=np.float64),
        nominal_coverage=nominal_coverage,
    )
