"""Geostatistical sanity — variogram recovery and leave-one-out kriging CV (prospect.md §10).

prospect.md §10 names two validation strategies that are neither calibration-coverage nor
cross-backend agreement, and both ask whether the inference is geostatistically sound at all:

- **variogram / length-scale recovery** — "recovered variograms/length-scales match the generating
  model on synthetic data". :func:`empirical_variogram` estimates the semivariance of a scattered
  sample and :func:`fit_variogram` fits a Gaussian model to it, recovering the ``sill``, the
  ``nugget``, and — the load-bearing one — the **correlation length**. Run against a field drawn
  from a known length scale, it answers the question a resource field lives or dies on: does this
  model think ice is as spatially clumpy as it actually is? A model that recovers the wrong length
  scale will send the swarm to the wrong place next, however well-calibrated its variance is.
- **leave-one-out kriging cross-validation** — "kriging cross-validation (leave-one-out) error
  within bounds". :func:`loo_cross_validation` refits the field with each observation held out in
  turn and scores the prediction at the held-out point. That is the honest out-of-sample error of
  the inference path itself, on data it has never seen — which coverage on a single fit cannot tell
  you.

Both are pure NumPy over a scattered sample, so they hold any backend to account through the Core
contract alone, without knowing what it is.

Backlog: prospect.md §10 — https://github.com/astro-mine/astro-mine-prospect/issues/33
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astro_mine.core.resource import Position, ResourceField

__all__ = [
    "LooReport",
    "VariogramFit",
    "empirical_variogram",
    "fit_variogram",
    "loo_cross_validation",
]

#: The grid-search resolution over the correlation length in :func:`fit_variogram` (the only
#: nonlinear parameter; the rest are solved in closed form at each candidate).
_LENGTH_CANDIDATES = 256


@dataclass(frozen=True)
class VariogramFit:
    """A fitted Gaussian variogram: ``gamma(h) = nugget + sill * (1 - exp(-h**2 / L**2))``.

    :attr:`correlation_length_m` is ``L`` — the range over which values stay correlated, and the
    parameter a prospecting model must get right. :attr:`sill` is the variance the semivariance
    saturates at (the field's total variance) and :attr:`nugget` the intercept as ``h -> 0``
    (measurement noise plus sub-cell variability). :attr:`rmse` is the fit residual against the
    empirical points.
    """

    correlation_length_m: float
    sill: float
    nugget: float
    rmse: float

    def gamma(self, lags_m: NDArray[np.float64]) -> NDArray[np.float64]:
        """The fitted semivariance at ``lags_m``."""
        h = np.asarray(lags_m, dtype=np.float64)
        scaled = 1.0 - np.exp(-(h**2) / (self.correlation_length_m**2))
        return np.asarray(self.nugget + self.sill * scaled, dtype=np.float64)


@dataclass(frozen=True)
class LooReport:
    """The leave-one-out kriging cross-validation result over a scattered sample.

    :attr:`rmse` is the out-of-sample root-mean-square prediction error and :attr:`mae` its robust
    counterpart. :attr:`standardized_rmse` divides each error by the field's *predicted* standard
    deviation before averaging, so it scores the **uncertainty**, not just the mean: a
    well-calibrated field predicts its own errors, and a standardized RMSE near 1 is what that looks
    like. Far above 1 is over-confidence — the stated variance is too small for the errors actually
    made; far below 1 is under-confidence.
    """

    rmse: float
    mae: float
    standardized_rmse: float
    n: int


def empirical_variogram(
    points: Sequence[Position],
    values: Sequence[float],
    *,
    n_lags: int = 12,
    max_lag_m: float | None = None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.intp]]:
    """The binned empirical semivariogram of a scattered sample.

    Returns ``(lag_centres, gamma, counts)``: for every pair of samples, half the squared difference
    of their values (the semivariance), binned by their separation. Pairs beyond ``max_lag_m``
    (default: half the sample's diameter, beyond which the bins are too sparse to be informative)
    are dropped, as are empty bins — so the arrays returned are always usable by
    :func:`fit_variogram`.
    """
    xy = np.asarray([(p[0], p[1]) for p in points], dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if xy.shape[0] != y.shape[0]:
        raise ValueError(f"points/values length mismatch: {xy.shape[0]} vs {y.shape[0]}")
    if xy.shape[0] < 3:
        raise ValueError("an empirical variogram needs at least 3 samples")

    i, j = np.triu_indices(xy.shape[0], k=1)
    separation = np.linalg.norm(xy[i] - xy[j], axis=1)
    semivariance = 0.5 * (y[i] - y[j]) ** 2
    limit = 0.5 * float(separation.max()) if max_lag_m is None else float(max_lag_m)
    if limit <= 0.0:
        raise ValueError(f"max_lag_m must be positive, got {limit}")

    keep = separation <= limit
    edges = np.linspace(0.0, limit, n_lags + 1)
    which = np.digitize(separation[keep], edges[1:-1])
    counts = np.bincount(which, minlength=n_lags).astype(np.intp)
    totals = np.bincount(which, weights=semivariance[keep], minlength=n_lags)
    centres = 0.5 * (edges[:-1] + edges[1:])

    populated = counts > 0
    gamma = np.asarray(totals[populated] / counts[populated], dtype=np.float64)
    return np.asarray(centres[populated], dtype=np.float64), gamma, counts[populated]


def fit_variogram(
    lags_m: NDArray[np.float64],
    gamma: NDArray[np.float64],
    *,
    counts: NDArray[np.intp] | None = None,
) -> VariogramFit:
    """Fit the Gaussian variogram model to an empirical variogram — recovering the length scale.

    A grid search over the correlation length (the only nonlinear parameter) with the ``sill`` and
    ``nugget`` solved in closed form by weighted least squares at each candidate, weighting each lag
    bin by its pair ``counts`` — an under-populated bin is noisy and should not steer the fit.
    Simple and derivative-free: this is a *validation* tool, and its failure mode should be
    "obviously wrong", not "subtly converged somewhere else".
    """
    h = np.asarray(lags_m, dtype=np.float64)
    g = np.asarray(gamma, dtype=np.float64)
    if h.size < 3:
        raise ValueError("fitting a variogram needs at least 3 populated lag bins")
    w = np.ones_like(h) if counts is None else np.asarray(counts, dtype=np.float64)

    span = float(h.max())
    best: VariogramFit | None = None
    for length in np.linspace(0.05 * span, 2.0 * span, _LENGTH_CANDIDATES):
        # gamma(h) == nugget * 1 + sill * basis(h) is linear in (nugget, sill) at a fixed length.
        basis = 1.0 - np.exp(-(h**2) / (length**2))
        design = np.stack([np.ones_like(h), basis], axis=1)
        sqrt_w = np.sqrt(w)[:, None]
        solution, *_ = np.linalg.lstsq(design * sqrt_w, g * sqrt_w.ravel(), rcond=None)
        nugget, sill = float(solution[0]), float(solution[1])
        if sill <= 0.0:  # a non-increasing variogram is not a spatial-correlation model
            continue
        residual = design @ solution - g
        rmse = float(np.sqrt(np.average(residual**2, weights=w)))
        if best is None or rmse < best.rmse:
            best = VariogramFit(
                correlation_length_m=float(length),
                sill=sill,
                nugget=max(nugget, 0.0),
                rmse=rmse,
            )
    if best is None:
        raise ValueError(
            "no increasing Gaussian variogram fits this sample — it shows no spatial correlation"
        )
    return best


def loo_cross_validation(
    points: Sequence[Position],
    values: Sequence[float],
    fit: Callable[[Sequence[Position], Sequence[float]], ResourceField],
) -> LooReport:
    """Leave-one-out kriging cross-validation of the field that ``fit`` builds from a sample.

    Refits the field ``n`` times, each time holding one sample out, and scores the prediction at the
    held-out location against its true value. ``fit`` is any ``(points, values) -> ResourceField``
    builder — the grid, GP, and GMRF backends all satisfy it — so a backend is scored purely through
    the Core contract, on data it has genuinely not seen.
    """
    n = len(points)
    if n != len(values):
        raise ValueError(f"points/values length mismatch: {n} vs {len(values)}")
    if n < 3:
        raise ValueError("leave-one-out cross-validation needs at least 3 samples")

    errors = np.empty(n, dtype=np.float64)
    standardized = np.empty(n, dtype=np.float64)
    for k in range(n):
        train_points = [p for i, p in enumerate(points) if i != k]
        train_values = [v for i, v in enumerate(values) if i != k]
        field = fit(train_points, train_values)
        held_out = points[k]
        errors[k] = field.mean(held_out) - values[k]
        sigma = float(np.sqrt(max(field.variance(held_out), 1e-12)))
        standardized[k] = errors[k] / sigma
    return LooReport(
        rmse=float(np.sqrt(np.mean(errors**2))),
        mae=float(np.mean(np.abs(errors))),
        standardized_rmse=float(np.sqrt(np.mean(standardized**2))),
        n=n,
    )
