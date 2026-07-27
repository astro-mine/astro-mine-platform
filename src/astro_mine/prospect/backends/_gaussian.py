"""Gaussian posterior summaries shared by the resource-field backends (RM-P0-PROSPECT-02).

Both the grid and GP backends model the per-point posterior as a univariate Gaussian — a mean
paired with a variance. This module turns that ``(mean, variance)`` into the **calibrated,
ordered quantiles** and the **seeded, reproducible samples** the Core ResourceField contract
requires (prospect.md §2.1; conventions.md §1.5). Centralizing it keeps the two backends'
uncertainty reporting identical and honest — a backend swap never changes the *shape* of the
uncertainty, only the inference behind it.

Backlog: RM-P0-PROSPECT-02 — https://github.com/astro-mine/astro-mine-prospect/issues/2
"""

from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np

__all__ = ["gaussian_quantile", "gaussian_samples"]


def _std(variance: float) -> float:
    if variance < 0.0:
        raise ValueError(f"variance must be non-negative, got {variance}")
    return math.sqrt(variance)


def gaussian_quantile(mean: float, variance: float, q: float) -> float:
    """The value at quantile level ``q`` (in ``[0, 1]``) of ``N(mean, variance)``.

    Monotone in ``q`` by construction (the normal inverse-CDF), so reported quantiles are
    ordered and consistent with the mean (``q=0.5`` returns the mean exactly). A degenerate
    (zero-variance) field returns the mean for every ``q``; the ``q in {0, 1}`` endpoints of a
    non-degenerate Gaussian are ``∓inf`` (its support is unbounded).
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile level q must be in [0, 1], got {q}")
    std = _std(variance)
    if std == 0.0 or q == 0.5:
        return mean
    if q == 0.0:
        return -math.inf
    if q == 1.0:
        return math.inf
    return mean + NormalDist().inv_cdf(q) * std


def gaussian_samples(
    mean: float, variance: float, *, n: int, seed: int | None
) -> tuple[float, ...]:
    """Draw ``n`` samples from ``N(mean, variance)`` — a Monte-Carlo realization.

    Deterministic given ``seed`` (conventions.md §1.5): the same seed reproduces the same
    draws; ``seed=None`` is non-deterministic. A zero-variance field returns the mean repeated;
    ``n=0`` returns an empty tuple.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if n == 0:
        return ()
    std = _std(variance)
    if std == 0.0:
        return tuple(mean for _ in range(n))
    rng = np.random.default_rng(seed)
    return tuple(float(x) for x in rng.normal(loc=mean, scale=std, size=n))
