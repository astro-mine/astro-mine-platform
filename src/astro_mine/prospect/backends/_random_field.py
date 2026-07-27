"""A seeded, spatially-correlated standard-normal field — the latent a realization is drawn from.

A realization of a resource field is not per-cell independent noise: ice is *spatially structured*,
and a ground truth without that structure is a strictly easier (and dishonest) prospecting problem
than the one the belief backends model. This module supplies the primitive that structure is built
from: a ``(n_rows, n_cols)`` draw ``z`` that is **standard normal at every cell** (zero mean, unit
variance — so it can be scaled onto any prior's per-cell mean/variance without changing the
marginals) but **correlated between cells** over a given correlation length.

It is Gaussian-smoothed white noise, renormalized: white noise convolved with a kernel is a
zero-mean Gaussian field whose covariance is the kernel's autocorrelation, and dividing by the
resulting (analytically known) standard deviation restores unit marginal variance. Deterministic
under ``seed`` (conventions.md §1.5), pure NumPy, and O(N * k) rather than the O(N**3) of a dense
Cholesky — so it scales to the lattices the GMRF backend targets.

**What ``correlation_length_m`` means here.** It is the **practical range** — the separation at
which the correlation has decayed to about 0.1 — which is the same thing the GMRF/SPDE backend means
by it (the Matern/INLA convention: ``rho = sqrt(8 nu)/kappa``, correlation ~= 0.14 at ``h = rho``).
It is *not* the smoothing kernel's standard deviation; the kernel std is derived from it
(:data:`_PRACTICAL_RANGE_PER_SIGMA`). Both realization backends therefore decay over the length the
caller asked for. They are **not** interchangeable beyond that: a Gaussian kernel yields an
infinitely smooth field and a Matern nu=1 one does not, so at short lags the Gaussian field is the
more strongly correlated of the two (lag-1 ~= 0.77 against the GMRF's ~= 0.60 at a 3-cell range).
Same range, genuinely different covariance shapes — the choice of backend is a modelling choice.

**Why the white noise is drawn on a larger domain and cropped.** The normalizer ``sqrt(sum k**2)``
is exact only when each output cell is a weighted sum of *distinct, independent* noise samples.
Any padding scheme that manufactures the border out of the field's own interior (reflect,
symmetric, wrap) reuses samples, so the summands are correlated, the true variance exceeds
``sum k**2``, and the draw comes out systematically **over-dispersed** — badly so when the kernel
is a large fraction of the domain. That would quietly break the one property this primitive
exists to guarantee, and with it the sealed truth's "same marginals as the prior" contract. So
the noise is drawn on a domain extended by the kernel's half-width and convolved in ``valid``
mode: every output cell then sums distinct samples, the normalizer is exact, and the result is
stationary right up to the edge — no border artifacts to explain away.

Backlog: RM-P1-PROSPECT-10 — https://github.com/astro-mine/astro-mine-prospect/issues/30
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

__all__ = ["correlated_standard_normal"]

#: The kernel is truncated at this many standard deviations; beyond it a Gaussian kernel carries
#: less than 0.02% of its mass, so the truncation is numerically irrelevant and keeps the
#: convolution O(N * k) with a small k.
_TRUNCATION = 3.0

#: ``practical range / kernel sigma``. Smoothing white noise with a Gaussian kernel of std ``s``
#: gives a field with autocorrelation ``exp(-h**2 / (4 s**2))``, which falls to 0.1 at
#: ``h = 2 s sqrt(ln 10) ~= 3.035 s``. Since ``correlation_length_m`` is the practical range (the
#: convention shared with the GMRF backend — see the module docstring), the kernel std the smoother
#: must actually use is ``correlation_length_m / _PRACTICAL_RANGE_PER_SIGMA``. Reading the requested
#: range *as* the kernel std — as this module used to — produces a field correlated over roughly
#: three times the length the caller asked for.
_PRACTICAL_RANGE_PER_SIGMA = 2.0 * math.sqrt(math.log(10.0))  # ~= 3.0349


def correlated_standard_normal(
    n_rows: int,
    n_cols: int,
    *,
    dx_m: float,
    dy_m: float,
    correlation_length_m: float,
    seed: int,
) -> NDArray[np.float64]:
    """A seeded ``(n_rows, n_cols)`` standard-normal field correlated over ``correlation_length_m``.

    ``correlation_length_m`` is the **practical range**: correlation decays to ~0.1 at that
    separation (the convention the GMRF backend shares — see the module docstring), so the kernel is
    built with std ``correlation_length_m / _PRACTICAL_RANGE_PER_SIGMA``.

    Every cell is marginally ``N(0, 1)``; cells within the correlation length are positively
    correlated. Scaling it by a prior's per-cell standard deviation and adding the prior mean
    therefore yields a realization that **preserves the prior's marginals exactly** while carrying
    spatial structure — precisely what a sealed ground truth must do, since a truth whose marginals
    had drifted from its prior would silently invalidate the calibration gate.
    """
    if correlation_length_m <= 0.0:
        raise ValueError(f"correlation_length_m must be positive, got {correlation_length_m}")
    rng = np.random.default_rng(seed)
    kernel = _gaussian_kernel(dx_m=dx_m, dy_m=dy_m, correlation_length_m=correlation_length_m)
    if kernel.size == 1:  # a correlation length far below one cell: white noise already is the draw
        return np.asarray(rng.standard_normal((n_rows, n_cols)), dtype=np.float64)

    # Draw the noise on a domain extended by the kernel's half-width, then convolve in `valid`
    # mode: every output cell is a weighted sum of *distinct* iid samples, so Var[sum_i k_i w_i]
    # is exactly sum_i k_i**2 and the normalizer below is exact (see the module docstring on why
    # padding the field from its own interior would instead over-disperse the draw).
    half_y, half_x = (kernel.shape[0] - 1) // 2, (kernel.shape[1] - 1) // 2
    white = rng.standard_normal((n_rows + 2 * half_y, n_cols + 2 * half_x))
    windows = np.lib.stride_tricks.sliding_window_view(white, kernel.shape)
    smoothed = np.einsum("rcij,ij->rc", windows, kernel)
    return np.asarray(smoothed / float(np.sqrt(np.sum(kernel * kernel))), dtype=np.float64)


def _gaussian_kernel(
    *, dx_m: float, dy_m: float, correlation_length_m: float
) -> NDArray[np.float64]:
    """The Gaussian smoothing kernel on the cell lattice, truncated at :data:`_TRUNCATION` sigmas.

    ``correlation_length_m`` is the **practical range**, so the kernel's standard deviation is
    ``correlation_length_m / _PRACTICAL_RANGE_PER_SIGMA`` — the field then decays to ~0.1
    correlation at exactly the requested length, as the GMRF backend's does. Expressed in *cells*,
    so the correlation length is honoured in metres on a non-square grid.
    """
    sigma_m = correlation_length_m / _PRACTICAL_RANGE_PER_SIGMA
    sx = sigma_m / dx_m
    sy = sigma_m / dy_m
    half_x = int(np.ceil(_TRUNCATION * sx))
    half_y = int(np.ceil(_TRUNCATION * sy))
    ix = np.arange(-half_x, half_x + 1, dtype=np.float64)
    iy = np.arange(-half_y, half_y + 1, dtype=np.float64)
    gx = np.exp(-0.5 * (ix / sx) ** 2)
    gy = np.exp(-0.5 * (iy / sy) ** 2)
    return np.asarray(np.outer(gy, gx), dtype=np.float64)
