"""GridField — the simple, dependency-light reference backend (prospect.md §11).

A regular 2-D grid over the field's :class:`~astro_mine.prospect.field.FieldGrid` domain holding
a per-cell Gaussian posterior (``mean`` + ``variance`` arrays). A query bilinearly interpolates
the mean and variance at a position and reports calibrated quantiles / seeded samples through the
shared :mod:`~astro_mine.prospect.backends._gaussian` helper — so it satisfies the Core
ResourceField contract *identically* to the GP backend, a true drop-in alternative
(RM-P0-PROSPECT-02). Conditioning on scattered observations is a transparent Gaussian
measurement-fusion smoother (an RBF-weighted local precision update): where observations are
dense the posterior mean tracks the data and the variance shrinks below the prior; far away it
reverts to the prior. It is the *simple, fast reference* path — a principled GMRF/SPDE field is
deferred to P1, and the ordered observation-log / belief machinery is RM-P0-PROSPECT-04.

Backlog: RM-P0-PROSPECT-02 — https://github.com/astro-mine/astro-mine-prospect/issues/2
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from astro_mine.core.resource import Position
from astro_mine.core.units import Epoch
from astro_mine.prospect.backends._gaussian import gaussian_quantile, gaussian_samples
from astro_mine.prospect.backends._training import validate_training
from astro_mine.prospect.field.base import BaseResourceField
from astro_mine.prospect.field.metadata import FieldGrid, FieldMetadata

__all__ = ["GridField"]


class GridField(BaseResourceField):
    """A gridded Gaussian resource field — per-cell ``mean``/``variance`` with bilinear queries.

    Construct a posterior with :meth:`build` (a constant prior, optionally conditioned on
    observations); the low-level constructor takes the ``(n_rows, n_cols)`` mean/variance arrays
    directly. Queries outside the grid clamp to the nearest edge cell.
    """

    def __init__(
        self,
        metadata: FieldMetadata,
        mean: NDArray[np.float64],
        variance: NDArray[np.float64],
    ) -> None:
        grid = metadata.grid
        if grid is None:
            raise ValueError("GridField requires metadata.grid (a FieldGrid spatial domain)")
        shape = (grid.n_rows, grid.n_cols)
        if mean.shape != shape or variance.shape != shape:
            raise ValueError(
                f"mean/variance arrays must have shape {shape} (n_rows, n_cols); "
                f"got mean {mean.shape}, variance {variance.shape}"
            )
        if bool(np.any(variance < 0.0)):
            raise ValueError("variance grid must be non-negative everywhere")
        super().__init__(metadata)
        self._grid = grid
        self._mean = np.ascontiguousarray(mean, dtype=np.float64)
        self._variance = np.ascontiguousarray(variance, dtype=np.float64)
        self._dx = (grid.max_x_m - grid.min_x_m) / grid.n_cols
        self._dy = (grid.max_y_m - grid.min_y_m) / grid.n_rows

    @classmethod
    def build(
        cls,
        metadata: FieldMetadata,
        *,
        train_points: Sequence[Position] | None = None,
        train_values: Sequence[float] | None = None,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        length_scale: float | None = None,
        noise: float = 1e-4,
    ) -> GridField:
        """Build a grid posterior: a constant prior, optionally fused with observations.

        With no observations the field is the constant ``N(prior_mean, prior_variance)`` over
        the whole grid. Given ``train_points``/``train_values`` it applies an RBF-weighted
        Gaussian precision update with characteristic ``length_scale`` (default: 10% of the
        larger grid extent) and per-observation ``noise`` variance.
        """
        grid = metadata.grid
        if grid is None:
            raise ValueError("GridField requires metadata.grid (a FieldGrid spatial domain)")
        if prior_variance <= 0.0:
            raise ValueError(f"prior_variance must be positive, got {prior_variance}")
        shape = (grid.n_rows, grid.n_cols)
        mean = np.full(shape, float(prior_mean), dtype=np.float64)
        variance = np.full(shape, float(prior_variance), dtype=np.float64)
        xy, y = validate_training(train_points, train_values)
        if y.size > 0:
            mean, variance = _fuse(
                grid,
                xy,
                y,
                prior_mean=float(prior_mean),
                prior_variance=float(prior_variance),
                length_scale=length_scale,
                noise=float(noise),
            )
        return cls(metadata, mean, variance)

    def mean(self, position: Position, *, epoch: Epoch | None = None) -> float:
        return self._interp(self._mean, position)

    def variance(self, position: Position, *, epoch: Epoch | None = None) -> float:
        return self._interp(self._variance, position)

    def mean_grid(self) -> NDArray[np.float64]:
        """The full per-cell posterior mean grid ``(n_rows, n_cols)`` (a mutable copy)."""
        return self._mean.copy()

    def variance_grid(self) -> NDArray[np.float64]:
        """The full per-cell posterior variance grid ``(n_rows, n_cols)`` (a mutable copy)."""
        return self._variance.copy()

    def quantile(self, position: Position, q: float, *, epoch: Epoch | None = None) -> float:
        return gaussian_quantile(self.mean(position), self.variance(position), q)

    def sample(
        self,
        position: Position,
        *,
        n: int = 1,
        seed: int | None = None,
        epoch: Epoch | None = None,
    ) -> tuple[float, ...]:
        return gaussian_samples(self.mean(position), self.variance(position), n=n, seed=seed)

    def _interp(self, arr: NDArray[np.float64], position: Position) -> float:
        """Bilinearly interpolate ``arr`` at ``position`` (cell-centered; edge-clamped)."""
        x, y, _z = position
        fc = (x - self._grid.min_x_m) / self._dx - 0.5
        fr = (y - self._grid.min_y_m) / self._dy - 0.5
        fc = min(max(fc, 0.0), float(self._grid.n_cols - 1))
        fr = min(max(fr, 0.0), float(self._grid.n_rows - 1))
        c0 = int(np.floor(fc))
        r0 = int(np.floor(fr))
        c1 = min(c0 + 1, self._grid.n_cols - 1)
        r1 = min(r0 + 1, self._grid.n_rows - 1)
        tc = fc - c0
        tr = fr - r0
        top = arr[r0, c0] * (1.0 - tc) + arr[r0, c1] * tc
        bot = arr[r1, c0] * (1.0 - tc) + arr[r1, c1] * tc
        return float(top * (1.0 - tr) + bot * tr)


def _fuse(
    grid: FieldGrid,
    xy: NDArray[np.float64],
    y: NDArray[np.float64],
    *,
    prior_mean: float,
    prior_variance: float,
    length_scale: float | None,
    noise: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """RBF-weighted Gaussian precision fusion of observations into per-cell mean/variance."""
    if noise <= 0.0:
        raise ValueError(f"noise must be positive, got {noise}")
    if length_scale is None:
        ls = 0.1 * max(grid.max_x_m - grid.min_x_m, grid.max_y_m - grid.min_y_m)
    else:
        ls = float(length_scale)
    if ls <= 0.0:
        raise ValueError(f"length_scale must be positive, got {ls}")

    _dx = (grid.max_x_m - grid.min_x_m) / grid.n_cols
    _dy = (grid.max_y_m - grid.min_y_m) / grid.n_rows
    xs = grid.min_x_m + (np.arange(grid.n_cols) + 0.5) * _dx
    ys = grid.min_y_m + (np.arange(grid.n_rows) + 0.5) * _dy
    gx, gy = np.meshgrid(xs, ys)  # (n_rows, n_cols)
    centers = np.stack([gx.ravel(), gy.ravel()], axis=1)  # (M, 2)

    diff = centers[:, None, :] - xy[None, :, :]  # (M, K, 2)
    dist2 = np.sum(diff * diff, axis=2)  # (M, K)
    weights = np.exp(-dist2 / (2.0 * ls * ls))  # (M, K)

    prior_precision = 1.0 / prior_variance
    eff_precision = weights.sum(axis=1) / noise  # (M,)
    weighted_obs = (weights * y[None, :]).sum(axis=1) / noise  # (M,)
    post_precision = prior_precision + eff_precision
    post_variance = 1.0 / post_precision
    post_mean = post_variance * (prior_precision * prior_mean + weighted_obs)

    shape = (grid.n_rows, grid.n_cols)
    return post_mean.reshape(shape), post_variance.reshape(shape)
