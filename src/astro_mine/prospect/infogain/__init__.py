"""Information-gain maps for active perception (prospect.md §3, §6; RM-P0-PROSPECT-06).

Derivatives of a :class:`~astro_mine.prospect.belief.field.BeliefField` that quantify *where it is
most worth looking* and *how much has been learned*. Everything is built from the per-cell Gaussian
differential entropy of the belief, consistent with its per-cell Gaussian posterior model:

- **variance map** (:func:`variance_map`) — posterior variance per cell; the max-variance
  acquisition (field-unit², higher = more uncertain).
- **mutual-information map** (:func:`mutual_information_map`) — the expected total entropy reduction
  from one noisy observation at each candidate cell, evaluated against the belief's own
  RBF-precision conditioning model; the active-perception "where to sample next" signal (nats, ≥ 0).
- **entropy / information gain** (:func:`field_entropy`, :func:`information_gain`) — the nats of
  uncertainty a belief holds, and the nats removed between two beliefs. This is the single
  definition Bench's information-gain metric scores along a trace (RM-P0-BENCH-03): Bench reduces
  it, Prospect owns it.

All are pure, deterministic functions of the belief (conventions.md §1.5) and operate on the
**agent-facing belief only** — never the sealed ground truth (whose variance is zero; ground-truth
isolation is RM-P0-PROSPECT-05).

Cost: the mutual-information map is O(grid-cells²); fine for the anchor PSR tile, with large-grid
tiling deferred to P1 (prospect.md §8). The per-cell-independent Gaussian treatment is the Phase-0
reference definition; a full-covariance GP/GMRF EIG is P1 (prospect.md §11).

Backlog: RM-P0-PROSPECT-06 — https://github.com/astro-mine/astro-mine-prospect/issues/6
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from astro_mine.core.resource import Position
from astro_mine.prospect.belief.field import BeliefField
from astro_mine.prospect.field.metadata import FieldGrid
from astro_mine.prospect.infogain.isru import (
    ISRUYieldModel,
    ISRUYieldObjective,
    evpi_map,
    evsi_map,
    expected_isru_yield,
)

__all__ = [
    "ISRUYieldModel",
    "ISRUYieldObjective",
    "InfoGainKind",
    "best_sample_position",
    "entropy_map",
    "evpi_map",
    "evsi_map",
    "expected_isru_yield",
    "field_entropy",
    "information_gain",
    "information_gain_map",
    "mutual_information_map",
    "variance_map",
]

#: The kinds the unified :func:`information_gain_map` dispatches on. ``variance`` and
#: ``mutual_information`` are entropy-denominated (nats); ``evpi`` and ``evsi`` are
#: ISRU-yield-denominated (they require an
#: :class:`~astro_mine.prospect.infogain.isru.ISRUYieldModel`) — the acquisition
#: [Allocate](allocate.md) trades against production ROI (RM-P1-PROSPECT-11).
InfoGainKind = Literal["variance", "mutual_information", "evpi", "evsi"]

#: Constant in the Gaussian differential entropy ``H = 0.5 * ln(2*pi*e * variance)``.
_LOG_2PIE = math.log(2.0 * math.pi * math.e)


def variance_map(belief: BeliefField) -> NDArray[np.float64]:
    """The per-cell posterior variance ``(n_rows, n_cols)`` — the max-variance acquisition map.

    Highest where the belief is least certain (unobserved regions). Units are the field unit
    squared; higher means more worth sampling.
    """
    return belief.variance_grid()


def entropy_map(belief: BeliefField) -> NDArray[np.float64]:
    """The per-cell Gaussian differential entropy ``(n_rows, n_cols)`` in nats.

    ``H_i = 0.5 * ln(2*pi*e * variance_i)`` — the information content of each cell's posterior.
    Differential entropy may be negative for sub-unit variances; the meaningful, sign-stable
    quantity is its reduction, :func:`information_gain`.
    """
    return 0.5 * (_LOG_2PIE + np.log(belief.variance_grid()))


def field_entropy(belief: BeliefField) -> float:
    """The total belief entropy (nats): the sum of :func:`entropy_map` over all cells."""
    return float(entropy_map(belief).sum())


def information_gain(reference: BeliefField, updated: BeliefField) -> float:
    """Nats of uncertainty removed going from ``reference`` to ``updated`` over the same grid.

    ``H(reference) - H(updated)`` — the total information the swarm gained (e.g. a prior belief vs a
    conditioned posterior). Direction: higher is better. This is the definition Bench's
    information-gain metric scores (RM-P0-BENCH-03), recovered from belief states along a trace.
    """
    ref_grid = _grid_of(reference)
    upd_grid = _grid_of(updated)
    if (ref_grid.n_rows, ref_grid.n_cols) != (upd_grid.n_rows, upd_grid.n_cols):
        raise ValueError(
            "information_gain requires both beliefs on the same grid shape; got "
            f"{(ref_grid.n_rows, ref_grid.n_cols)} and {(upd_grid.n_rows, upd_grid.n_cols)}"
        )
    return field_entropy(reference) - field_entropy(updated)


def mutual_information_map(belief: BeliefField, *, noise_sigma: float) -> NDArray[np.float64]:
    """Expected entropy reduction (nats) from one noisy observation at each candidate cell.

    For a sensor with measurement noise ``noise_sigma`` placed at cell ``c``'s center, the belief's
    RBF-precision conditioning adds precision ``w(c, i) / noise_sigma**2`` to every cell ``i`` (the
    same weights the belief conditions with: ``w(c, i) = exp(-dist(c, i)**2 / (2 * L**2))`` for the
    belief's length scale ``L``). The expected total entropy drop is

        ``MI(c) = 0.5 * sum_i ln(1 + variance_i * w(c, i) / noise_sigma**2)``   (nats, >= 0),

    the mutual information between observing at ``c`` and the field: highest where sampling reduces
    the most uncertainty (the active-perception "where to sample next" map).
    """
    if noise_sigma <= 0.0:
        raise ValueError(f"noise_sigma must be positive, got {noise_sigma}")
    grid = _grid_of(belief)
    centers = _cell_centers(grid)  # (M, 2)
    variance = belief.variance_grid().ravel()  # (M,)
    ls = belief.length_scale
    diff = centers[:, None, :] - centers[None, :, :]  # (M_candidate, M_cell, 2)
    dist2 = np.sum(diff * diff, axis=2)  # (M, M)
    weights = np.exp(-dist2 / (2.0 * ls * ls))  # w_i(c): row = candidate c, col = cell i
    gain = 0.5 * np.log1p(weights * variance[None, :] / (noise_sigma * noise_sigma))  # (M, M)
    return gain.sum(axis=1).reshape(grid.n_rows, grid.n_cols)


def information_gain_map(
    belief: BeliefField,
    *,
    kind: InfoGainKind = "variance",
    noise_sigma: float | None = None,
    yield_model: ISRUYieldModel | None = None,
) -> NDArray[np.float64]:
    """A unified info-gain map dispatching on ``kind``.

    - ``"variance"`` (default) / ``"mutual_information"`` — entropy-denominated (nats);
      ``"mutual_information"`` requires ``noise_sigma`` (the prospective sensor's measurement
      noise).
    - ``"evpi"`` / ``"evsi"`` — ISRU-yield-denominated; both require ``yield_model`` (the production
      objective, :class:`~astro_mine.prospect.infogain.isru.ISRUYieldModel`), and ``"evsi"`` also
      requires ``noise_sigma`` (RM-P1-PROSPECT-11).
    """
    if kind == "variance":
        return variance_map(belief)
    if kind == "mutual_information":
        if noise_sigma is None:
            raise ValueError(
                "kind='mutual_information' requires noise_sigma (the sensor's measurement noise)"
            )
        return mutual_information_map(belief, noise_sigma=noise_sigma)
    if kind == "evpi":
        if yield_model is None:
            raise ValueError("kind='evpi' requires yield_model (the ISRU production objective)")
        return evpi_map(belief, yield_model)
    if kind == "evsi":
        if yield_model is None:
            raise ValueError("kind='evsi' requires yield_model (the ISRU production objective)")
        if noise_sigma is None:
            raise ValueError("kind='evsi' requires noise_sigma (the sensor's measurement noise)")
        return evsi_map(belief, yield_model, noise_sigma=noise_sigma)
    raise ValueError(
        f"unknown info-gain kind {kind!r}; use 'variance', 'mutual_information', 'evpi', or 'evsi'"
    )


def best_sample_position(info_map: NDArray[np.float64], belief: BeliefField) -> Position:
    """The center position ``(x, y, 0.0)`` of the highest-value cell in ``info_map``.

    Where active perception should sample next. ``info_map`` must match the belief's grid shape; on
    ties the first (row-major) maximum wins.
    """
    grid = _grid_of(belief)
    if info_map.shape != (grid.n_rows, grid.n_cols):
        raise ValueError(
            f"info_map shape {info_map.shape} does not match the belief grid "
            f"{(grid.n_rows, grid.n_cols)}"
        )
    row, col = np.unravel_index(int(np.argmax(info_map)), info_map.shape)
    dx = (grid.max_x_m - grid.min_x_m) / grid.n_cols
    dy = (grid.max_y_m - grid.min_y_m) / grid.n_rows
    x = grid.min_x_m + (int(col) + 0.5) * dx
    y = grid.min_y_m + (int(row) + 0.5) * dy
    return (float(x), float(y), 0.0)


def _grid_of(belief: BeliefField) -> FieldGrid:
    grid = belief.metadata.grid
    assert grid is not None  # a BeliefField always carries its grid (enforced in __init__)
    return grid


def _cell_centers(grid: FieldGrid) -> NDArray[np.float64]:
    """The ``(M, 2)`` cell-center coordinates, row-major (matching the belief's grid ravel)."""
    dx = (grid.max_x_m - grid.min_x_m) / grid.n_cols
    dy = (grid.max_y_m - grid.min_y_m) / grid.n_rows
    xs = grid.min_x_m + (np.arange(grid.n_cols) + 0.5) * dx
    ys = grid.min_y_m + (np.arange(grid.n_rows) + 0.5) * dy
    gx, gy = np.meshgrid(xs, ys)  # (n_rows, n_cols)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)
