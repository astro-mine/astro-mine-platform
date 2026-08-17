# SPDX-License-Identifier: Apache-2.0
"""EVPI tied to ISRU yield — active perception valued in production ROI (prospect.md §3, §11).

Max-variance and mutual information (:mod:`astro_mine.prospect.infogain`) value a sample in *nats*:
where the belief is most uncertain, where an observation removes the most entropy. Neither knows
what the swarm is *for*. This module adds the acquisition that does: it ties information gain to a
concrete **ISRU-yield production objective** — a develop-or-skip production decision per cell —
and values a sample by how much it improves that decision's expected payoff. It is the fourth
info-gain objective the extension point lists (prospect.md §3 "expected value of information for
ISRU yield"; §11 "EVPI when tied to a concrete ISRU production objective"), pluggable alongside the
others and consumable by [Allocate](allocate.md) as the information side of its info-gain-vs-ROI
trade (RM-P1-ALLOC-04).

Model. Each cell holds a resource concentration ``x_i`` with the belief's per-cell Gaussian
posterior ``N(mu_i, var_i)`` (the same per-cell-independent treatment the entropy/MI maps use — a
full-covariance EIG is P1, prospect.md §11). A production decision develops the cell (net payoff
``net_i(x) = k * relu(x - cutoff) - dev_cost``) or skips it (payoff ``0``), where
``k = value_coefficient * cell_area`` folds the resource's unit value, extraction efficiency, and
per-cell mass into one coefficient. The scale is arbitrary but internally consistent, so EVPI/EVSI
land in the *same value units as ROI* and Allocate trades them directly.

- :func:`expected_isru_yield` — the production objective's value under the current belief: the sum
  over cells of the best (develop/skip) expected payoff.
- :func:`evpi_map` — per-cell **EVPI**: the expected payoff gained by resolving a cell's
  uncertainty *before* deciding (``E[max(0, net)] - max(0, E[net]) >= 0``), in closed form.
- :func:`evsi_map` — per-cell **EVSI**: the expected payoff gained from one *noisy* observation at
  each candidate cell, via the belief's RBF pre-posterior (the yield-denominated analogue of
  :func:`~astro_mine.prospect.infogain.mutual_information_map`). This is the "where to sample next"
  map when what you care about is production, not entropy.

Backlog: RM-P1-PROSPECT-11 — astro-mine-prospect#21
"""

from __future__ import annotations

import hashlib

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field
from scipy.special import ndtr  # standard normal CDF, vectorized

from astro_mine.core.resource import Position
from astro_mine.prospect.belief.field import BeliefField
from astro_mine.prospect.field.metadata import FieldGrid

__all__ = [
    "ISRUYieldModel",
    "ISRUYieldObjective",
    "evpi_map",
    "evsi_map",
    "expected_isru_yield",
]

#: ``1 / sqrt(2*pi)`` — the standard normal density normaliser.
_INV_SQRT_2PI = 1.0 / np.sqrt(2.0 * np.pi)


def _phi(z: NDArray[np.float64]) -> NDArray[np.float64]:
    """The standard normal density ``phi(z)`` (vectorized, no SciPy dependency on the hot path)."""
    density: NDArray[np.float64] = _INV_SQRT_2PI * np.exp(-0.5 * z * z)
    return density


class ISRUYieldModel(BaseModel):
    """The ISRU-yield production objective an EVPI/EVSI acquisition is tied to (the ROI knobs).

    A cell is either **developed** — net payoff ``k * relu(concentration - cutoff) - dev_cost`` — or
    **skipped** (payoff ``0``). ``value_coefficient`` is the payoff per unit of concentration-excess
    per square metre (it folds the resource's unit value, extraction efficiency, and per-cell mass
    into one number); ``k`` per cell is ``value_coefficient * cell_area``. ``cutoff`` is the
    concentration below which extraction is not worthwhile; ``dev_cost`` the fixed value cost of
    bringing a cell into production. The absolute scale is arbitrary but consistent, so the
    resulting
    EVPI/EVSI are denominated in the same units as ROI (Allocate trades them directly).

    Frozen and content-addressable: the model is part of the acquisition's reproducibility key,
    recorded with the run (conventions.md §5).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    value_coefficient: float = Field(gt=0.0)
    cutoff: float = Field(ge=0.0, default=0.0)
    dev_cost: float = Field(ge=0.0, default=0.0)
    #: Gauss-Hermite nodes for the EVSI pre-posterior integral (deterministic; more = tighter).
    quadrature_nodes: int = Field(ge=3, le=64, default=9)

    @property
    def content_hash(self) -> str:
        """A stable SHA-256 over the canonicalized model — the acquisition's reproducibility key."""
        canonical = self.model_dump_json()
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _cell_area_m2(grid: FieldGrid) -> float:
    """The area of one grid cell in square metres (projected CRS metres)."""
    dx = (grid.max_x_m - grid.min_x_m) / grid.n_cols
    dy = (grid.max_y_m - grid.min_y_m) / grid.n_rows
    return float(dx * dy)


def _grid_of(belief: BeliefField) -> FieldGrid:
    grid = belief.metadata.grid
    assert grid is not None  # a BeliefField always carries its grid (enforced in __init__)
    return grid


def _expected_net(
    mean: NDArray[np.float64],
    variance: NDArray[np.float64],
    *,
    k: float,
    cutoff: float,
    dev_cost: float,
) -> NDArray[np.float64]:
    """``E[net]`` per cell under ``N(mean, variance)``: ``k * E[relu(x - cutoff)] - dev_cost``.

    ``E[relu(x - c)] = (mu - c) * Phi(z) + sigma * phi(z)`` with ``z = (mu - c) / sigma`` — the
    rectified-Gaussian expectation. This is the expected payoff of *developing* the cell given the
    belief; the develop/skip decision compares it against ``0``.
    """
    sigma = np.sqrt(variance)
    z = (mean - cutoff) / sigma
    expected_relu = (mean - cutoff) * ndtr(z) + sigma * _phi(z)
    net: NDArray[np.float64] = k * expected_relu - dev_cost
    return net


def _perfect_yield(
    mean: NDArray[np.float64],
    variance: NDArray[np.float64],
    *,
    k: float,
    cutoff: float,
    dev_cost: float,
) -> NDArray[np.float64]:
    """``E[max(0, net)]`` per cell — the payoff with the concentration resolved *before* deciding.

    ``net(x)`` is monotone increasing in ``x``, crossing zero at ``x* = cutoff + dev_cost / k``, so
    ``max(0, net(x)) = k * (x - cutoff) - dev_cost`` for ``x >= x*`` and ``0`` below. The
    expectation
    has the closed form ``[k*(mu-cutoff) - dev_cost] * (1 - Phi(z*)) + k*sigma*phi(z*)`` with
    ``z* = (x* - mu) / sigma``.
    """
    sigma = np.sqrt(variance)
    x_star = cutoff + dev_cost / k
    z_star = (x_star - mean) / sigma
    upper_tail = 1.0 - ndtr(z_star)  # P(x >= x*)
    value: NDArray[np.float64] = (k * (mean - cutoff) - dev_cost) * upper_tail + k * sigma * _phi(
        z_star
    )
    return value


def expected_isru_yield(belief: BeliefField, model: ISRUYieldModel) -> float:
    """The ISRU production objective under the current belief: total expected develop/skip payoff.

    Each cell is developed iff its expected net payoff is positive; the field's value is the sum of
    ``max(0, E[net_i])`` over cells. This is the objective the EVPI/EVSI acquisitions are *tied to*
    — resolving uncertainty is only worth something insofar as it improves this number.
    """
    grid = _grid_of(belief)
    k = model.value_coefficient * _cell_area_m2(grid)
    mean = belief.mean_grid()
    variance = belief.variance_grid()
    expected_net = _expected_net(mean, variance, k=k, cutoff=model.cutoff, dev_cost=model.dev_cost)
    return float(np.maximum(0.0, expected_net).sum())


def evpi_map(belief: BeliefField, model: ISRUYieldModel) -> NDArray[np.float64]:
    """Per-cell EVPI ``(n_rows, n_cols)``: the value of resolving each cell's uncertainty.

    ``EVPI_i = E[max(0, net_i)] - max(0, E[net_i]) >= 0`` — the expected payoff a *perfect*
    measurement at the cell would unlock by letting the develop/skip decision see the true
    concentration. Highest where uncertainty straddles the develop/skip break-even; ``0`` where the
    decision is already clear. Denominated in ROI value units (the yield the objective produces).
    """
    grid = _grid_of(belief)
    k = model.value_coefficient * _cell_area_m2(grid)
    mean = belief.mean_grid()
    variance = belief.variance_grid()
    perfect = _perfect_yield(mean, variance, k=k, cutoff=model.cutoff, dev_cost=model.dev_cost)
    prior = np.maximum(
        0.0, _expected_net(mean, variance, k=k, cutoff=model.cutoff, dev_cost=model.dev_cost)
    )
    return np.maximum(0.0, perfect - prior)


def evsi_map(
    belief: BeliefField, model: ISRUYieldModel, *, noise_sigma: float
) -> NDArray[np.float64]:
    """Per-cell EVSI ``(n_rows, n_cols)``: the value of one noisy observation at each candidate
    cell.

    Observing at candidate ``c`` (a sensor of noise ``noise_sigma``) sharpens every cell ``i`` by
    the
    belief's own RBF conditioning: it adds precision ``w(c, i) / noise_sigma**2`` — the *same*
    weights :meth:`~astro_mine.prospect.belief.field.BeliefField.update` conditions with — dropping
    cell ``i``'s variance from ``var_i`` to ``post_var_i``. The pre-posterior mean of cell ``i`` is
    then ``N(mu_i, var_i - post_var_i)``; integrating the develop/skip payoff over it (deterministic
    Gauss-Hermite quadrature) and summing over cells gives ``EVSI(c)`` — the yield-denominated
    "where to sample next" map, the ROI analogue of
    :func:`~astro_mine.prospect.infogain.mutual_information_map`.

    Per-cell-independent, like the MI map (prospect.md §11); cost is ``O(cells**2 * nodes)`` — fine
    for the anchor PSR tile, with large-grid tiling deferred to P1 (prospect.md §8).
    """
    if noise_sigma <= 0.0:
        raise ValueError(f"noise_sigma must be positive, got {noise_sigma}")
    grid = _grid_of(belief)
    k = model.value_coefficient * _cell_area_m2(grid)
    mean = belief.mean_grid().ravel()  # (M,)
    variance = belief.variance_grid().ravel()  # (M,)
    ls = belief.length_scale
    centers = _cell_centers(grid)  # (M, 2)

    diff = centers[:, None, :] - centers[None, :, :]  # (M_candidate, M_cell, 2)
    dist2 = np.sum(diff * diff, axis=2)  # (M, M)
    weights = np.exp(-dist2 / (2.0 * ls * ls))  # w(c, i): row = candidate c, col = cell i

    prior_precision = 1.0 / variance  # (M,)
    added = weights / (noise_sigma * noise_sigma)  # (M_c, M_i) precision one obs at c adds to i
    post_variance = 1.0 / (prior_precision[None, :] + added)  # (M_c, M_i)
    revision_var = np.maximum(0.0, variance[None, :] - post_variance)  # (M_c, M_i)

    # Gauss-Hermite over the pre-posterior mean: E_{m' ~ N(mu, revision_var)}[max(0, net(m'))].
    nodes, gh_weights = np.polynomial.hermite_e.hermegauss(model.quadrature_nodes)
    gh_weights = gh_weights * _INV_SQRT_2PI  # normalise to a N(0,1) expectation

    prior_value = np.maximum(
        0.0, _expected_net(mean, variance, k=k, cutoff=model.cutoff, dev_cost=model.dev_cost)
    )  # (M_i,) — the current per-cell decision value

    # E_{m' ~ N(mu_i, revision_var)}[ value of the develop/skip decision after observing ], per
    # (candidate c, cell i). Difference against the prior value *per cell* — never as a sum-then-
    # subtract of two O(1e8) totals, which would drown the ~O(1) EVSI in float cancellation.
    post_value = np.zeros_like(post_variance)  # (M_c, M_i)
    sigma_rev = np.sqrt(revision_var)
    for node, gw in zip(nodes, gh_weights, strict=True):
        mprime = mean[None, :] + sigma_rev * node  # (M_c, M_i)
        net = _expected_net(
            mprime, post_variance, k=k, cutoff=model.cutoff, dev_cost=model.dev_cost
        )
        post_value += gw * np.maximum(0.0, net)
    evsi_per_cell = np.maximum(0.0, post_value - prior_value[None, :])  # (M_c, M_i), >= 0 per cell
    return evsi_per_cell.sum(axis=1).reshape(grid.n_rows, grid.n_cols)


def _cell_centers(grid: FieldGrid) -> NDArray[np.float64]:
    """The ``(M, 2)`` cell-center coordinates, row-major (matching the belief grid ravel)."""
    dx = (grid.max_x_m - grid.min_x_m) / grid.n_cols
    dy = (grid.max_y_m - grid.min_y_m) / grid.n_rows
    xs = grid.min_x_m + (np.arange(grid.n_cols) + 0.5) * dx
    ys = grid.min_y_m + (np.arange(grid.n_rows) + 0.5) * dy
    gx, gy = np.meshgrid(xs, ys)  # (n_rows, n_cols)
    return np.stack([gx.ravel(), gy.ravel()], axis=1)


class ISRUYieldObjective:
    """An info-gain objective valued in ISRU-production ROI (the ``InfoGainObjective`` of §3).

    Binds an :class:`ISRUYieldModel` and exposes the current production value, the EVPI/EVSI maps,
    and the best next sample — the object [Allocate](allocate.md)/[Mind](mind.md) consume to trade
    *information value* against *production cost* in the same units (RM-P1-ALLOC-04). Content-
    addressable via :attr:`content_hash` so a run pins the exact acquisition it used.
    """

    def __init__(self, model: ISRUYieldModel) -> None:
        self._model = model

    @property
    def model(self) -> ISRUYieldModel:
        """The ISRU-yield production model this objective is tied to."""
        return self._model

    @property
    def content_hash(self) -> str:
        """The bound model's content hash — the acquisition's reproducibility key."""
        return self._model.content_hash

    def expected_yield(self, belief: BeliefField) -> float:
        """The production objective value under ``belief`` (:func:`expected_isru_yield`)."""
        return expected_isru_yield(belief, self._model)

    def evpi_map(self, belief: BeliefField) -> NDArray[np.float64]:
        """The per-cell EVPI map over ``belief`` (:func:`evpi_map`)."""
        return evpi_map(belief, self._model)

    def evsi_map(self, belief: BeliefField, *, noise_sigma: float) -> NDArray[np.float64]:
        """The per-cell EVSI map for a sensor of ``noise_sigma`` over ``belief``
        (:func:`evsi_map`)."""
        return evsi_map(belief, self._model, noise_sigma=noise_sigma)

    def best_sample_position(self, belief: BeliefField, *, noise_sigma: float) -> Position:
        """Where sampling next buys the most production value — the EVSI argmax cell center."""
        from astro_mine.prospect.infogain import best_sample_position

        return best_sample_position(self.evsi_map(belief, noise_sigma=noise_sigma), belief)
