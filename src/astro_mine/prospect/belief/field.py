# SPDX-License-Identifier: Apache-2.0
"""BeliefField — a Bayesian posterior over a resource field with a replayable log (prospect.md §5).

The agents' evolving estimate: a :class:`BeliefField` carries a dataset-derived
:class:`~astro_mine.prospect.priors.recipe.Prior` and an **ordered observation log**, and serves the
posterior that conditioning the prior on that log yields. :meth:`BeliefField.update` appends
observations and returns a *new* belief (the field is immutable) whose posterior is a pure function
of ``(prior, full ordered log, length_scale)`` — so it is **content-addressed** and **replayable**:
conditioning incrementally and conditioning on the whole log in one call produce byte-identical
posteriors (RM-P0-PROSPECT-04 acceptance). The chain prior -> update -> ... -> posterior is exactly
the sequence of log prefixes (prospect.md §5).

Conditioning is a per-cell, **likelihood-weighted** Gaussian precision fusion of the
spatially-varying prior with the log. Each observation is read under the sensor likelihood it is
tagged with (:mod:`astro_mine.prospect.sensors`), contributing its instrument's spatial
**footprint** and **depth-response gain** as well as its ``noise_sigma`` — so the belief conditions
under the same observation model Sim rendered the reading with (prospect.md §3, §6), while an
untagged reading falls back to the zero-footprint, unit-gain point model: the pre-instrument
arithmetic, unchanged. It is the deterministic, dependency-light reference path — the same
measurement-fusion smoother the grid backend uses (RM-P0-PROSPECT-02), generalized from a constant
prior to the dataset prior. A GP/GMRF-backed belief is deferred to P1 (prospect.md §11). Implemented
here so the shipped #2 grid backend is untouched; the equivalence is pinned by a test (constant
prior == ``GridField.build``).

Backlog: RM-P0-PROSPECT-04 — astro-mine-prospect#4
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable

import numpy as np
from numpy.typing import NDArray

from astro_mine.core.resource import Position
from astro_mine.core.units import Epoch
from astro_mine.prospect.backends.grid import GridField
from astro_mine.prospect.belief.observation import FieldObservation
from astro_mine.prospect.field.base import BaseResourceField
from astro_mine.prospect.field.metadata import FieldGrid, FieldMetadata
from astro_mine.prospect.priors.recipe import Prior
from astro_mine.prospect.sensors import resolve_likelihood

__all__ = ["BeliefField"]


class BeliefField(BaseResourceField):
    """A posterior resource field carrying a prior + an ordered, replayable observation log.

    Construct the initial (un-observed) belief with :meth:`from_prior`, then :meth:`update` it with
    observations. Every instance is immutable; :meth:`update` returns a new belief. Queries
    (``mean``/``variance``/``quantile``/``sample``) serve the conditioned posterior via an internal
    :class:`~astro_mine.prospect.backends.grid.GridField`; :attr:`content_hash` content-addresses
    the posterior by its ``(prior, log, length_scale)`` so a Bench scenario pins it (§5).
    """

    def __init__(
        self,
        metadata: FieldMetadata,
        prior_mean: NDArray[np.float64],
        prior_variance: NDArray[np.float64],
        log: tuple[FieldObservation, ...],
        *,
        length_scale: float | None,
        prior_hash: str,
    ) -> None:
        grid = metadata.grid
        if grid is None:
            raise ValueError("BeliefField requires metadata.grid (a FieldGrid spatial domain)")
        shape = (grid.n_rows, grid.n_cols)
        if prior_mean.shape != shape or prior_variance.shape != shape:
            raise ValueError(
                f"prior mean/variance must have shape {shape} (n_rows, n_cols); "
                f"got mean {prior_mean.shape}, variance {prior_variance.shape}"
            )
        if bool(np.any(prior_variance <= 0.0)):
            raise ValueError("belief prior_variance must be positive everywhere")
        resolved_ls = (
            0.1 * max(grid.max_x_m - grid.min_x_m, grid.max_y_m - grid.min_y_m)
            if length_scale is None
            else float(length_scale)
        )
        if resolved_ls <= 0.0:
            raise ValueError(f"length_scale must be positive, got {resolved_ls}")
        super().__init__(metadata)
        self._prior_mean = np.ascontiguousarray(prior_mean, dtype=np.float64)
        self._prior_variance = np.ascontiguousarray(prior_variance, dtype=np.float64)
        self._prior_mean.flags.writeable = False
        self._prior_variance.flags.writeable = False
        self._log = log
        self._length_scale = resolved_ls
        self._prior_hash = prior_hash
        post_mean, post_variance = _condition(
            self._prior_mean, self._prior_variance, grid, log, length_scale=resolved_ls
        )
        self._field = GridField(metadata, post_mean, post_variance)

    @classmethod
    def from_prior(cls, prior: Prior, *, length_scale: float | None = None) -> BeliefField:
        """The initial belief: the ``prior`` itself, with an empty observation log.

        ``length_scale`` is the conditioning correlation length (default: 10% of the larger grid
        extent, matching the grid backend); it is fixed for the belief's whole update chain so the
        posterior depends only on the prior and the log.
        """
        return cls(
            prior.metadata,
            prior.mean,
            prior.variance,
            (),
            length_scale=length_scale,
            prior_hash=prior.content_hash,
        )

    def update(self, observations: Iterable[FieldObservation]) -> BeliefField:
        """Append ``observations`` to the log and return the re-conditioned belief (a new field).

        The new belief's log is this one's log followed by ``observations``, in order. The posterior
        is re-fused from the prior over the full log, so updating incrementally and updating in
        one batch over the same ordered observations are identical (the replay property, §5).
        """
        new_log = self._log + tuple(observations)
        return BeliefField(
            self._metadata,
            self._prior_mean,
            self._prior_variance,
            new_log,
            length_scale=self._length_scale,
            prior_hash=self._prior_hash,
        )

    @property
    def log(self) -> tuple[FieldObservation, ...]:
        """The ordered observation log conditioned into this posterior."""
        return self._log

    @property
    def prior_hash(self) -> str:
        """The content hash of the prior this belief conditions (the chain's root)."""
        return self._prior_hash

    @property
    def length_scale(self) -> float:
        """The RBF conditioning correlation length (metres), fixed for this belief's update chain.

        The characteristic length over which one observation informs neighbouring cells — needed by
        the mutual-information map (:mod:`astro_mine.prospect.infogain`) to reproduce the weights
        this belief conditions with.
        """
        return self._length_scale

    def mean_grid(self) -> NDArray[np.float64]:
        """The per-cell posterior mean grid ``(n_rows, n_cols)`` (a copy).

        The agent-facing best estimate, gridded — paired with :meth:`variance_grid` it is the
        substrate the ISRU-yield acquisition (:mod:`astro_mine.prospect.infogain.isru`) values a
        develop/skip production decision over (RM-P1-PROSPECT-11).
        """
        return self._field.mean_grid()

    def variance_grid(self) -> NDArray[np.float64]:
        """The per-cell posterior variance grid ``(n_rows, n_cols)`` (a copy).

        The agent-facing uncertainty surface, gridded — the substrate for information-gain maps
        (:mod:`astro_mine.prospect.infogain`, RM-P0-PROSPECT-06). Belief variance is the swarm's own
        uncertainty, never the sealed ground truth.
        """
        return self._field.variance_grid()

    def mean(self, position: Position, *, epoch: Epoch | None = None) -> float:
        return self._field.mean(position)

    def variance(self, position: Position, *, epoch: Epoch | None = None) -> float:
        return self._field.variance(position)

    def quantile(self, position: Position, q: float, *, epoch: Epoch | None = None) -> float:
        return self._field.quantile(position, q)

    def sample(
        self,
        position: Position,
        *,
        n: int = 1,
        seed: int | None = None,
        epoch: Epoch | None = None,
    ) -> tuple[float, ...]:
        return self._field.sample(position, n=n, seed=seed)

    @property
    def content_hash(self) -> str:
        """A stable content address over the prior, the conditioning length scale, and the log.

        Two beliefs with the same prior and the same ordered log share a hash — so replaying a log
        (in any chunking) reproduces the same posterior identity (prospect.md §5).
        """
        digest = hashlib.sha256()
        digest.update(self._prior_hash.encode("utf-8"))
        digest.update(repr(self._length_scale).encode("utf-8"))
        for observation in self._log:
            digest.update(observation.model_dump_json().encode("utf-8"))
        return digest.hexdigest()


def _condition(
    prior_mean: NDArray[np.float64],
    prior_variance: NDArray[np.float64],
    grid: FieldGrid,
    log: tuple[FieldObservation, ...],
    *,
    length_scale: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Likelihood-weighted Gaussian precision fusion of an ordered log into a per-cell posterior.

    Generalizes the grid backend's constant-prior smoother to the spatially-varying dataset prior,
    to per-observation noise, **and to per-instrument likelihoods** (prospect.md §3, §6). Each
    observation is read under the sensor likelihood it is tagged with, which supplies

    - ``w`` — the per-cell weights: the instrument's spatial **footprint** convolved with the
      belief's correlation length (a broad-footprint neutron reading informs a whole neighbourhood;
      a drill assay informs its own cell);
    - ``g`` — the **depth-response gain** relating the instrument's depth window to the field's
      reference column.

    A reading ``y`` of noise ``s`` then contributes precision ``w * g**2 / s**2`` about the field,
    at the effective field value ``y / g`` — so a surface-only NIR reading is honestly weak evidence
    about the buried column, and a drill assay honestly strong. An **untagged** observation resolves
    to the zero-footprint, unit-gain default likelihood, for which ``w`` is the belief's own RBF
    weight and ``g == 1``: the pre-instrument arithmetic exactly. With an empty log the posterior is
    the prior.
    """
    shape = (grid.n_rows, grid.n_cols)
    if not log:
        return prior_mean.copy(), prior_variance.copy()

    n_cells = grid.n_rows * grid.n_cols
    contrib = np.empty((n_cells, len(log)), dtype=np.float64)  # (M, K) likelihood precision
    field_values = np.empty(len(log), dtype=np.float64)  # (K,) depth-corrected values
    for k, observation in enumerate(log):
        likelihood = resolve_likelihood(observation.likelihood)
        weights = likelihood.conditioning_weights(
            grid, observation.position, correlation_length_m=length_scale
        )
        contrib[:, k] = weights * likelihood.precision(observation.noise_sigma)
        field_values[k] = likelihood.to_field_value(observation.value)

    prior_precision = 1.0 / prior_variance.ravel()  # (M,)
    eff_precision = contrib.sum(axis=1)  # (M,)
    weighted_obs = (contrib * field_values[None, :]).sum(axis=1)  # (M,)
    post_precision = prior_precision + eff_precision
    post_variance = 1.0 / post_precision
    post_mean = post_variance * (prior_precision * prior_mean.ravel() + weighted_obs)
    return post_mean.reshape(shape), post_variance.reshape(shape)
