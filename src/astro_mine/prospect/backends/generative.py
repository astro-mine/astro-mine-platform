"""GenerativeEnsembleField — a deep-generative / normalizing-flow backend (prospect.md §11).

Where the GP and GMRF backends carry a *parametric* (Gaussian) marginal at every point, some
resource fields are **non-Gaussian or multimodal** — a skewed ice distribution, a bimodal
"either a rich lens or barren" belief. This backend represents such a field with a **learned
generative model** and reports uncertainty as an **ensemble of sampled realizations** rather than a
closed-form mean/variance (prospect.md §11 "uncertainty representation" row).

Design (reduced-order, honest, deterministic):

- A **calibrated Gaussian base** ``N(μ(x), s²(x))`` per cell, from the same RBF measurement-fusion
  smoother the grid backend uses (:class:`~astro_mine.prospect.backends.grid.GridField`) — it
  reverts to the prior away from data, so the ensemble stays calibrated.
- A small **monotone normalizing flow** ``ĝ: R→R`` (a positive-slope sum-of-softplus transform,
  the invertible 1-D flow primitive; RealNVP/neural-spline flows are the scale-up path) trained on
  CPU under a fixed seed to warp a standard-normal latent into the *shape* of the standardized
  observation residuals — introducing skew/heavy tails while, by standardization, preserving the
  base mean and variance so calibration is not sacrificed.
- A query draws the **ensemble** ``{μ(x) + s(x)·ĝ(u_i)}`` from a fixed seeded latent set and reports
  ``mean``/``variance``/``quantile`` as *empirical* statistics over it — a genuinely non-Gaussian
  marginal. Its uncertainty representation is therefore **ensemble** (declared for Zarr tagging,
  :attr:`GenerativeEnsembleField.UNCERTAINTY_REPRESENTATION`; RM-P1-PROSPECT-10).

Like the GMRF it supports the replayable, content-addressed belief :meth:`update` and a
ground-truth :meth:`realize` draw, so it is usable as both a ``BeliefField`` and a
``GroundTruthField`` variant. Training and all draws are seeded (conventions.md §1.5).

Backlog: RM-P1-PROSPECT-10 — https://github.com/astro-mine/astro-mine-prospect/issues/20
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from statistics import NormalDist
from typing import Any, ClassVar

import numpy as np
import torch
from numpy.typing import NDArray

from astro_mine.core.resource import Position
from astro_mine.core.units import Epoch
from astro_mine.prospect.backends._random_field import correlated_standard_normal
from astro_mine.prospect.backends.grid import GridField
from astro_mine.prospect.belief.observation import FieldObservation
from astro_mine.prospect.field.base import BaseResourceField
from astro_mine.prospect.field.metadata import FieldMetadata

__all__ = ["GenerativeEnsembleField", "MonotoneFlow"]

#: The default number of realizations in a query ensemble — large enough for stable empirical
#: quantiles at the credible levels the calibration gate scores, small enough to stay interactive.
DEFAULT_ENSEMBLE_SIZE = 256


class MonotoneFlow:
    """A trained 1-D monotone normalizing flow ``ĝ(u) = (g(u) - m) / s``.

    ``g(u) = a·u + Σ_k h_k · softplus(u - t_k)`` with ``a > 0`` and ``h_k ≥ 0`` is strictly
    increasing (an invertible flow), and is standardized by ``(m, s)`` — the mean/std of ``g`` over
    a standard normal — so ``ĝ(U)`` has zero mean and unit variance for ``U ~ N(0, 1)``. Warping the
    base by ``ĝ`` therefore changes the *shape* of the marginal (skew, tails) while preserving the
    base mean and variance. Evaluated in pure NumPy at query time (torch is training-only).
    """

    def __init__(
        self, a: float, h: NDArray[np.float64], t: NDArray[np.float64], mean: float, std: float
    ) -> None:
        self._a = a
        self._h = h
        self._t = t
        self._mean = mean
        self._std = std

    def warp(self, u: NDArray[np.float64]) -> NDArray[np.float64]:
        """The standardized warp ``ĝ(u)`` (zero-mean, unit-variance over a standard normal)."""
        g = self._a * u + (
            self._h[None, :] * np.logaddexp(0.0, u[..., None] - self._t[None, :])
        ).sum(axis=-1)
        return (g - self._mean) / self._std

    @property
    def is_identity(self) -> bool:
        """Whether the flow is (numerically) the identity — no learned non-Gaussian shape."""
        return bool(np.allclose(self._h, 0.0, atol=1e-4))


def _fit_flow(
    residuals: NDArray[np.float64], *, n_knots: int, n_iter: int, reg: float, seed: int
) -> MonotoneFlow:
    """Fit a :class:`MonotoneFlow` to the empirical distribution of standardized ``residuals``.

    Matches the flow's quantiles to the residual quantiles (a 1-D optimal-transport objective) with
    an identity-pull regularizer, on CPU in double precision under ``seed``. With too few residuals
    to estimate shape the flow stays the identity, so the ensemble reduces to the calibrated
    Gaussian base — the honest default when data is scarce.
    """
    probs = np.linspace(0.02, 0.98, 49)
    latent_q = np.array([NormalDist().inv_cdf(p) for p in probs], dtype=np.float64)
    if residuals.size < 4:
        return MonotoneFlow(1.0, np.zeros(n_knots), np.linspace(-2, 2, n_knots), 0.0, 1.0)

    torch.manual_seed(seed)
    resid_q = torch.as_tensor(np.quantile(residuals, probs), dtype=torch.float64)
    lq = torch.as_tensor(latent_q, dtype=torch.float64)
    u0 = torch.as_tensor(  # a fixed standard-normal grid for standardization
        np.array([NormalDist().inv_cdf(p) for p in np.linspace(0.005, 0.995, 400)]),
        dtype=torch.float64,
    )
    raw_a = torch.zeros(1, dtype=torch.float64, requires_grad=True)  # softplus(0)≈0.69
    raw_h = torch.full((n_knots,), -4.0, dtype=torch.float64, requires_grad=True)  # ≈0 → identity
    knots = torch.as_tensor(np.linspace(-2.0, 2.0, n_knots), dtype=torch.float64)
    opt = torch.optim.Adam([raw_a, raw_h], lr=0.05)

    def warp(u: Any) -> Any:
        a = torch.nn.functional.softplus(raw_a)
        h = torch.nn.functional.softplus(raw_h)
        g = a * u + (h[None, :] * torch.nn.functional.softplus(u[:, None] - knots[None, :])).sum(1)
        g0 = a * u0 + (h[None, :] * torch.nn.functional.softplus(u0[:, None] - knots[None, :])).sum(
            1
        )
        return (g - g0.mean()) / g0.std().clamp_min(1e-6)

    for _ in range(n_iter):
        opt.zero_grad()
        pred = warp(lq)
        loss = ((pred - resid_q) ** 2).mean() + reg * ((pred - lq) ** 2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        a = float(torch.nn.functional.softplus(raw_a).item())
        h = torch.nn.functional.softplus(raw_h).detach().numpy().astype(np.float64)
        g0 = a * u0 + (
            torch.as_tensor(h)[None, :] * torch.nn.functional.softplus(u0[:, None] - knots[None, :])
        ).sum(1)
        return MonotoneFlow(
            a, h, knots.numpy().astype(np.float64), float(g0.mean()), float(g0.std())
        )


class GenerativeEnsembleField(BaseResourceField):
    """A deep-generative resource field — a calibrated Gaussian base warped by a learned flow.

    Construct with :meth:`build` (single-shot) or :meth:`from_prior` + :meth:`update` (the
    replayable belief form). Queries draw a fixed, seeded ensemble of realizations and report
    empirical mean/variance/quantiles over it — a non-Gaussian marginal whose uncertainty
    representation is ``"ensemble"``.
    """

    #: This backend reports an **ensemble** of sampled realizations (empirical marginal), declared
    #: in :meth:`zarr_attrs` so a persisted field is self-describing (RM-P1-PROSPECT-10).
    UNCERTAINTY_REPRESENTATION: ClassVar[str] = "ensemble"

    def __init__(
        self,
        metadata: FieldMetadata,
        *,
        prior_mean: float,
        prior_variance: float,
        length_scale: float | None,
        noise: float,
        log: tuple[FieldObservation, ...],
        ensemble_size: int,
        n_knots: int,
        n_iter: int,
        reg: float,
        seed: int,
    ) -> None:
        grid = metadata.grid
        if grid is None:
            raise ValueError("GenerativeEnsembleField requires metadata.grid (a FieldGrid domain)")
        if ensemble_size <= 1:
            raise ValueError(f"ensemble_size must be > 1, got {ensemble_size}")
        super().__init__(metadata)
        self._grid = grid
        self._prior_mean = prior_mean
        self._prior_variance = prior_variance
        self._length_scale = length_scale
        self._noise = noise
        self._log = log
        self._ensemble_size = ensemble_size
        self._n_knots = n_knots
        self._n_iter = n_iter
        self._reg = reg
        self._seed = seed

        points = [(o.x_m, o.y_m, o.z_m) for o in log] or None
        values = [o.value for o in log] or None
        self._base = GridField.build(
            metadata,
            train_points=points,
            train_values=values,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            length_scale=length_scale,
            noise=noise,
        )
        residuals = self._standardized_residuals(log)
        self._flow = _fit_flow(residuals, n_knots=n_knots, n_iter=n_iter, reg=reg, seed=seed)
        # A fixed, seeded latent set → deterministic queries (same field ⇒ same ensemble).
        self._latents = np.random.default_rng(seed).standard_normal(ensemble_size)
        self._warped = self._flow.warp(self._latents)  # standardized shape offsets

    def _standardized_residuals(self, log: tuple[FieldObservation, ...]) -> NDArray[np.float64]:
        """Observation residuals ``(y - μ)/s`` under the base — the flow's training signal."""
        if not log:
            return np.zeros((0,), dtype=np.float64)
        out = np.empty(len(log), dtype=np.float64)
        for i, o in enumerate(log):
            pos = (o.x_m, o.y_m, o.z_m)
            std = float(np.sqrt(max(self._base.variance(pos), 1e-12)))
            out[i] = (o.value - self._base.mean(pos)) / std
        return out

    @classmethod
    def from_prior(
        cls,
        metadata: FieldMetadata,
        *,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        length_scale: float | None = None,
        noise: float = 1e-2,
        ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
        n_knots: int = 6,
        n_iter: int = 200,
        reg: float = 0.5,
        seed: int = 0,
    ) -> GenerativeEnsembleField:
        """The un-observed generative prior (an empty log): the base prior with an identity flow."""
        return cls(
            metadata,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            length_scale=length_scale,
            noise=noise,
            log=(),
            ensemble_size=ensemble_size,
            n_knots=n_knots,
            n_iter=n_iter,
            reg=reg,
            seed=seed,
        )

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
        noise: float = 1e-2,
        ensemble_size: int = DEFAULT_ENSEMBLE_SIZE,
        n_knots: int = 6,
        n_iter: int = 200,
        reg: float = 0.5,
        seed: int = 0,
    ) -> GenerativeEnsembleField:
        """Single-shot: fit the generative field to ``train_points``/``train_values``.

        A convenience wrapper over :meth:`from_prior` + :meth:`update` for the ``make_backend``
        path. With no observations the field is the (Gaussian) prior — the flow is the identity.
        """
        field = cls.from_prior(
            metadata,
            prior_mean=prior_mean,
            prior_variance=prior_variance,
            length_scale=length_scale,
            noise=noise,
            ensemble_size=ensemble_size,
            n_knots=n_knots,
            n_iter=n_iter,
            reg=reg,
            seed=seed,
        )
        if train_points is None and train_values is None:
            return field
        if train_points is None or train_values is None:
            raise ValueError("train_points and train_values must be provided together, or neither")
        if len(train_points) != len(train_values):
            raise ValueError(
                f"train_points and train_values length mismatch: "
                f"{len(train_points)} vs {len(train_values)}"
            )
        observations = [
            FieldObservation(
                x_m=p[0], y_m=p[1], z_m=p[2], value=float(v), noise_sigma=float(np.sqrt(noise))
            )
            for p, v in zip(train_points, train_values, strict=True)
        ]
        return field.update(observations)

    def update(self, observations: Iterable[FieldObservation]) -> GenerativeEnsembleField:
        """Append ``observations`` and return the re-fit field (a new field).

        The new field's log is this one's log followed by ``observations``; the base and flow re-fit
        over the full ordered log under the same seed, so incremental and batch conditioning are
        identical (the replay property, RM-P0-PROSPECT-04).
        """
        return GenerativeEnsembleField(
            self._metadata,
            prior_mean=self._prior_mean,
            prior_variance=self._prior_variance,
            length_scale=self._length_scale,
            noise=self._noise,
            log=self._log + tuple(observations),
            ensemble_size=self._ensemble_size,
            n_knots=self._n_knots,
            n_iter=self._n_iter,
            reg=self._reg,
            seed=self._seed,
        )

    @property
    def log(self) -> tuple[FieldObservation, ...]:
        """The ordered observation log conditioned into this field."""
        return self._log

    @property
    def flow(self) -> MonotoneFlow:
        """The fitted monotone flow warping the Gaussian base into the ensemble's shape."""
        return self._flow

    def ensemble(self, position: Position) -> NDArray[np.float64]:
        """The fixed, seeded ensemble of realizations at ``position`` — the uncertainty object.

        ``μ(x) + s(x)·ĝ(u_i)`` over the field's fixed latent set: a genuinely non-Gaussian sample of
        the marginal from which mean/variance/quantiles are read empirically.
        """
        mean = self._base.mean(position)
        std = float(np.sqrt(max(self._base.variance(position), 0.0)))
        return mean + std * self._warped

    def mean(self, position: Position, *, epoch: Epoch | None = None) -> float:
        return float(np.mean(self.ensemble(position)))

    def variance(self, position: Position, *, epoch: Epoch | None = None) -> float:
        return float(np.var(self.ensemble(position)))

    def quantile(self, position: Position, q: float, *, epoch: Epoch | None = None) -> float:
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"quantile level q must be in [0, 1], got {q}")
        return float(np.quantile(self.ensemble(position), q, method="linear"))

    def sample(
        self,
        position: Position,
        *,
        n: int = 1,
        seed: int | None = None,
        epoch: Epoch | None = None,
    ) -> tuple[float, ...]:
        if n < 0:
            raise ValueError(f"n must be non-negative, got {n}")
        if n == 0:
            return ()
        mean = self._base.mean(position)
        std = float(np.sqrt(max(self._base.variance(position), 0.0)))
        latents = np.random.default_rng(seed).standard_normal(n)
        return tuple(float(v) for v in (mean + std * self._flow.warp(latents)))

    def realize(
        self, *, seed: int, correlation_length_m: float | None = None
    ) -> NDArray[np.float64]:
        """Draw one sealed full-field realization ``(n_rows, n_cols)`` (the ground-truth variant).

        A per-cell warped draw ``μ + s·ĝ(u)`` — a non-Gaussian realization of the whole field,
        reproducible in ``seed``.

        ``correlation_length_m`` makes the realization **spatially correlated**: the latent ``u`` is
        drawn as a seeded, correlated standard-normal field
        (:func:`~astro_mine.prospect.backends._random_field.correlated_standard_normal`) instead of
        per-cell independent noise, so the draw carries spatial structure *and* the flow's
        non-Gaussian shape. This is the form the sealed ground-truth path uses
        (:func:`~astro_mine.prospect.belief.ground_truth.sample_ground_truth`); ``None`` keeps the
        per-cell-independent latent (the default, unchanged).

        It is the **practical range** — the separation at which correlation decays to ~0.1 — the
        same convention the GMRF backend uses, so the two realization backends structure a truth
        over the same length when asked for the same number. Their covariance *shapes* still differ
        (Gaussian-smooth here, Matérn nu = 1 there), so their short-lag correlation differs; only
        the range agrees.
        """
        grid = self._grid
        dx = (grid.max_x_m - grid.min_x_m) / grid.n_cols
        dy = (grid.max_y_m - grid.min_y_m) / grid.n_rows
        if correlation_length_m is None:
            u = np.random.default_rng(seed).standard_normal((grid.n_rows, grid.n_cols))
        else:
            u = correlated_standard_normal(
                grid.n_rows,
                grid.n_cols,
                dx_m=dx,
                dy_m=dy,
                correlation_length_m=correlation_length_m,
                seed=seed,
            )
        warped = self._flow.warp(u.ravel()).reshape((grid.n_rows, grid.n_cols))
        out = np.empty((grid.n_rows, grid.n_cols), dtype=np.float64)
        for r in range(grid.n_rows):
            for c in range(grid.n_cols):
                pos = (grid.min_x_m + (c + 0.5) * dx, grid.min_y_m + (r + 0.5) * dy, 0.0)
                std = float(np.sqrt(max(self._base.variance(pos), 0.0)))
                out[r, c] = self._base.mean(pos) + std * warped[r, c]
        return out

    @property
    def content_hash(self) -> str:
        """A stable content address over the base config, flow hyper-parameters, and the log."""
        digest = hashlib.sha256()
        digest.update(self._metadata.model_dump_json().encode("utf-8"))
        digest.update(
            repr(
                (
                    self._prior_mean,
                    self._prior_variance,
                    self._length_scale,
                    self._noise,
                    self._ensemble_size,
                    self._n_knots,
                    self._n_iter,
                    self._reg,
                    self._seed,
                )
            ).encode("utf-8")
        )
        for observation in self._log:
            digest.update(observation.model_dump_json().encode("utf-8"))
        return digest.hexdigest()

    def zarr_attrs(self) -> dict[str, str]:
        """The self-describing attributes stamped onto a persisted (Zarr) field.

        Records the ``uncertainty_representation`` (``"ensemble"``) and the ensemble size alongside
        species/unit so a stored or Hub-published field declares how its uncertainty is encoded
        (RM-P1-PROSPECT-10).
        """
        return {
            "uncertainty_representation": self.UNCERTAINTY_REPRESENTATION,
            "backend": "generative",
            "ensemble_size": str(self._ensemble_size),
            "species": self.species,
            "unit": self.unit,
        }
