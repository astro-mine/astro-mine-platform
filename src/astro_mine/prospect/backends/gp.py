"""GPField — the GPyTorch sparse/variational GP backend (prospect.md §8, §11).

A sparse Gaussian-process posterior over the resource field, built on GPyTorch's inducing-point
kernel (the Titsias sparse/variational GP with a closed-form predictive — "SGPR"), the
recommended default large-scale path (prospect.md §11). It conditions on scattered observations
and answers the Core ResourceField contract: a Gaussian *latent-function* posterior at each query
point (mean + variance), with calibrated quantiles / seeded samples via the shared
:mod:`~astro_mine.prospect.backends._gaussian` helper — a drop-in alternative to the grid backend
(RM-P0-PROSPECT-02).

Inputs and outputs are standardized internally for numerical conditioning (planar coordinates
span kilometres); hyperparameters are fit by maximizing the sparse exact-marginal-likelihood on
**CPU in double precision under a fixed seed**, so a fit is reproducible (conventions.md §1.5). An
empty observation set degenerates to the GP prior. The ordered observation-log / belief machinery
is RM-P0-PROSPECT-04; the stochastic-ELBO SVGP variant for very large data is deferred to P1.

Backlog: RM-P0-PROSPECT-02 — https://github.com/astro-mine/astro-mine-prospect/issues/2
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import gpytorch
import torch

from astro_mine.core.resource import Position
from astro_mine.core.units import Epoch
from astro_mine.prospect.backends._gaussian import gaussian_quantile, gaussian_samples
from astro_mine.prospect.backends._training import validate_training
from astro_mine.prospect.field.base import BaseResourceField
from astro_mine.prospect.field.metadata import FieldMetadata

__all__ = ["GPField"]


class _SGPRModel(gpytorch.models.ExactGP):  # type: ignore[misc]  # gpytorch base is untyped (Any)
    """A sparse GP: constant mean + an inducing-point-approximated scaled RBF covariance."""

    def __init__(self, train_x: Any, train_y: Any, likelihood: Any, inducing_points: Any) -> None:
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.InducingPointKernel(
            gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=2)),
            inducing_points=inducing_points,
            likelihood=likelihood,
        )

    def forward(self, x: Any) -> Any:
        return gpytorch.distributions.MultivariateNormal(self.mean_module(x), self.covar_module(x))


class GPField(BaseResourceField):
    """A GPyTorch sparse-GP resource field, fit once on construction.

    With no observations the field is the flat ``N(prior_mean, prior_variance)`` GP prior; given
    ``train_points``/``train_values`` it fits the sparse GP and serves its latent posterior.
    """

    def __init__(
        self,
        metadata: FieldMetadata,
        *,
        train_points: Sequence[Position] | None = None,
        train_values: Sequence[float] | None = None,
        prior_mean: float = 0.0,
        prior_variance: float = 1.0,
        n_inducing: int = 32,
        n_iter: int = 100,
        learning_rate: float = 0.1,
        noise: float = 1e-2,
        seed: int = 0,
    ) -> None:
        super().__init__(metadata)
        xy, y = validate_training(train_points, train_values)
        self._model: Any = None
        if y.shape[0] == 0:
            if prior_variance <= 0.0:
                raise ValueError(f"prior_variance must be positive, got {prior_variance}")
            self._prior_mean = float(prior_mean)
            self._prior_variance = float(prior_variance)
            return
        self._fit(
            xy, y, n_inducing=n_inducing, n_iter=n_iter, lr=learning_rate, noise=noise, seed=seed
        )

    def _fit(
        self,
        xy: Any,
        y: Any,
        *,
        n_inducing: int,
        n_iter: int,
        lr: float,
        noise: float,
        seed: int,
    ) -> None:
        torch.manual_seed(seed)
        train_x = torch.as_tensor(xy, dtype=torch.float64)
        train_y = torch.as_tensor(y, dtype=torch.float64)
        # Standardize inputs/outputs so the unit-scale RBF kernel is well-conditioned.
        self._x_mean = train_x.mean(dim=0)
        self._x_std = train_x.std(dim=0).clamp_min(1e-12)
        self._y_mean = train_y.mean()
        self._y_std = train_y.std().clamp_min(1e-12)
        xs = (train_x - self._x_mean) / self._x_std
        ys = (train_y - self._y_mean) / self._y_std

        n_train = int(train_x.shape[0])
        m = min(n_inducing, n_train)
        idx = torch.linspace(0, n_train - 1, steps=m).round().long()
        inducing = xs[idx].clone()

        likelihood = gpytorch.likelihoods.GaussianLikelihood().double()
        likelihood.noise = noise
        model = _SGPRModel(xs, ys, likelihood, inducing).double()
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

        model.train()
        likelihood.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for _ in range(n_iter):
            optimizer.zero_grad()
            loss = -mll(model(xs), ys)
            loss.backward()
            optimizer.step()

        model.eval()
        likelihood.eval()
        self._model = model

    def _predict(self, position: Position) -> tuple[float, float]:
        if self._model is None:
            return self._prior_mean, self._prior_variance
        test_x = torch.as_tensor([[position[0], position[1]]], dtype=torch.float64)
        test_xs = (test_x - self._x_mean) / self._x_std
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = self._model(test_xs)
        mean = float(posterior.mean.item()) * float(self._y_std) + float(self._y_mean)
        variance = float(posterior.variance.item()) * float(self._y_std) ** 2
        return mean, max(variance, 0.0)

    def mean(self, position: Position, *, epoch: Epoch | None = None) -> float:
        return self._predict(position)[0]

    def variance(self, position: Position, *, epoch: Epoch | None = None) -> float:
        return self._predict(position)[1]

    def quantile(self, position: Position, q: float, *, epoch: Epoch | None = None) -> float:
        mean, variance = self._predict(position)
        return gaussian_quantile(mean, variance, q)

    def sample(
        self,
        position: Position,
        *,
        n: int = 1,
        seed: int | None = None,
        epoch: Epoch | None = None,
    ) -> tuple[float, ...]:
        mean, variance = self._predict(position)
        return gaussian_samples(mean, variance, n=n, seed=seed)
