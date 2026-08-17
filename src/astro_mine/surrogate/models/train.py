# SPDX-License-Identifier: Apache-2.0
"""Training the learned-DEM GNS ensemble (RM-P1-SURR-02).

Turns the frozen DEM fixture into transition examples, trains a **deep ensemble** of GNS models
(seeded, with **noise-injection / pushforward** so the network is robust to its own rollout drift
— surrogate.md §8), and predicts a mean + epistemic std the conformal layer calibrates. The
network predicts a per-particle acceleration; the surrogate integrates it (semi-implicit) to the
next state. Deterministic: same seed + same fixture reproduce the same weights in-process (torch
CPU is not bit-portable across builds, so CI gates by tolerance, not a bit-exact golden).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt
import torch

from astro_mine.surrogate.models.dataset import DemDataset
from astro_mine.surrogate.models.gns import (
    GNS,
    build_edges,
    edge_features,
    node_features,
)

__all__ = [
    "Normalizer",
    "TrainConfig",
    "Transition",
    "build_transitions",
    "ensemble_predict",
    "integrate",
    "train_ensemble",
]

FloatArray = npt.NDArray[np.float64]

#: The radius-graph connection cutoff, in particle radii (fixed by the fixture's packing).
_CUTOFF_RADII = 2.6
_PARTICLE_RADIUS_M = 0.02


@dataclass(frozen=True)
class TrainConfig:
    """Small-by-design training hyperparameters (CPU, tiny reference bed)."""

    hidden: int = 32
    message_passing_steps: int = 2
    epochs: int = 200
    learning_rate: float = 5e-3
    noise_std: float = 0.05  # pushforward noise on the (normalized) node features
    ensemble_size: int = 3


@dataclass(frozen=True)
class Transition:
    """One ``(state_t, tool_x, config) -> next_state`` DEM transition."""

    state: FloatArray  # (N, 4) pos_x,pos_z,vel_x,vel_z at t
    tool_x_m: float
    config: FloatArray  # (P,) density,friction,restitution,tool_speed
    next_state: FloatArray  # (N, 4) at t+1


@dataclass(frozen=True)
class Normalizer:
    """Standardization stats (mean/std) for node features and the acceleration target."""

    node_mean: FloatArray
    node_std: FloatArray
    accel_mean: FloatArray
    accel_std: FloatArray
    config_mean: FloatArray = field(default_factory=lambda: np.zeros(2))
    config_std: FloatArray = field(default_factory=lambda: np.ones(2))

    def config_norm(self, config: FloatArray) -> FloatArray:
        """Normalize the (density, friction) subset used as the network's global features."""
        return (config[:2] - self.config_mean) / self.config_std


def build_transitions(dataset: DemDataset) -> list[Transition]:
    """Every ``(state_t -> state_{t+1})`` transition across all configs in the fixture."""
    transitions: list[Transition] = []
    for c in range(dataset.n_configs):
        for t in range(dataset.n_steps):
            transitions.append(
                Transition(
                    state=dataset.states[c, t],
                    tool_x_m=float(dataset.tool_x[c, t]),
                    config=dataset.params[c],
                    next_state=dataset.states[c, t + 1],
                )
            )
    return transitions


def _acceleration(transition: Transition, dt_s: float) -> FloatArray:
    """The macro-step acceleration target ``(v_{t+1} - v_t) / dt`` (N, 2)."""
    return (transition.next_state[:, 2:] - transition.state[:, 2:]) / dt_s


def integrate(state: FloatArray, accel: FloatArray, dt_s: float) -> FloatArray:
    """Semi-implicit Euler: next velocity from ``accel``, next position from next velocity."""
    next_vel = state[:, 2:] + accel * dt_s
    next_pos = state[:, :2] + next_vel * dt_s
    return np.hstack([next_pos, next_vel]).astype(np.float64)


def fit_normalizer(transitions: list[Transition], dataset: DemDataset) -> Normalizer:
    """Compute standardization stats over the training transitions."""
    node_rows, accel_rows, configs = [], [], []
    for tr in transitions:
        node_rows.append(node_features(tr.state, tr.tool_x_m, tr.config[:2], dataset.bed_width_m))
        accel_rows.append(_acceleration(tr, dataset.dt_s))
        configs.append(tr.config[:2])
    nodes = np.vstack(node_rows)
    accels = np.vstack(accel_rows)
    cfg = np.vstack(configs)
    return Normalizer(
        node_mean=nodes.mean(0),
        node_std=nodes.std(0) + 1e-6,
        accel_mean=accels.mean(0),
        accel_std=accels.std(0) + 1e-6,
        config_mean=cfg.mean(0),
        config_std=cfg.std(0) + 1e-6,
    )


def _graph_tensors(
    state: FloatArray, tool_x_m: float, config: FloatArray, dataset: DemDataset, norm: Normalizer
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos = state[:, :2]
    edge_index = build_edges(pos, _CUTOFF_RADII * _PARTICLE_RADIUS_M)
    nf = node_features(state, tool_x_m, norm.config_norm(config), dataset.bed_width_m)
    nf = (nf - norm.node_mean) / norm.node_std
    ef = edge_features(pos, edge_index)
    return (
        torch.from_numpy(nf).float(),
        torch.from_numpy(edge_index),
        torch.from_numpy(ef).float(),
    )


@dataclass(frozen=True)
class _Batch:
    """All training transitions concatenated into one block-diagonal graph (fast full-batch GD)."""

    node_feat: torch.Tensor  # (sum_N, F) normalized
    edge_index: torch.Tensor  # (2, sum_E) with node offsets applied
    edge_feat: torch.Tensor  # (sum_E, EDGE_FEATURES)
    target: torch.Tensor  # (sum_N, 2) normalized acceleration


def _build_batch(
    transitions: list[Transition], dataset: DemDataset, normalizer: Normalizer
) -> _Batch:
    """Concatenate every transition into one disconnected-components graph for batched training.

    The radius graph is built from the (clean) positions once; noise injection perturbs the node
    *features* each epoch, so the edge structure is precomputed and reused — turning thousands of
    tiny per-transition passes into one full-batch pass per epoch.
    """
    node_rows, edge_cols, edge_rows, targets = [], [], [], []
    offset = 0
    cutoff = _CUTOFF_RADII * _PARTICLE_RADIUS_M
    for tr in transitions:
        pos = tr.state[:, :2]
        edge_index = build_edges(pos, cutoff)
        nf = node_features(
            tr.state, tr.tool_x_m, normalizer.config_norm(tr.config), dataset.bed_width_m
        )
        node_rows.append((nf - normalizer.node_mean) / normalizer.node_std)
        edge_rows.append(edge_features(pos, edge_index))
        edge_cols.append(edge_index + offset)
        targets.append(
            (_acceleration(tr, dataset.dt_s) - normalizer.accel_mean) / normalizer.accel_std
        )
        offset += pos.shape[0]
    return _Batch(
        node_feat=torch.from_numpy(np.vstack(node_rows)).float(),
        edge_index=torch.from_numpy(np.hstack(edge_cols)),
        edge_feat=torch.from_numpy(np.vstack(edge_rows)).float(),
        target=torch.from_numpy(np.vstack(targets)).float(),
    )


def train_ensemble(
    dataset: DemDataset,
    transitions: list[Transition],
    normalizer: Normalizer,
    config: TrainConfig,
    *,
    seed: int,
) -> list[GNS]:
    """Train ``ensemble_size`` GNS models (different seeds) with noise-injection/pushforward.

    Full-batch gradient descent over the block-diagonal graph of all transitions; each epoch
    re-draws pushforward noise on the (normalized) node features so the network is robust to its
    own rollout drift.
    """
    batch = _build_batch(transitions, dataset, normalizer)
    models: list[GNS] = []
    for member in range(config.ensemble_size):
        torch.manual_seed(seed + member)
        generator = torch.Generator().manual_seed(seed + member)
        model = GNS(hidden=config.hidden, message_passing_steps=config.message_passing_steps)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        model.train()
        for _ in range(config.epochs):
            noise = torch.randn(batch.node_feat.shape, generator=generator) * config.noise_std
            optimizer.zero_grad()
            pred = model(batch.node_feat + noise, batch.edge_index, batch.edge_feat)
            loss = torch.mean((pred - batch.target) ** 2)
            loss.backward()
            optimizer.step()
        model.eval()
        models.append(model)
    return models


def ensemble_predict(
    models: list[GNS],
    state: FloatArray,
    tool_x_m: float,
    config: FloatArray,
    dataset: DemDataset,
    normalizer: Normalizer,
) -> tuple[FloatArray, FloatArray]:
    """Predict ``(next_state, per-particle-channel std)`` from the ensemble.

    Each member predicts a normalized acceleration → denormalize → integrate to a next state; the
    ensemble **mean** is the prediction and the ensemble **std** over next-state channels is the
    epistemic uncertainty the conformal layer scales.
    """
    nf, ei, ef = _graph_tensors(state, tool_x_m, config, dataset, normalizer)
    predictions = []
    with torch.no_grad():
        for model in models:
            accel = model(nf, ei, ef).numpy() * normalizer.accel_std + normalizer.accel_mean
            predictions.append(integrate(state, accel, dataset.dt_s))
    stacked = np.stack(predictions)  # (K, N, 4)
    return stacked.mean(axis=0), stacked.std(axis=0)
