"""Hand-rolled message-passing GNN — the learned-DEM particle simulator (RM-P1-SURR-02).

A GNS/MeshGraphNet-style encoder-processor-decoder over a particle radius-graph, implemented
directly in PyTorch (no torch_geometric): it predicts a per-particle acceleration the surrogate
integrates to the next state. Physics-informed inductive biases: edges carry **relative**
position (translation invariance), the target is an **acceleration** (frame-consistent), and the
message/update MLPs share weights across particles (permutation equivariance) — surrogate.md §2.6.

torch lives here and in the sibling training/serving modules only; the contract layer
(:mod:`astro_mine.surrogate.model`) never imports it.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

__all__ = [
    "EDGE_FEATURES",
    "GNS",
    "NODE_FEATURES",
    "OUTPUT_DIM",
    "build_edges",
    "edge_features",
    "node_features",
]

FloatArray = npt.NDArray[np.float64]

#: Per-particle node features: velocity (2) + signed distances to floor/left-wall/right-wall/blade
#: (4) + broadcast excavation config (density, friction) (2).
NODE_FEATURES = 8
#: Per-edge features: relative position (2) + separation distance (1).
EDGE_FEATURES = 3
#: Network output: a 2-D per-particle acceleration.
OUTPUT_DIM = 2


def build_edges(pos: FloatArray, cutoff_m: float) -> FloatArray:
    """A directed radius graph ``edge_index`` ``(2, E)`` — ``[src, dst]`` for ``|xi-xj| < cutoff``.

    ``src`` is the neighbour (sender ``j``), ``dst`` the receiver ``i``; self-edges are excluded.
    """
    diff = pos[:, None, :] - pos[None, :, :]
    dist = np.sqrt(np.square(diff).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    src, dst = np.where(dist < cutoff_m)
    return np.stack([src, dst]).astype(np.int64)


def node_features(
    state: FloatArray, tool_x_m: float, config_norm: FloatArray, bed_width_m: float
) -> FloatArray:
    """Per-particle node features ``(N, NODE_FEATURES)`` from a particle state.

    ``state`` is ``(N, 4)`` = ``(pos_x, pos_z, vel_x, vel_z)``; ``config_norm`` is the normalized
    ``(density, friction)`` broadcast to every particle so the network is conditioned on the soil.
    Boundary features (distance to floor/walls/blade) let the network learn the wall/tool forces.
    """
    pos, vel = state[:, :2], state[:, 2:]
    boundary = np.column_stack(
        [pos[:, 1], pos[:, 0], bed_width_m - pos[:, 0], pos[:, 0] - tool_x_m]
    )
    globals_ = np.tile(config_norm, (state.shape[0], 1))
    return np.hstack([vel, boundary, globals_]).astype(np.float64)


def edge_features(pos: FloatArray, edge_index: FloatArray) -> FloatArray:
    """Per-edge features ``(E, EDGE_FEATURES)`` = relative position + distance.

    Relative (not absolute) position is the translation-invariance inductive bias.
    """
    src, dst = edge_index[0], edge_index[1]
    rel = pos[dst] - pos[src]
    dist = np.sqrt(np.square(rel).sum(axis=-1, keepdims=True))
    return np.hstack([rel, dist]).astype(np.float64)


class _MLP(nn.Module):  # type: ignore[misc]  # nn.Module is Any under the torch mypy override
    """A small ReLU MLP with a final linear layer."""

    def __init__(self, sizes: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            if i < len(sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor = self.net(x)
        return out


class GNS(nn.Module):  # type: ignore[misc]  # nn.Module is Any under the torch mypy override
    """Encoder-processor-decoder graph net predicting a per-particle acceleration.

    ``hidden`` is the latent width; ``message_passing_steps`` the number of processor rounds. Kept
    small — the reference bed is tiny and CI trains on CPU.
    """

    def __init__(self, *, hidden: int = 32, message_passing_steps: int = 2) -> None:
        super().__init__()
        self.steps = message_passing_steps
        self.node_encoder = _MLP([NODE_FEATURES, hidden, hidden])
        self.edge_encoder = _MLP([EDGE_FEATURES, hidden, hidden])
        self.message_mlps = nn.ModuleList(
            _MLP([3 * hidden, hidden, hidden]) for _ in range(message_passing_steps)
        )
        self.update_mlps = nn.ModuleList(
            _MLP([2 * hidden, hidden, hidden]) for _ in range(message_passing_steps)
        )
        self.decoder = _MLP([hidden, hidden, OUTPUT_DIM])

    def forward(self, node_feat: Tensor, edge_index: Tensor, edge_feat: Tensor) -> Tensor:
        """Predict the normalized per-particle acceleration ``(N, 2)``."""
        h = self.node_encoder(node_feat)
        e = self.edge_encoder(edge_feat)
        src, dst = edge_index[0], edge_index[1]
        n_nodes = node_feat.shape[0]
        for step in range(self.steps):
            messages = self.message_mlps[step](torch.cat([h[src], h[dst], e], dim=-1))
            aggregated = torch.zeros(n_nodes, h.shape[1], dtype=h.dtype)
            aggregated.index_add_(0, dst, messages)
            h = h + self.update_mlps[step](torch.cat([h, aggregated], dim=-1))  # residual update
        out: Tensor = self.decoder(h)
        return out
