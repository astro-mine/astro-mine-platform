"""Centralized value modules for CTDE (RM-P1-LEARN-03; learn.md §3, §11).

The training-time value functions that consume the global information a decentralized actor
does *not* see at execution — the honest CTDE bargain:

- :class:`CentralizedCritic` — MAPPO's critic, a scalar value over the global
  ``SwarmEnv.state()`` vector (declared by a
  :class:`~astro_mine.learn.algos._contract.CentralizedCriticSpec`).
- :class:`VDNMixer` / :class:`QMixer` — QMIX's mixers, combining per-agent chosen Q-values
  into a joint action-value. VDN sums; QMIX uses a monotonic hypernetwork conditioned on the
  global state, preserving the ``argmax`` consistency that makes decentralized execution
  sound.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from astro_mine.learn.algos._contract import CentralizedCriticSpec
from astro_mine.learn.models.mlp import MLP

__all__ = ["CentralizedCritic", "QMixer", "VDNMixer"]


class CentralizedCritic(nn.Module):
    """MAPPO's centralized critic: a scalar value over the global ``state()`` vector."""

    def __init__(self, spec: CentralizedCriticSpec, hidden_sizes: Sequence[int]) -> None:
        super().__init__()
        self.spec = spec
        self.norm = nn.LayerNorm(spec.global_state_dim)
        self.net = MLP(spec.global_state_dim, hidden_sizes, 1)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        value: torch.Tensor = self.net(self.norm(state)).squeeze(-1)
        return value


class VDNMixer(nn.Module):
    """Additive value decomposition: the joint Q is the sum of per-agent chosen Q-values."""

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # agent_qs: (batch, n_agents); state unused (kept for a uniform mixer signature).
        total: torch.Tensor = agent_qs.sum(dim=1)
        return total


class QMixer(nn.Module):
    """Monotonic QMIX mixer: a hypernetwork over the global state produces non-negative
    weights so ``argmax`` per agent stays consistent with the joint ``argmax``."""

    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 32) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim
        self.norm = nn.LayerNorm(state_dim)
        self.hyper_w1 = nn.Linear(state_dim, n_agents * embed_dim)
        self.hyper_w2 = nn.Linear(state_dim, embed_dim)
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1)
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        batch = agent_qs.shape[0]
        state = self.norm(state)
        w1 = torch.abs(self.hyper_w1(state)).view(batch, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(batch, 1, self.embed_dim)
        hidden = torch.relu(torch.bmm(agent_qs.view(batch, 1, self.n_agents), w1) + b1)
        w2 = torch.abs(self.hyper_w2(state)).view(batch, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(batch, 1, 1)
        out = torch.bmm(hidden, w2) + b2
        return out.view(batch)
