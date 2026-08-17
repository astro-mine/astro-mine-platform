# SPDX-License-Identifier: Apache-2.0
"""Recurrent policy core for partial observability (RM-P1-LEARN-03; learn.md §3).

A single-step GRU cell used as the trunk of a recurrent
:class:`~astro_mine.learn.models.mlp.DictActorCritic`.
Under the SwarmEnv's observation masking (an agent that cannot see this tick receives the
neutral zero observation), a recurrent core lets the policy carry belief across masked
ticks — the natural architecture for the charter §8 partial-observability problem.
"""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["GRUCore"]


class GRUCore(nn.Module):
    """A one-step GRU cell mapping an input (and previous hidden state) to a new hidden
    feature, which doubles as the trunk feature."""

    def __init__(self, in_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = nn.GRUCell(in_dim, hidden_size)

    def initial_state(self, batch: int = 1) -> torch.Tensor:
        return torch.zeros(batch, self.hidden_size, dtype=torch.float32)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        if hidden is None:
            hidden = self.initial_state(x.shape[0])
        out: torch.Tensor = self.cell(x, hidden)
        return out
