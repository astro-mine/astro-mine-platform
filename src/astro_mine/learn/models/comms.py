# SPDX-License-Identifier: Apache-2.0
"""Comms-learning message modules over the CommsModel channel (RM-P1-LEARN-03; learn.md §11).

The first-class research-track entry point: because *comms-limited cooperation is the
charter §8 problem*, learned messages must ride the **same** gated/dropped/delayed channel
the :class:`~astro_mine.learn.envs.CommsModel` imposes, so a comms-learning result is
comparable to a comms-blind baseline on the Bench leaderboard.

:class:`MessageModule` encodes a per-agent message from local features and aggregates the
messages of the peers that are *reachable this tick* — the reachability mask taken from
``infos[agent]["comms"]`` (the CommsModel's post-gate/drop/delay verdict), never from a
side channel. A message that the channel dropped simply does not arrive; the aggregate over
an isolated agent is zero. This keeps the comms constraint identical across algorithms
(learn.md §2, §3).

:class:`CommsEncoder` is the swarm-level module the registered comms-learning baseline
(``comms_ppo``; :mod:`astro_mine.learn.algos.comms_ppo`) trains: SADF agents are
*heterogeneous* (different observation widths), so each agent owns a linear projection into
one shared message-feature width before the single shared :class:`MessageModule` encodes and
mean-pools the peer messages that actually arrived. The aggregate is **differentiable** — it
is recomputed inside the PPO update from the recorded reachability, so the message encoder
receives real gradients from the team objective (a differentiable-message CTDE variant).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from astro_mine.core.env.model import AgentId

__all__ = ["CommsEncoder", "MessageModule", "reach_matrix"]


def reach_matrix(
    agents: Sequence[AgentId],
    reach: Mapping[AgentId, tuple[AgentId, ...]],
    live: Sequence[AgentId] | None = None,
) -> NDArray[np.float32]:
    """The row-normalizable reachability mask for one tick — Torch-free.

    ``reach`` is the executor's recorded per-agent reachable-peer verdict (the peers whose
    message *arrived* this tick, straight off ``infos[agent]["comms"]``). Entry ``(i, j)`` is
    ``1.0`` iff agent ``j``'s message reaches agent ``i``. Rows/columns of agents that are not
    live this tick stay zero, so a departed agent neither sends nor receives — the same
    "an isolated agent receives the zero vector" semantics :class:`MessageModule` implements.
    """
    index = {agent: i for i, agent in enumerate(agents)}
    present = set(agents if live is None else live)
    mask = np.zeros((len(agents), len(agents)), dtype=np.float32)
    for agent, peers in reach.items():
        if agent not in index or agent not in present:
            continue
        for peer in peers:
            if peer in index and peer in present:
                mask[index[agent], index[peer]] = 1.0
    return mask


class MessageModule(nn.Module):
    """Encode + reachable-peer-aggregate learned messages over the CommsModel channel.

    ``forward`` takes stacked per-agent local features ``(..., n_agents, feat_dim)`` and a
    row-normalizable reachability mask ``(..., n_agents, n_agents)`` (1 where the column
    agent's message reaches the row agent this tick) and returns the aggregated peer-message
    context ``(..., n_agents, msg_dim)`` — a masked mean-pool, so an unreachable peer
    contributes nothing and an isolated agent receives the zero vector. A leading batch
    dimension (one entry per vectorized env copy / rollout step) is broadcast over."""

    def __init__(self, feat_dim: int, msg_dim: int) -> None:
        super().__init__()
        self.msg_dim = msg_dim
        self.encode = nn.Linear(feat_dim, msg_dim)

    def forward(self, features: torch.Tensor, reach_mask: torch.Tensor) -> torch.Tensor:
        messages = torch.tanh(self.encode(features))
        # Sum over the *peer* (column) axis, so the mean-pool works unchanged for a single
        # tick (n, n) and for a batch of ticks/env copies (batch, n, n).
        denom = reach_mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
        aggregate: torch.Tensor = torch.matmul(reach_mask, messages) / denom
        return aggregate


class CommsEncoder(nn.Module):
    """The swarm-level learned-message channel for heterogeneous agents (comms-learning).

    Each agent projects its own (differently-sized) flat observation into one shared
    message-feature width; the single shared :class:`MessageModule` then encodes one message
    per agent and mean-pools the messages that *reached* each agent this tick. The result is
    the per-agent peer-message context that widens the actor's trunk
    (:class:`~astro_mine.learn.models.mlp.DictActorCritic`'s ``comms_dim`` knob).

    An agent absent from ``obs`` (departed, or not live this tick) contributes the zero
    feature and must have a zero mask row/column (:func:`reach_matrix` guarantees this).

    Each agent's observation is **LayerNorm**'d before it is projected, for the same reason
    :class:`~astro_mine.learn.models.mlp.DictActorCritic` normalizes its trunk: the SwarmEnv's
    observation is raw SI (pose in metres, battery in *joules* ~1e5), and feeding that
    unnormalized into a ``tanh`` projection saturates it — the message encoder would receive a
    vanishing gradient and never learn what to say."""

    def __init__(self, obs_dims: Mapping[AgentId, int], *, feat_dim: int, msg_dim: int) -> None:
        super().__init__()
        self.agents: tuple[AgentId, ...] = tuple(obs_dims)
        self.feat_dim = feat_dim
        self.msg_dim = msg_dim
        # ModuleList (not ModuleDict) keyed positionally: an AgentId is an arbitrary string and
        # ModuleDict forbids '.' in keys.
        self.obs_norm = nn.ModuleList([nn.LayerNorm(obs_dims[agent]) for agent in self.agents])
        self.project = nn.ModuleList(
            [nn.Linear(obs_dims[agent], feat_dim) for agent in self.agents]
        )
        self.messages = MessageModule(feat_dim, msg_dim)

    def forward(
        self, obs: Mapping[AgentId, torch.Tensor], reach: torch.Tensor
    ) -> dict[AgentId, torch.Tensor]:
        """Per-agent aggregated peer-message context.

        ``obs`` maps each *live* agent to its flat observation ``(batch, obs_dim)``; ``reach``
        is the ``(batch, n_agents, n_agents)`` mask. Returns one ``(batch, msg_dim)`` context
        per agent in :attr:`agents` (zero for an isolated or absent agent)."""
        batch = reach.shape[0]
        features = []
        for i, agent in enumerate(self.agents):
            row = obs.get(agent)
            if row is None:
                features.append(torch.zeros(batch, self.feat_dim, dtype=torch.float32))
            else:
                features.append(torch.tanh(self.project[i](self.obs_norm[i](row))))
        aggregate = self.messages(torch.stack(features, dim=1), reach)
        return {agent: aggregate[:, i, :] for i, agent in enumerate(self.agents)}
