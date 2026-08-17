# SPDX-License-Identifier: Apache-2.0
"""Policy/value/comms model building blocks shared across the baselines (learn.md §3).

- :class:`MLP` / :class:`DictActorCritic` / :class:`AgentQNet` — feed-forward trunk, the
  heterogeneous-action actor-critic (IPPO/MAPPO), and the discrete Q-net (QMIX);
- :class:`GRUCore` — a recurrent core for partial observability;
- :class:`CentralizedCritic` / :class:`VDNMixer` / :class:`QMixer` — CTDE value modules over
  the global ``SwarmEnv.state()``;
- :class:`MessageModule` / :class:`CommsEncoder` — learned messages over the CommsModel
  channel, per-agent-projected and reachable-peer-aggregated (the comms-learning track).
"""

from __future__ import annotations

from astro_mine.learn.models.comms import CommsEncoder, MessageModule, reach_matrix
from astro_mine.learn.models.critics import CentralizedCritic, QMixer, VDNMixer
from astro_mine.learn.models.mlp import MLP, ActorOutput, AgentQNet, DictActorCritic
from astro_mine.learn.models.rnn import GRUCore

__all__ = [
    "MLP",
    "ActorOutput",
    "AgentQNet",
    "CentralizedCritic",
    "CommsEncoder",
    "DictActorCritic",
    "GRUCore",
    "MessageModule",
    "QMixer",
    "VDNMixer",
    "reach_matrix",
]
