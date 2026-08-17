# SPDX-License-Identifier: Apache-2.0
"""Environment wrappers — the Core Environment API presented as RL envs (learn.md §3).

Learn turns a simulatable Core world into trainable RL problems by wrapping the Core
:class:`~astro_mine.core.env.protocol.Environment` contract as a PettingZoo
``ParallelEnv`` (multi-agent) and a Gymnasium ``Env`` (single-agent / centralized). The
adapter reaches the world only through ``astro_mine.core`` — the narrow waist — and
derives per-agent, SADF-capability-keyed spaces from the assets' Core-typed
:class:`~astro_mine.core.sadf.model.Asset` documents.

See :mod:`astro_mine.learn.envs.adapter` for the full public surface.
"""

from __future__ import annotations

from astro_mine.learn.envs.adapter import (
    CentralizedEnv,
    SingleAgentEnv,
    SwarmEnv,
    comms_channel,
    make_swarm_env,
    observation_mask,
)
from astro_mine.learn.envs.comms import (
    BandwidthBudgetConfig,
    CommsLedger,
    CommsModel,
    CommsModelConfig,
    DelayConfig,
    DropConfig,
    RangeGateConfig,
)

__all__ = [
    "BandwidthBudgetConfig",
    "CentralizedEnv",
    "CommsLedger",
    "CommsModel",
    "CommsModelConfig",
    "DelayConfig",
    "DropConfig",
    "RangeGateConfig",
    "SingleAgentEnv",
    "SwarmEnv",
    "comms_channel",
    "make_swarm_env",
    "observation_mask",
]
