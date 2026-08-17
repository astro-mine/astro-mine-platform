# SPDX-License-Identifier: Apache-2.0
"""Comms-limited / partial-observability env wrappers (RM-P1-LEARN-02).

The declarative :class:`CommsModel` — line-of-sight/range gating, message bandwidth budget,
stochastic drop, and fixed/sampled delay — modelled *in the environment* so the constraint
is identical across every algorithm (learn.md §2, §3; charter §8). Compose it with a
:class:`~astro_mine.learn.envs.adapter.parallel_env.SwarmEnv` via ``comms_model=``.
"""

from __future__ import annotations

from astro_mine.learn.envs.comms.config import (
    BandwidthBudgetConfig,
    CommsModelConfig,
    DelayConfig,
    DropConfig,
    RangeGateConfig,
)
from astro_mine.learn.envs.comms.ledger import CommsLedger, LinkTally
from astro_mine.learn.envs.comms.model import CommsModel

__all__ = [
    "BandwidthBudgetConfig",
    "CommsLedger",
    "CommsModel",
    "CommsModelConfig",
    "DelayConfig",
    "DropConfig",
    "LinkTally",
    "RangeGateConfig",
]
