# SPDX-License-Identifier: Apache-2.0
"""Policy/Planner API v0.1 — the composition seam (RM-P0-CORE-03).

The minimal mechanism that lets the tiers compose *under one policy object without
interface changes* (the acceptance criterion): :class:`ComposedPolicy` runs its stages
in order, threading each stage's :class:`~astro_mine.core.messages.ActionBatch` into the
next stage's :attr:`DecisionContext.upstream` — so an allocator's assignments feed a
controller, all behind the single :class:`Policy` contract.

This is only the *seam*. The rich hierarchy — behaviour-tree executive, fallback
branches, replan triggers, validity horizons — is Mind's (Phase 1); Core ships just
enough that every Phase-1 layer composes through a stable signature.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.model import ActionBatch, Observation
from astro_mine.core.policy.model import DecisionContext
from astro_mine.core.policy.protocol import Policy

__all__ = ["ComposedPolicy"]


class ComposedPolicy:
    """Sequentially composes policies into one :class:`Policy`.

    Each stage decides given the observations and a context whose ``upstream`` is the
    prior stage's output (seeded from the incoming ``context.upstream``); the final
    stage's batch is returned. A controller and an allocator thus compose under one
    object — ``ComposedPolicy(allocator, controller)`` — with no interface change.
    """

    def __init__(self, *stages: Policy) -> None:
        if not stages:
            raise ValueError("ComposedPolicy requires at least one stage")
        self._stages = stages

    @property
    def stages(self) -> tuple[Policy, ...]:
        return self._stages

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        result = context.upstream if context.upstream is not None else ActionBatch()
        for stage in self._stages:
            result = stage.decide(observations, replace(context, upstream=result))
        return result
