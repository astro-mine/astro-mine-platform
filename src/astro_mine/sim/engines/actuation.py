# SPDX-License-Identifier: Apache-2.0
"""Action dispatch — routing an ``ActionBatch`` into the engines (RM-P0-SIM-03).

The stepping core hands each step's :class:`~astro_mine.core.messages.model.ActionBatch`
to every live engine through :meth:`~astro_mine.sim.engines.adapter.RegimeEngine.apply_actions`
*before* it advances them. Each engine actuates only the agents it owns and ignores the
rest, so the same batch can fan out across a heterogeneous set of engines (the multi-engine
co-step is RM-P0-SIM-04). This module is the small shared helper an engine uses to pick out
the action addressed to one of its agents.

An :class:`~astro_mine.core.messages.model.Action` is a tagged union over
:class:`~astro_mine.core.messages.enums.ActionKind` (``actuator`` / ``mode`` / ``task``);
how an engine interprets each kind is the engine's business — the mobility engine reads a
``goto`` task, the manipulation engine reads ``actuator`` joint setpoints, and so on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astro_mine.core.messages.model import Action, ActionBatch

__all__ = ["actions_by_agent"]


def actions_by_agent(batch: ActionBatch) -> dict[str, Action]:
    """Index a batch by ``agent_id`` (one decision per agent; a later entry wins).

    Engines call this once per step and look up only the agents they own, so an empty
    batch — or a batch addressed to other engines' agents — is a no-op for this engine."""
    return {action.agent_id: action for action in batch.actions}
