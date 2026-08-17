# SPDX-License-Identifier: Apache-2.0
"""Reference symbolic task planner (RM-P1-MIND-03).

Reads the mission tier's decomposition from ``DecisionContext.upstream`` and selects each
agent's tactical target: the centre of its assigned prospect region, or hold-position when the
tier above gave it nothing. Pure and deterministic — the symbolic proposal the motion planner
then checks for feasibility (PDDLStream-style interleaving).
"""

from __future__ import annotations

from collections.abc import Mapping

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.model import ActionBatch, Observation, Vec3

__all__ = ["ReferenceTaskPlanner"]


class ReferenceTaskPlanner:
    """Selects each agent's GOTO target from the mission decomposition."""

    def targets(
        self, observations: Mapping[AgentId, Observation], upstream: ActionBatch
    ) -> dict[AgentId, Vec3]:
        """Per-agent target position: the assigned region centre, else hold at current pose."""
        by_agent = {action.agent_id: action for action in upstream.actions}
        targets: dict[AgentId, Vec3] = {}
        for agent_id in sorted(observations):
            assigned = by_agent.get(agent_id)
            if (
                assigned is not None
                and assigned.task is not None
                and assigned.task.prospect is not None
            ):
                targets[agent_id] = assigned.task.prospect.region.center_m
            else:
                targets[agent_id] = observations[agent_id].self_state.pose.translation_m
        return targets
