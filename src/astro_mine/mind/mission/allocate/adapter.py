"""The Allocate delegation adapter (RM-P1-MIND-04).

:class:`AllocationAdapter` is the ``mission/allocate/`` thin adapter: it assembles an
:class:`~astro_mine.mind.mission.allocate.model.AllocationRequest` from the mission tier's
decomposition (its ``DecisionContext.upstream``), publishes it under the shared
``allocation.request`` extras key, and delegates to an injected solver policy through the Core
:class:`~astro_mine.core.policy.protocol.Allocator` sub-interface — mapping the returned
assignment back into per-agent roles/regions for TAMP. It embeds **no** solver of its own; the
combinatorics are Allocate's (the reference stand-in or the real ``AllocationPlanner``). When the
solver reports a structured :class:`~astro_mine.mind.mission.allocate.model.Allocation` (via the
optional :class:`AllocationReporter` seam), the adapter carries its solver+seed provenance so a
delegated decision reproduces (RM-P1-MIND-04/07).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Protocol, runtime_checkable

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.model import ActionBatch, Observation
from astro_mine.core.policy.model import DecisionContext
from astro_mine.core.policy.protocol import Policy
from astro_mine.mind.mission.allocate.model import (
    Allocation,
    AllocationAsset,
    AllocationRequest,
    AllocationTask,
)

__all__ = ["ALLOCATION_REQUEST_KEY", "AllocationAdapter", "AllocationReporter", "assemble_request"]

#: The ``DecisionContext.extras`` key the request is published under — the same convention
#: ``astro_mine.allocate.AllocationPlanner`` reads (``REQUEST_KEY``), so the real solver drops in.
ALLOCATION_REQUEST_KEY = "allocation.request"


@runtime_checkable
class AllocationReporter(Protocol):
    """A solver that can surface its most-recent structured
    :class:`~astro_mine.mind.mission.allocate.model.Allocation` (for provenance capture)."""

    def allocation(self) -> Allocation | None: ...


def assemble_request(
    observations: Mapping[AgentId, Observation],
    upstream: ActionBatch,
    *,
    deadline_s: float | None = None,
) -> AllocationRequest:
    """Assemble an :class:`AllocationRequest` from the mission decomposition + observations."""
    tasks = tuple(
        AllocationTask(
            task_id=f"t{index}",
            region=action.task.prospect.region,
            center=action.task.prospect.region.center_m,
        )
        for index, action in enumerate(upstream.actions)
        if action.task is not None and action.task.prospect is not None
    )
    assets = tuple(
        AllocationAsset(
            agent_id=agent_id, position=observations[agent_id].self_state.pose.translation_m
        )
        for agent_id in sorted(observations)
    )
    return AllocationRequest(
        request_id="mission-decomposition-replan", tasks=tasks, assets=assets, deadline_s=deadline_s
    )


class AllocationAdapter:
    """Delegates assignment to an injected ``solver`` (a Core Allocator) and returns roles."""

    def __init__(self, solver: Policy, *, deadline_s: float | None = None) -> None:
        self._solver = solver
        self._deadline_s = deadline_s
        self._last_allocation: Allocation | None = None

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        upstream = context.upstream if context.upstream is not None else ActionBatch()
        request = assemble_request(observations, upstream, deadline_s=self._deadline_s)
        solver_context = replace(
            context, extras={**dict(context.extras), ALLOCATION_REQUEST_KEY: request}
        )
        assignment = self._solver.decide(observations, solver_context)
        if isinstance(self._solver, AllocationReporter):
            self._last_allocation = self._solver.allocation()
        return assignment

    def allocation(self) -> Allocation | None:
        """The structured allocation from the most recent delegation (for plan provenance)."""
        return self._last_allocation
