# SPDX-License-Identifier: Apache-2.0
"""The allocation-delegation contract — request, response, and the adapter that carries it.

Mind owns *decomposition and execution*; Allocate owns *who does what, when, where*, and Mind
embeds no combinatorial solver (charter §5.4; mind.md §6). The delegation between them crosses the
Core :class:`~astro_mine.core.policy.protocol.Allocator` sub-interface, and this module is
everything that crossing needs: the request/response values, the ``extras`` key the request travels
under, the optional reporter seam a solver implements to surface its structured result, and the
:class:`AllocationAdapter` that ties the three together.

**Why this lives at the waist.** It began in Mind, where its own docstring recorded the intent to
migrate it here (RFC-0006), and the migration is what dissolves the ``allocate -> mind`` edge.
Three parties construct this contract — Mind's reference stand-in, Allocate's real
``AllocationPlanner``, and Mind's executive — which by conventions.md §3.3 makes it Core's. The
practical confirmation is that every type it names was already Core's: :class:`AgentId`,
:class:`~astro_mine.core.messages.model.Observation`,
:class:`~astro_mine.core.messages.model.ActionBatch`,
:class:`~astro_mine.core.messages.model.ProspectTask`, :class:`DecisionContext`,
:class:`~astro_mine.core.policy.protocol.Policy`. Nothing about the move required a new Core
concept, which is the sign that the module was at the wrong address rather than doing the wrong
thing.

**Mechanism, not policy** (core.md §2.2). Nothing here decides what a good allocation is — that is
the injected solver's entire job. The adapter assembles, publishes, delegates, and records
provenance; :func:`assemble_request` reads Core's own ``ProspectTask`` off a Core ``ActionBatch``,
which is a statement about Core's schema rather than a domain opinion.

The DTOs are deliberately minimal: the coupled power/comms-window/terrain constraints and the rich
objective live on Allocate's own ``AllocationRequest``. Frozen values, so a delegated decision is
reproducible.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.model import ActionBatch, Observation, Vec3, Volume
from astro_mine.core.policy.model import DecisionContext
from astro_mine.core.policy.protocol import Policy

__all__ = [
    "ALLOCATION_REQUEST_KEY",
    "Allocation",
    "AllocationAdapter",
    "AllocationAsset",
    "AllocationProvenance",
    "AllocationReporter",
    "AllocationRequest",
    "AllocationTask",
    "Assignment",
    "assemble_request",
]

#: The ``DecisionContext.extras`` key the request is published under — the same convention
#: ``astro_mine.allocate.AllocationPlanner`` reads (``REQUEST_KEY``), so the real solver drops in.
ALLOCATION_REQUEST_KEY = "allocation.request"



@dataclass(frozen=True, slots=True)
class AllocationTask:
    """A task to assign — a prospect region to visit. ``task_id`` is stable within a request;
    ``region`` is the Core keep-in volume; ``center`` caches its centroid for scoring."""

    task_id: str
    region: Volume
    center: Vec3


@dataclass(frozen=True, slots=True)
class AllocationAsset:
    """An asset available for assignment: the agent and its current position."""

    agent_id: AgentId
    position: Vec3


@dataclass(frozen=True, slots=True)
class AllocationRequest:
    """The assembled allocation problem: the ``tasks`` to cover, the ``assets`` to cover them
    with, and an optional anytime ``deadline_s`` (accept a good-enough incumbent by then rather
    than block for optimal). ``request_id`` identifies the replan."""

    request_id: str
    tasks: tuple[AllocationTask, ...]
    assets: tuple[AllocationAsset, ...]
    deadline_s: float | None = None


@dataclass(frozen=True, slots=True)
class Assignment:
    """One asset↔task assignment produced by the solver."""

    agent_id: AgentId
    task_id: str


@dataclass(frozen=True, slots=True)
class AllocationProvenance:
    """The solver provenance Mind carries into its plan provenance (RM-P1-MIND-04): the
    ``solver`` that produced the assignment and the ``seed`` it recorded, so a delegated
    decision reproduces (Allocate RM-P1-ALLOC-07)."""

    solver: str
    seed: int | None = None


@dataclass(frozen=True, slots=True)
class Allocation:
    """The solver's answer: the ``assignments``, whether it is an ``incumbent`` (a good-enough
    anytime result rather than a proven optimum), and the ``provenance`` to reproduce it."""

    assignments: tuple[Assignment, ...]
    provenance: AllocationProvenance
    incumbent: bool = True
    by_agent: dict[AgentId, str] = field(default_factory=dict)


@runtime_checkable
class AllocationReporter(Protocol):
    """A solver that can surface its most-recent structured
    :class:`~astro_mine.core.plan.allocation.Allocation` (for provenance capture)."""

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
