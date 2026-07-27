"""The Allocate delegation payload (RM-P1-MIND-04).

The request/response the mission tier hands across the Core
:class:`~astro_mine.core.policy.protocol.Allocator` sub-interface. These are **Mind-owned**
delegation DTOs — deliberately minimal (the coupled power/comms-window/terrain constraints and
the rich objective live on Allocate's own ``AllocationRequest``) — and documented to migrate to a
Core-owned allocation message schema (astro-mine-core RFC-0006), mirroring how RM-P1-MIND-01
localized ``ReplanTrigger``/``BeliefView``. Frozen values, so a delegated decision is
reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.model import Vec3, Volume

__all__ = [
    "Allocation",
    "AllocationAsset",
    "AllocationProvenance",
    "AllocationRequest",
    "AllocationTask",
    "Assignment",
]


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
