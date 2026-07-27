"""Allocate delegation (RM-P1-MIND-04).

The thin adapter that delegates the coupled task-allocation/scheduling problem to
Astro-Mine-Allocate and turns the returned assignment into roles/regions for TAMP. Mind owns
*decomposition and execution*; Allocate owns *who does what, when, where* — Mind embeds **no**
combinatorial solver (the boundary, charter §5.4; mind.md §6).

The delegation crosses the Core :class:`~astro_mine.core.policy.protocol.Allocator` sub-interface:
:class:`~astro_mine.mind.mission.allocate.adapter.AllocationAdapter` assembles an
:class:`~astro_mine.mind.mission.allocate.model.AllocationRequest` from the mission
decomposition, publishes it under the shared ``allocation.request`` extras key, and delegates to
an injected solver policy — resolved through the registry, never a sibling import. Base Mind
ships the deterministic ``GreedyReferenceAllocator``
as the delegated-to stand-in (as MIND-01 stood in for Guard with the pass-through shield); the
real ``astro_mine.allocate.AllocationPlanner`` binds via the companion entry-point shim behind
the same adapter. The request/response DTOs are Mind-owned for now and documented to migrate to a
Core-owned allocation schema (astro-mine-core RFC-0006).
"""

from __future__ import annotations

from astro_mine.mind.mission.allocate.adapter import (
    ALLOCATION_REQUEST_KEY,
    AllocationAdapter,
    assemble_request,
)
from astro_mine.mind.mission.allocate.model import (
    Allocation,
    AllocationAsset,
    AllocationProvenance,
    AllocationRequest,
    AllocationTask,
    Assignment,
)

__all__ = [
    "ALLOCATION_REQUEST_KEY",
    "Allocation",
    "AllocationAdapter",
    "AllocationAsset",
    "AllocationProvenance",
    "AllocationRequest",
    "AllocationTask",
    "Assignment",
    "assemble_request",
]
