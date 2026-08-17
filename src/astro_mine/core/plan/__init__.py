# SPDX-License-Identifier: Apache-2.0
"""Plan / ContingentPlan — the Core-owned delay-tolerant plan contract (RFC-0006).

A first-class Core message schema (mind.md §5): the time-stamped, validity-horizoned plan the
autonomy stack acts on — Mind composes it, Ops replays it, View renders it, Bench scores it.
Schema only; planning (Mind) and evaluation (Bench/Ops) live above Core.

The canonical schema is ``schema/plan.schema.json`` (shipped in-package); the typed models live in
:mod:`astro_mine.core.plan.model`. Public API: :func:`load_plan` / :func:`validate_plan` (parse +
validate, structural + semantic) and :func:`load_schema` (the canonical JSON Schema as a dict).
The Protobuf wire form is deferred to the first cross-process consumer (RFC-0006, P2).

RFC-0006 also reserved this package for the Core-owned *allocation* schema Mind's delegation
DTOs were documented to migrate into; :mod:`astro_mine.core.plan.allocation` is that migration.
"""

from __future__ import annotations

from astro_mine.core.plan import allocation, loader, model
from astro_mine.core.plan.allocation import (
    ALLOCATION_REQUEST_KEY,
    Allocation,
    AllocationAdapter,
    AllocationAsset,
    AllocationProvenance,
    AllocationReporter,
    AllocationRequest,
    AllocationTask,
    Assignment,
    assemble_request,
)
from astro_mine.core.plan.loader import (
    PlanError,
    PlanValidationError,
    load_plan,
    load_schema,
    validate_plan,
)
from astro_mine.core.plan.model import (
    PLAN_VERSION,
    Assumption,
    ContingencyBranch,
    ContingentPlan,
    Plan,
    PlanDocument,
    PlanProvenance,
    PlanValidity,
)

__all__ = [
    "ALLOCATION_REQUEST_KEY",
    "PLAN_VERSION",
    "Allocation",
    "AllocationAdapter",
    "AllocationAsset",
    "AllocationProvenance",
    "AllocationReporter",
    "AllocationRequest",
    "AllocationTask",
    "Assignment",
    "Assumption",
    "ContingencyBranch",
    "ContingentPlan",
    "Plan",
    "PlanDocument",
    "PlanError",
    "PlanProvenance",
    "PlanValidationError",
    "PlanValidity",
    "allocation",
    "assemble_request",
    "load_plan",
    "load_schema",
    "loader",
    "model",
    "validate_plan",
]
