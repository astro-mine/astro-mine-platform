# SPDX-License-Identifier: Apache-2.0
"""``AllocationRequest`` / ``Allocation`` — the solver-neutral canonical model (RM-P1-ALLOC-01).

The Core-typed input/output of the allocation sub-interface (allocate.md §3, "Key
abstractions"). An :class:`AllocationRequest` states *who is available* (:class:`AssetRef`\\ s
with SADF capability tags + budgets), *what must be done* (:class:`Task`\\ s with a Core
:class:`~astro_mine.core.messages.enums.TaskKind`, location, time windows, precedence, and
uncertain value), *toward what* (:class:`Objective`), and *within what budget*
(:class:`SolveBudget`). An :class:`Allocation` returns *who does what, when* (per-asset
time-ordered :class:`ScheduledTask`\\ s), the realized objective, optimality gap, binding
constraints, full reproducibility provenance, and a feasibility/optimality status — or, on
infeasibility, an explicit certificate slot (populated by RM-P1-ALLOC-06).

Every wire model is frozen and ``extra="forbid"`` (immutable, content-addressed, fails
loud on a typo'd field) and every quantity is SI (conventions.md §5). The upstream
constraint *truth* — the Link contact graph, Worlds traversability, Fleet SADF budgets,
Prospect value — is threaded through the :class:`ConstraintContext`, a frozen container of
**Core-typed handles** delivered via the Core ``DecisionContext.extras`` channel and
consumed through Core contracts, never sibling imports (allocate.md §6; the narrow waist).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astro_mine.allocate.enums import AllocationStatus, ConstraintKind, ObjectiveSense
from astro_mine.core.hashing import content_hash_json
from astro_mine.core.messages.enums import TaskKind
from astro_mine.core.messages.model import ContactPlan, Volume
from astro_mine.core.objective.model import ObjectiveSpec
from astro_mine.core.resource.protocol import ResourceField
from astro_mine.core.sadf import Asset, CapabilityTag
from astro_mine.core.world.protocol import WorldProvider

__all__ = [
    "Allocation",
    "AllocationProvenance",
    "AllocationRequest",
    "AssetRef",
    "AssetSchedule",
    "BindingConstraint",
    "ConstraintContext",
    "InfeasibilityCertificate",
    "Objective",
    "ObjectiveContribution",
    "ObjectiveDecomposition",
    "ScheduledTask",
    "SolveBudget",
    "Task",
    "TimeWindow",
    "ValueEstimate",
]


#: Absolute floor for the models' float comparisons — matches the IR verifier's epsilon, so a
#: schedule the verifier accepts is never rejected by the response type (and vice versa).
_EPS = 1.0e-9


class _Model(BaseModel):
    """Base for every request/response model: immutable, reject unknown fields loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    def content_hash(self) -> str:
        """The ``sha256:<hex>`` content address of this model (its content-addressed identity).

        Over the canonical JSON of the model — the platform's one content-address primitive
        (:func:`astro_mine.core.hashing.content_hash_json`) — so a plan can pin the exact
        request it solved and identical inputs hash identically across machines.
        """
        return content_hash_json(self.model_dump(mode="json"))


# --- request side ----------------------------------------------------------------


class ValueEstimate(_Model):
    """An uncertain scalar value: a ``mean`` with an optional ``variance``.

    Value and durations carry distributions, not point guesses (allocate.md §2 principle 7);
    the RM-P1-ALLOC-01 objective uses the ``mean``, and robust/stochastic formulations
    (RM-P1-ALLOC later) consume the ``variance``.
    """

    mean: float
    variance: float | None = Field(default=None, ge=0.0)


class TimeWindow(_Model):
    """A closed ``[start_s, end_s]`` availability window in SI seconds of episode/sim time.

    A task may declare **several** windows (an orbiter over the horizon twice, a PSR lit only
    between two shadow passes). They are genuinely *disjoint alternatives*: the compiler encodes
    them as an exact disjunction (a ``WINDOW_SELECT`` variable per window), never as one
    ``[min(start), max(end)]`` envelope — an envelope would silently admit a start inside an
    availability **gap** (RM-P1-ALLOC-02).
    """

    start_s: float
    end_s: float

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end_s < self.start_s:
            raise ValueError(f"time window end_s ({self.end_s}) precedes start_s ({self.start_s})")
        return self

    def contains(self, start_s: float, *, eps: float = 1.0e-9) -> bool:
        """Whether ``start_s`` falls inside this closed window (within a float epsilon)."""
        return self.start_s - eps <= start_s <= self.end_s + eps


class Task(_Model):
    """One task to be assigned: a Core ``TaskKind``, where, when, after-what, and worth-what.

    ``kind`` is the Core task vocabulary (prospect/excavate/haul/…); ``location`` is an
    optional Core :class:`~astro_mine.core.messages.model.Volume` target region;
    ``resource_target_ref`` a content reference to the Prospect resource field/target the
    task acts on. ``required_capabilities`` are the Core SADF capability tags an asset must
    declare to be *eligible* — the task↔asset match the assignment-cover constraint enforces
    (allocate.md §6: constraints come from upstream truth). ``time_windows`` bound the start (as a
    disjunction when there is more than one); ``precedence`` lists task ids that must precede this
    one; ``value`` is its uncertain worth.

    ``duration_s`` is the task's **declared nominal** occupancy of an asset (SI seconds) — the
    interval length the per-asset ``NO_OVERLAP`` constraint reserves so one asset is never assigned
    two tasks at once (RM-P1-ALLOC-02). It is a *declared* nominal, not physics: where a per-pair
    duration is known (a tracked excavator and a wheeled rover cross the same slope differently)
    the cached :class:`~astro_mine.allocate.constraints.CostTable` supersedes it, and Allocate
    re-derives neither (allocate.md §6). ``0.0`` (the default) makes a task a zero-length point,
    which is exactly the pre-scheduling behavior.
    """

    task_id: str
    kind: TaskKind
    location: Volume | None = None
    resource_target_ref: str | None = None
    required_capabilities: list[CapabilityTag] = Field(default_factory=list)
    time_windows: list[TimeWindow] = Field(default_factory=list)
    precedence: list[str] = Field(default_factory=list)
    duration_s: float = Field(default=0.0, ge=0.0)
    value: ValueEstimate


class AssetRef(_Model):
    """A reference to an available asset — its id, SADF capability tags, and SI budgets.

    Resolved from Fleet SADF (allocate.md §3): ``capability_tags`` are the Core
    :class:`~astro_mine.core.sadf.CapabilityTag`\\ s the asset declares (matched against a
    task's ``required_capabilities``); ``budgets`` maps a named SI budget (e.g.
    ``energy_j``, ``time_s``) to its capacity — consumed by the power/energy constraint
    builders in RM-P1-ALLOC-03.
    """

    asset_id: str
    capability_tags: list[CapabilityTag] = Field(default_factory=list)
    budgets: dict[str, float] = Field(default_factory=dict)


class Objective(_Model):
    """The optimization objective: a ``sense`` plus a reference to / inlining of a Core
    :class:`~astro_mine.core.objective.model.ObjectiveSpec` and optional scalarization
    ``weights``.

    Core owns the objective↔metric binding (:mod:`astro_mine.core.objective`); Allocate only
    states the sense it optimizes and the per-criterion weights it scalarizes with —
    optimization lives above Core (core.md §3). ``spec`` inlines the ObjectiveSpec;
    ``objective_ref`` references it by content hash when it is resolved out-of-band.
    """

    sense: ObjectiveSense = ObjectiveSense.MAXIMIZE
    objective_ref: str | None = None
    spec: ObjectiveSpec | None = None
    weights: dict[str, float] = Field(default_factory=dict)


class SolveBudget(_Model):
    """The time/quality budget a solve runs within (allocate.md §2, principle 1/2).

    ``wall_clock_deadline_s`` caps the anytime solve; ``target_gap`` the acceptable
    optimality gap; ``deterministic`` demands a reproducible plan (same model + seed +
    pinned backend ⇒ same plan, principle 8); ``seed`` roots that reproducibility;
    ``workers`` bounds the backend's parallel search threads (a solver hint, honored only
    for a **non-deterministic** solve — deterministic mode fixes the worker count so the plan
    stays byte-reproducible, RM-P1-ALLOC-02/07). ``workers`` is an append-only additive field
    (RM-P1-ALLOC-02); an absent value lets the backend choose.
    """

    wall_clock_deadline_s: float | None = Field(default=None, gt=0.0)
    target_gap: float | None = Field(default=None, ge=0.0)
    deterministic: bool = True
    seed: int | None = None
    workers: int | None = Field(default=None, gt=0)


class AllocationRequest(_Model):
    """The complete allocation problem: tasks, assets, objective, and solve budget.

    The Core-typed input to the allocation sub-interface. Structural validation enforces
    unique task/asset ids, that every ``precedence`` reference resolves to a known task and
    is not self-referential, and that the precedence relation is acyclic — so a compiled IR
    and any solve over it are well-defined (allocate.md §3).
    """

    request_id: str
    tasks: list[Task] = Field(min_length=1)
    assets: list[AssetRef] = Field(min_length=1)
    objective: Objective = Field(default_factory=Objective)
    budget: SolveBudget = Field(default_factory=SolveBudget)

    @model_validator(mode="after")
    def _well_formed(self) -> Self:
        task_ids = [t.task_id for t in self.tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("duplicate task_id in AllocationRequest.tasks")
        asset_ids = [a.asset_id for a in self.assets]
        if len(set(asset_ids)) != len(asset_ids):
            raise ValueError("duplicate asset_id in AllocationRequest.assets")
        known = set(task_ids)
        for t in self.tasks:
            for p in t.precedence:
                if p == t.task_id:
                    raise ValueError(f"task {t.task_id!r} lists itself in precedence")
                if p not in known:
                    raise ValueError(f"task {t.task_id!r} precedence references unknown task {p!r}")
        self._assert_acyclic()
        return self

    def _assert_acyclic(self) -> None:
        # Kahn's algorithm: a precedence graph with a cycle has no valid topological order.
        preds = {t.task_id: set(t.precedence) for t in self.tasks}
        ready = sorted(tid for tid, ps in preds.items() if not ps)
        removed: list[str] = []
        while ready:
            tid = ready.pop(0)
            removed.append(tid)
            newly_ready = []
            for other, ps in preds.items():
                if tid in ps:
                    ps.discard(tid)
                    if not ps and other not in removed and other not in ready:
                        newly_ready.append(other)
            ready = sorted(ready + newly_ready)
        if len(removed) != len(preds):
            raise ValueError("AllocationRequest.tasks precedence relation contains a cycle")


# --- constraint context (Core-typed handles, not a wire document) ----------------


@dataclass(frozen=True, slots=True)
class ConstraintContext:
    """The upstream constraint truth a solve reasons against — Core-typed handles only.

    Mirrors the Core :class:`~astro_mine.core.policy.model.DecisionContext` (a lightweight,
    frozen container, **not** a serializable wire document): it aggregates the Core
    contracts the constraint builders consume — Worlds traversability
    (:class:`~astro_mine.core.world.protocol.WorldProvider`), Prospect value
    (:class:`~astro_mine.core.resource.protocol.ResourceField`), the Link contact graph
    (:class:`~astro_mine.core.messages.model.ContactPlan`), and the Fleet SADF
    :class:`~astro_mine.core.sadf.Asset` handles keyed by asset id — plus an open ``extras``
    map. It is delivered to :meth:`AllocationPlanner.decide` through
    ``DecisionContext.extras`` and consumed via those Core contracts, never through a sibling
    import (allocate.md §6). RM-P1-ALLOC-01 threads it; the physics constraint builders that
    read it land in RM-P1-ALLOC-03.

    ``info_values`` is the **injection surface for Prospect information value** (RM-P1-ALLOC-04):
    a per-task expected-value-of-perfect-information (EVPI) map, already denominated in the same
    ROI value units as the objective, that a producer (Mind / Studio / Bench) populates from
    [Prospect](prospect.md) — so the info-gain objective consumes Prospect's *distributional*
    value without a sibling import. Absent ⇒ no injected info value (the info-gain builder may fall
    back to a resource-variance proxy, or contribute nothing).
    """

    world: WorldProvider | None = None
    resource: ResourceField | None = None
    contacts: ContactPlan | None = None
    assets: Mapping[str, Asset] = field(default_factory=dict)
    extras: Mapping[str, Any] = field(default_factory=dict)
    info_values: Mapping[str, float] | None = None


# --- response side ---------------------------------------------------------------


class ScheduledTask(_Model):
    """One task placed on an asset's timeline: which task, its Core kind, and its window."""

    task_id: str
    kind: TaskKind
    start_s: float
    end_s: float

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.end_s < self.start_s:
            raise ValueError(f"scheduled end_s ({self.end_s}) precedes start_s ({self.start_s})")
        return self


class AssetSchedule(_Model):
    """One asset's time-ordered, **non-overlapping** task sequence.

    An asset is a single-capacity resource: it does one task at a time. The model enforces both
    halves structurally — starts are non-decreasing, *and* no task starts before its predecessor
    on the same asset has ended (RM-P1-ALLOC-02). A schedule that double-books an asset is not
    representable, so an infeasible plan cannot be smuggled through the response type itself.

    This is a *structural* invariant over the reported ``[start_s, end_s]`` intervals only. The
    authoritative temporal feasibility check is
    :func:`~astro_mine.allocate.verify_feasible`, which re-derives each interval's length from the
    **IR's** ``NO_OVERLAP`` sizes rather than trusting the plan's own ``end_s`` — so a backend that
    under-reports a task's end cannot hide an overlap (allocate.md §9).
    """

    asset_id: str
    tasks: list[ScheduledTask] = Field(default_factory=list)

    @model_validator(mode="after")
    def _time_ordered_and_disjoint(self) -> Self:
        for previous, current in itertools.pairwise(self.tasks):
            if current.start_s < previous.start_s:
                raise ValueError(f"AssetSchedule {self.asset_id!r} tasks are not time-ordered")
            if current.start_s < previous.end_s - _EPS:
                raise ValueError(
                    f"AssetSchedule {self.asset_id!r} double-books the asset: "
                    f"{current.task_id!r} starts at {current.start_s} before {previous.task_id!r} "
                    f"ends at {previous.end_s}"
                )
        return self


class BindingConstraint(_Model):
    """A constraint active at the returned plan — part of the plan's explanation
    (allocate.md §9/§10; RM-P1-ALLOC-06).

    ``constraint_id`` references the IR constraint; ``slack`` is its residual (``0`` when tight);
    ``source`` traces the constraint back to the upstream truth that produced it — the Link
    contact window, Worlds terrain, Fleet power budget, or the request's own time-window /
    precedence — so an operator learns *what* bound the result (``LUNAR-UX-004``). ``source`` is an
    append-only additive field (RM-P1-ALLOC-06); absent means the source was not classified.
    """

    constraint_id: str
    kind: ConstraintKind
    slack: float | None = None
    source: str | None = None


class ObjectiveContribution(_Model):
    """One family's (optionally one task's) contribution to the realized objective (RM-P1-ALLOC-06).

    ``family`` is the objective-family tag the term carries in the IR metadata
    (``objective_family::{term_id}`` — e.g. ``roi`` / ``info_gain`` from RM-P1-ALLOC-04, or the
    default ``value`` for the structural skeleton); ``value`` is the summed
    ``coefficient * variable_value`` of every objective term in that family (and, when a
    per-task breakdown is requested, restricted to ``task_id``). ``task_id`` is ``None`` for a
    family-level contribution.
    """

    family: str
    value: float
    task_id: str | None = None


class ObjectiveDecomposition(_Model):
    """The realized objective broken into its per-family contributions (allocate.md §10).

    Every plan ships with *why this objective value* — the info-gain-vs-ROI split (RM-P1-ALLOC-04)
    and any other objective family, each with its contribution. Computed over the solver-neutral IR
    from the same variable-value mapping the feasibility verifier uses, so ``total`` equals the sum
    of the ``contributions`` **and** the plan's ``realized_objective`` by construction — the
    property the RM-P1-ALLOC-06 acceptance test asserts.
    """

    total: float
    contributions: list[ObjectiveContribution] = Field(default_factory=list)


class InfeasibilityCertificate(_Model):
    """An explanation of *why* no feasible plan exists — the irreducible infeasible set.

    A reserved shape (allocate.md §10): the RM-P1-ALLOC-01 result type carries the *slot* so
    a consumer's contract is stable; RM-P1-ALLOC-06 populates the actual IIS. ``constraint_ids``
    / ``task_ids`` name the irreducible conflicting subset and ``explanation`` the human-readable
    reason (e.g. "no contact window long enough to relay the haul before the power floor").
    """

    constraint_ids: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    explanation: str | None = None


class AllocationProvenance(_Model):
    """The reproducibility provenance of a plan (allocate.md §5; conventions.md §5).

    Mirrors Core's build-time :class:`~astro_mine.core.registry.Provenance` and run-time
    :class:`~astro_mine.core.provenance.RunProvenance`: the content hashes of every pinned
    input, the IR version, the solver backend + its pinned version, the seed, the wall-clock
    budget actually consumed, and the Core interface versions the plan was produced against —
    so any plan (and any Bench score) reproduces exactly.
    """

    input_hashes: list[str] = Field(default_factory=list)
    ir_version: str
    backend: str
    backend_version: str | None = None
    seed: int | None = None
    budget_consumed_s: float | None = Field(default=None, ge=0.0)
    core_interface_versions: dict[str, str] = Field(default_factory=dict)


class Allocation(_Model):
    """The result: a feasible plan (or an infeasibility certificate) plus its explanation.

    ``status`` is the feasibility/optimality verdict. The feasibility contract is enforced
    structurally (:meth:`_feasibility_contract`): a feasible status (``OPTIMAL``/``FEASIBLE``)
    MUST carry a ``plan`` and MUST NOT carry a certificate; ``INFEASIBLE`` MUST carry no
    ``plan`` (its certificate slot is populated by RM-P1-ALLOC-06); and a result never
    carries both a plan and a certificate at once. The plan is
    *feasible-by-construction* but Allocate is **not** the safety authority — the result is
    wrappable directly by Guard, which re-checks every hard constraint independently
    (allocate.md §9).
    """

    status: AllocationStatus
    plan: list[AssetSchedule] | None = None
    realized_objective: float | None = None
    optimality_gap: float | None = Field(default=None, ge=0.0)
    binding_constraints: list[BindingConstraint] = Field(default_factory=list)
    objective_decomposition: ObjectiveDecomposition | None = None
    provenance: AllocationProvenance
    infeasibility_certificate: InfeasibilityCertificate | None = None

    @model_validator(mode="after")
    def _feasibility_contract(self) -> Self:
        feasible = self.status in (AllocationStatus.OPTIMAL, AllocationStatus.FEASIBLE)
        if feasible:
            if self.plan is None:
                raise ValueError(f"status={self.status} requires a plan")
            if self.infeasibility_certificate is not None:
                raise ValueError(
                    f"status={self.status} is feasible and must not carry an "
                    "infeasibility_certificate"
                )
        elif self.status is AllocationStatus.INFEASIBLE and self.plan is not None:
            raise ValueError("status=infeasible must not carry a plan")
        if self.plan is not None and self.infeasibility_certificate is not None:
            raise ValueError("an Allocation must not carry both a plan and a certificate")
        return self
