"""The ``[mind]`` binding — Allocate's ``AllocationPlanner`` as a Mind tier plugin (RFC-0006).

RFC-0006's **sibling-binding convention** ("The sibling-binding convention (Guard / Allocate)"):
[Mind](mind.md) discovers tier plugins through the ``astro_mine.mind.tier_plugins`` Python
entry-point group (conventions.md §7), and the *siblings* bind there — so co-installing Mind and
Allocate wires the **real** CP-SAT solver into a Mind stack with **no ``mind → allocate`` dependency
in either base package**. Base Mind keeps shipping its deterministic ``GreedyReferenceAllocator``
stand-in; a deployment that wants the real thing installs ``astro-mine-platform[allocate-mind]``
and names
this plugin in its :class:`~astro_mine.mind.spec.model.StackSpec`. Nothing else changes.

This module is the *only* place in Allocate that imports ``astro_mine.mind``, and it is reachable
only through the entry point — importing :mod:`astro_mine.allocate` never touches it, so the base
package stays Mind-free (allocate.md §6: the narrow waist; the ``[mind]`` extra declares the
dependency).

**What the shim adapts.** Mind's ``AllocationAdapter``
(:mod:`astro_mine.core.plan.allocation`)
publishes a *Mind-owned* delegation DTO under the shared ``allocation.request``
``DecisionContext.extras`` key — deliberately minimal (a region to visit per task, a position per
asset), because Mind must not depend on Allocate's rich request type (mind.md §6; RFC-0006
"Alternatives considered": allocation request/response stay **Allocate-owned**). Allocate's
:class:`~astro_mine.allocate.AllocationPlanner` reads that same key but requires *its own*
:class:`~astro_mine.allocate.AllocationRequest`. :class:`MindAllocationSolver` is the translation
between them, and it belongs **here** — on Allocate's side of the waist, inside the optional extra —
precisely because that is the only side that may know both vocabularies.

It is a Core :class:`~astro_mine.core.policy.protocol.Policy` (so Mind's registry gates and
instantiates it like any tier) that:

1. reads the delegation request from ``allocation.request`` — and passes an *Allocate-native*
   :class:`~astro_mine.allocate.AllocationRequest` straight through when a caller supplies one,
   so a Studio/Bench harness can hand the planner the full constrained problem;
2. lifts the Mind DTO into an Allocate ``AllocationRequest`` (each task a Core ``PROSPECT`` over its
   region, each agent an ``AssetRef``), solves it with the real planner, and
3. maps the plan back to per-agent ``PROSPECT`` directives — the roles/regions Mind's TAMP tier
   consumes — while exposing the solve as a Mind
   :class:`~astro_mine.core.plan.allocation.Allocation` through Mind's optional
   ``AllocationReporter`` seam, so a delegated decision carries the solver + seed provenance that
   reproduces it (RM-P1-MIND-04; RM-P1-ALLOC-07).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from astro_mine.allocate.api.manifest import build_allocation_manifest
from astro_mine.allocate.api.model import (
    AllocationRequest,
    AssetRef,
    Objective,
    SolveBudget,
    Task,
    TimeWindow,
    ValueEstimate,
)
from astro_mine.allocate.api.planner import AllocationPlanner
from astro_mine.allocate.enums import AllocationStatus, ObjectiveSense
from astro_mine.allocate.solvers.registry import CPSAT_BACKEND, known_backends
from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.enums import ActionKind, TaskKind
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    Observation,
    ProspectTask,
    TaskDirective,
)
from astro_mine.core.plan.allocation import (
    ALLOCATION_REQUEST_KEY,
    AllocationAdapter,
    Assignment,
)
from astro_mine.core.plan.allocation import (
    Allocation as MindAllocation,
)
from astro_mine.core.plan.allocation import (
    AllocationProvenance as MindProvenance,
)
from astro_mine.core.plan.allocation import (
    AllocationRequest as MindAllocationRequest,
)
from astro_mine.core.policy.model import DecisionContext
from astro_mine.core.registry import PluginManifest
from astro_mine.core.registry.tier import TierPlugin

__all__ = [
    "PLUGIN_NAME",
    "TIER_ROLE",
    "MindAllocationSolver",
    "allocation_planner_plugin",
    "allocation_planner_tier_manifest",
]

#: The name the plugin registers under — the id a Mind ``StackSpec`` names in its ``allocator``
#: tier binding, and the entry-point name in ``pyproject.toml``.
PLUGIN_NAME = "allocate.planner"

#: The Mind tier this plugin fills. Advertised in the manifest's open ``attributes`` map, which
#: Mind's composer cross-checks against the role the stack binds it to.
TIER_ROLE = "allocator"

#: A delegated prospect task's assumed worth. The Mind DTO carries no value estimate (it is a
#: *coverage* problem: visit every region), so every task is worth the same and the allocator
#: optimizes purely for a feasible cover. A caller that has real values passes an Allocate-native
#: request instead (see :meth:`MindAllocationSolver.decide`).
_DELEGATED_TASK_VALUE = 1.0

#: How long a delegated region occupies its agent. Any positive length works — its only job is to
#: make the per-asset ``NO_OVERLAP`` constraint bite, so that two regions starting in the same
#: ``[0, 0]`` window cannot land on the same agent (see :func:`_as_allocation_request`).
_DELEGATED_TASK_DURATION_S = 1.0


class MindAllocationSolver:
    """A Core ``Policy`` solving Mind's delegated allocation with the real ``AllocationPlanner``.

    The drop-in replacement for Mind's ``GreedyReferenceAllocator``: same ``decide`` contract, same
    ``allocation.request`` extras key, same per-agent ``PROSPECT`` directives out — but the
    assignment comes from CP-SAT over the Allocation IR instead of a nearest-task heuristic.
    """

    def __init__(self, *, backend: str = CPSAT_BACKEND) -> None:
        if backend not in known_backends():
            raise ValueError(
                f"unknown solver backend {backend!r}; known backends: {known_backends()}"
            )
        self._planner = AllocationPlanner(backend=backend)
        self._backend = backend
        self._last: MindAllocation | None = None

    @property
    def backend(self) -> str:
        """The solver backend this tier was instantiated with (recorded in provenance)."""
        return self._backend

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        """Solve the delegated request in ``context.extras``; return per-agent role directives."""
        request = context.extras.get(ALLOCATION_REQUEST_KEY)

        # An Allocate-native request needs no translation — the planner's own `decide` already
        # reads this key, and it carries the full constrained problem (ConstraintContext, config,
        # cost table) a Studio/Bench harness may have staged alongside it.
        if isinstance(request, AllocationRequest):
            self._last = None
            return self._planner.decide(observations, context)

        provenance = MindProvenance(solver=PLUGIN_NAME, seed=context.seed)
        if not isinstance(request, MindAllocationRequest):
            # Nothing to allocate this step (or a payload this tier does not speak): a valid no-op,
            # exactly as Mind's reference allocator treats it.
            self._last = MindAllocation(assignments=(), provenance=provenance)
            return ActionBatch()

        allocation = self._planner.solve(_as_allocation_request(request, seed=context.seed))
        if allocation.plan is None:
            # An honest INFEASIBLE/TIMEOUT: emit no actions rather than an unsatisfiable role.
            # Mind's executive degrades through the tier's declared fallback (mind.md §3,
            # principle 4).
            self._last = MindAllocation(assignments=(), provenance=provenance)
            return ActionBatch()

        regions = {task.task_id: task.region for task in request.tasks}
        assignments: list[Assignment] = []
        actions: list[Action] = []
        for schedule in allocation.plan:
            for scheduled in schedule.tasks:
                assignments.append(
                    Assignment(agent_id=schedule.asset_id, task_id=scheduled.task_id)
                )
                actions.append(
                    Action(
                        agent_id=schedule.asset_id,
                        kind=ActionKind.TASK,
                        sim_time_s=scheduled.start_s,
                        task=TaskDirective(
                            task_kind=TaskKind.PROSPECT,
                            prospect=ProspectTask(region=regions[scheduled.task_id]),
                        ),
                    )
                )

        self._last = MindAllocation(
            assignments=tuple(assignments),
            provenance=provenance,
            # A proven optimum is not an "incumbent"; anything else is the anytime contract's
            # good-enough answer at the deadline (allocate.md §2, principle 2).
            incumbent=allocation.status is not AllocationStatus.OPTIMAL,
            by_agent={a.agent_id: a.task_id for a in assignments},
        )
        return ActionBatch(actions=actions)

    def allocation(self) -> MindAllocation | None:
        """The most recent delegation's structured allocation (Mind's ``AllocationReporter``)."""
        return self._last


def _as_allocation_request(
    request: MindAllocationRequest, *, seed: int | None
) -> AllocationRequest:
    """Lift Mind's minimal delegation DTO into an Allocate :class:`AllocationRequest`.

    The delegated problem is a **per-tick matching**, and the translation says so explicitly:

    - Each region becomes a Core ``PROSPECT`` task and each agent an
      :class:`~astro_mine.allocate.AssetRef` with no required capabilities — the DTO models a
      coverage problem in which any agent can visit any region, so eligibility is universal and the
      solve *is* the assignment.
    - Each task must start **now** (a single ``[0, 0]`` window) and occupies its agent for a unit
      interval, so the per-asset ``NO_OVERLAP`` constraint (RM-P1-ALLOC-02) gives every agent a
      capacity of exactly one task this tick. That is not an embellishment: Mind's
      :class:`~astro_mine.core.plan.allocation.Allocation` reports ``by_agent`` as a 1:1 map
      and the adapter re-decides each tick, so "one role per agent per decision" *is* the contract.
    - Regions beyond the agent count are **deferred to the next replan** rather than making the
      exactly-one cover unsatisfiable. Mind's own reference allocator drops the same surplus (it
      loops over agents, not regions); an infeasible verdict here would strand the whole stack for a
      surplus the next tick would pick up anyway.

    The delegated ``deadline_s`` becomes the solve budget's wall-clock deadline (the anytime
    contract Mind asks for) and ``context.seed`` its seed, so the delegated decision reproduces
    (RM-P1-MIND-04; RM-P1-ALLOC-07).

    Note the translation carries **no per-pair preference**: the Mind DTO's positions/centres would
    let a solver prefer the nearest region, but Allocate's Phase-1 objective is per-*task* (a task
    is worth the same whoever does it), so every complete matching is equally optimal and CP-SAT
    returns a proven-optimal one. A distance-aware assignment needs a per-pair objective family
    (value minus traverse cost), which is the Phase-1-later ROI work — not something to smuggle in
    here.
    """
    assignable = request.tasks[: len(request.assets)]
    return AllocationRequest(
        request_id=request.request_id,
        tasks=[
            Task(
                task_id=task.task_id,
                kind=TaskKind.PROSPECT,
                location=task.region,
                time_windows=[TimeWindow(start_s=0.0, end_s=0.0)],
                duration_s=_DELEGATED_TASK_DURATION_S,
                value=ValueEstimate(mean=_DELEGATED_TASK_VALUE),
            )
            for task in assignable
        ],
        assets=[AssetRef(asset_id=asset.agent_id) for asset in request.assets],
        objective=Objective(sense=ObjectiveSense.MAXIMIZE),
        budget=SolveBudget(
            wall_clock_deadline_s=request.deadline_s,
            deterministic=request.deadline_s is None,
            seed=seed,
        ),
    )


def allocation_planner_tier_manifest(*, backend: str = CPSAT_BACKEND) -> PluginManifest:
    """The Core manifest Mind's registry gates this plugin through.

    Allocate's own :func:`~astro_mine.allocate.build_allocation_manifest` (it consumes Core's
    manifest schema, it does not invent one), plus the ``attributes.tier`` facet Mind's composer
    cross-checks against the ``allocator`` role a stack binds it to. The tier vocabulary is
    *Mind's*, so it is attached here — in the ``[mind]`` shim — and never leaks into Allocate's own
    :class:`~astro_mine.allocate.AllocationAttributes`.
    """
    manifest = build_allocation_manifest(
        name=PLUGIN_NAME,
        version=_allocate_version(),
        artifact_digest=f"sha256:{'0' * 64}",  # unsigned local plugin; Hub attaches the real digest
        backends=[backend],
    )
    return manifest.model_copy(update={"attributes": {**manifest.attributes, "tier": TIER_ROLE}})


def allocation_planner_plugin() -> TierPlugin:
    """Entry-point provider: the real ``AllocationPlanner`` as Mind's ``allocator`` tier.

    Registered under ``astro_mine.mind.tier_plugins`` by the ``[mind]`` extra, so
    :meth:`TierRegistry.from_entry_points
    <astro_mine.mind.registry.registry.TierRegistry.from_entry_points>` discovers it with no Mind
    change at all. The factory wraps the solver in Mind's own
    :class:`~astro_mine.core.plan.allocation.AllocationAdapter` — the same shape Mind's
    reference allocator plugin uses — so the mission tier's decomposition is assembled into the
    delegated request, and the solver's provenance is captured back through the adapter's reporter
    seam. ``params`` may select the solver ``backend`` (default CP-SAT) and the anytime
    ``deadline_s``.
    """
    return TierPlugin(
        manifest=allocation_planner_tier_manifest(),
        factory=lambda params: AllocationAdapter(
            MindAllocationSolver(backend=str(params.get("backend", CPSAT_BACKEND))),
            deadline_s=_deadline(params),
        ),
    )


def _deadline(params: Mapping[str, Any]) -> float | None:
    return float(params["deadline_s"]) if "deadline_s" in params else None


def _allocate_version() -> str:
    from astro_mine.allocate import __version__

    return __version__
