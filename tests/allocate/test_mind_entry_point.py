"""Consumer contract: a Mind stack resolves the **real** ``AllocationPlanner`` (RFC-0006).

RFC-0006's sibling-binding convention says an ``astro-mine-allocate`` ``[mind]`` extra "ships a
provider that wraps ``AllocationPlanner`` as the ``allocator``-role tier plugin, behind Mind's
``AllocationAdapter``", published under the shared ``allocation.request``
``DecisionContext.extras`` key. This file proves that end to end against the **real installed Mind**
— not a mock of it:

1. the entry point is advertised under ``astro_mine.mind.tier_plugins`` and its provider returns a
   Core-manifest-gated ``TierPlugin``;
2. Mind's own :class:`~astro_mine.mind.registry.registry.TierRegistry` discovers it by entry point,
   gates the manifest through Core, and instantiates it;
3. a minimal Mind ``StackSpec`` naming it in the ``allocator`` tier **composes**, and the composed
   tier is Allocate's CP-SAT planner — *not* Mind's ``GreedyReferenceAllocator`` stand-in
   (RM-P1-MIND-04);
4. driving that stack's allocator produces per-agent role directives and the Mind-side
   :class:`~astro_mine.mind.mission.allocate.model.Allocation` provenance a delegated decision
   reproduces from.

The whole test skips when Mind is not installed — the ``[mind]`` extra is optional, and Allocate's
base package must import and work without it (allocate.md §6/§7).
"""

from __future__ import annotations

import importlib.util
from importlib.metadata import entry_points

import pytest

from astro_mine.core.messages.enums import ActionKind, TaskKind
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    Observation,
    ProspectTask,
    Quat,
    StateSample,
    TaskDirective,
    Transform,
    Vec3,
    Volume,
)
from astro_mine.core.policy.model import DecisionContext
from astro_mine.core.registry.enums import PluginKind

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("astro_mine.mind") is None,
    reason="the [mind] extra (astro-mine-mind) is not installed",
)

from astro_mine.allocate.mind import (  # noqa: E402
    PLUGIN_NAME,
    MindAllocationSolver,
    allocation_planner_plugin,
)
from astro_mine.mind.compose.composer import compose  # noqa: E402 - after the skip guard
from astro_mine.mind.mission.allocate.adapter import (  # noqa: E402
    ALLOCATION_REQUEST_KEY,
    AllocationAdapter,
)
from astro_mine.mind.mission.allocate.reference import GreedyReferenceAllocator  # noqa: E402
from astro_mine.mind.registry.registry import (  # noqa: E402
    ENTRY_POINT_GROUP,
    TierPlugin,
    TierRegistry,
)
from astro_mine.mind.spec.enums import TierRole  # noqa: E402
from astro_mine.mind.spec.model import (  # noqa: E402
    ShieldBinding,
    StackSpec,
    StackSpecDocument,
    TierBinding,
)
from tests.allocate.constraint_factories import MOON_FRAME  # noqa: E402

_SHIELD_PLUGIN = "mind.reference.shield"


def _region(x: float) -> Volume:
    return Volume(
        frame="MOON_ME",
        center_m=Vec3(x=x, y=0.0, z=0.0),
        dimensions_m=Vec3(x=20.0, y=20.0, z=2.0),
    )


def _observations(*agent_ids: str) -> dict[str, Observation]:
    return {
        agent_id: Observation(
            tick=0,
            sim_time_s=0.0,
            agent_id=agent_id,
            self_state=StateSample(
                agent_id=agent_id,
                frame=MOON_FRAME,
                pose=Transform(
                    translation_m=Vec3(x=float(index) * 10.0, y=0.0, z=0.0),
                    rotation_quat_xyzw=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
            ),
        )
        for index, agent_id in enumerate(agent_ids)
    }


def _mission_decomposition(*regions: Volume) -> ActionBatch:
    """The mission tier's output: one PROSPECT region per action — what the adapter reads."""
    return ActionBatch(
        actions=[
            Action(
                agent_id="rover-0",
                kind=ActionKind.TASK,
                task=TaskDirective(
                    task_kind=TaskKind.PROSPECT, prospect=ProspectTask(region=region)
                ),
            )
            for region in regions
        ]
    )


def _minimal_stack(plugin: str = PLUGIN_NAME) -> StackSpecDocument:
    """The smallest legal Mind stack that delegates assignment: one allocator tier + the shield."""
    return StackSpecDocument(
        stack_spec_version="0.1",
        stack_spec=StackSpec(
            id="allocate-delegation",
            name="Delegated allocation over the real CP-SAT planner",
            tiers=[TierBinding(role=TierRole.ALLOCATOR, plugin=plugin)],
            shield=ShieldBinding(plugin=_SHIELD_PLUGIN),
        ),
    )


# --- the entry point exists and is a gated Core plugin -------------------------------


def test_the_plugin_is_advertised_under_minds_tier_plugin_entry_point_group() -> None:
    advertised = {ep.name: ep for ep in entry_points(group=ENTRY_POINT_GROUP)}
    assert PLUGIN_NAME in advertised, (
        f"astro-mine-allocate must advertise {PLUGIN_NAME!r} under {ENTRY_POINT_GROUP!r} "
        "(RFC-0006 sibling-binding convention)"
    )
    plugin = advertised[PLUGIN_NAME].load()()
    assert isinstance(plugin, TierPlugin)


def test_the_manifest_declares_a_policy_plugin_bound_to_the_allocator_tier() -> None:
    manifest = allocation_planner_plugin().manifest
    assert manifest.name == PLUGIN_NAME
    assert manifest.kind is PluginKind.POLICY  # the allocation sub-interface of Core's Policy API
    # Mind's composer cross-checks this facet against the role the stack binds the plugin to.
    assert manifest.attributes["tier"] == "allocator"
    assert manifest.attributes["backends"] == ["cp-sat"]
    assert manifest.core_interfaces["policy"] == "0.1.0"


def test_minds_registry_discovers_and_instantiates_the_plugin_by_entry_point() -> None:
    registry = TierRegistry.from_entry_points()
    assert PLUGIN_NAME in registry

    policy = registry.instantiate(PLUGIN_NAME)
    # Mind wraps it in *its own* adapter — the plugin binds behind AllocationAdapter, exactly as
    # RFC-0006 specifies, so the mission tier's decomposition is assembled into the request.
    assert isinstance(policy, AllocationAdapter)


# --- a minimal Mind stack composes over the real planner ------------------------------


def test_a_minimal_stack_spec_composes_the_real_planner_not_the_reference_stand_in() -> None:
    graph = compose(_minimal_stack(), TierRegistry.from_entry_points())

    (tier,) = [node for node in graph.tiers if node.role is TierRole.ALLOCATOR]
    assert tier.plugin_name == PLUGIN_NAME

    adapter = tier.policy
    assert isinstance(adapter, AllocationAdapter)
    solver = adapter._solver  # the delegated-to policy behind Mind's adapter
    assert isinstance(solver, MindAllocationSolver), (
        "the composed allocator tier must be Allocate's real planner"
    )
    assert not isinstance(solver, GreedyReferenceAllocator)
    assert solver.backend == "cp-sat"


def test_the_stack_binding_selects_the_solver_backend_through_plugin_params() -> None:
    document = StackSpecDocument(
        stack_spec_version="0.1",
        stack_spec=StackSpec(
            id="allocate-delegation-stub",
            name="Delegated allocation over the no-dependency backend",
            tiers=[
                TierBinding(
                    role=TierRole.ALLOCATOR,
                    plugin=PLUGIN_NAME,
                    params={"backend": "trivial-stub", "deadline_s": 0.5},
                )
            ],
            shield=ShieldBinding(plugin=_SHIELD_PLUGIN),
        ),
    )
    graph = compose(document, TierRegistry.from_entry_points())
    (tier,) = [node for node in graph.tiers if node.role is TierRole.ALLOCATOR]
    adapter = tier.policy
    assert isinstance(adapter, AllocationAdapter)
    solver = adapter._solver
    assert isinstance(solver, MindAllocationSolver)
    assert solver.backend == "trivial-stub"


# --- and it actually allocates ---------------------------------------------------------


def test_the_composed_tier_allocates_the_delegated_request_and_reports_provenance() -> None:
    graph = compose(_minimal_stack(), TierRegistry.from_entry_points())
    (tier,) = [node for node in graph.tiers if node.role is TierRole.ALLOCATOR]
    adapter = tier.policy
    assert isinstance(adapter, AllocationAdapter)

    observations = _observations("rover-0", "rover-1")
    context = DecisionContext(
        seed=7, upstream=_mission_decomposition(_region(100.0), _region(-100.0))
    )
    batch = adapter.decide(observations, context)

    # Every agent gets a PROSPECT role over one of the decomposed regions — the shape Mind's TAMP
    # tier consumes, identical to what the reference stand-in produces.
    assert len(batch.actions) == 2
    assert all(a.task is not None and a.task.prospect is not None for a in batch.actions)
    assert {a.agent_id for a in batch.actions} == {"rover-0", "rover-1"}

    # And the solve is reported back through Mind's AllocationReporter seam, so the delegated
    # decision carries the solver + seed that reproduce it (RM-P1-MIND-04; RM-P1-ALLOC-07).
    allocation = adapter.allocation()
    assert allocation is not None
    assert allocation.provenance.solver == PLUGIN_NAME
    assert allocation.provenance.seed == 7
    assert len(allocation.assignments) == 2
    assert sorted(allocation.by_agent) == ["rover-0", "rover-1"]
    assert len(set(allocation.by_agent.values())) == 2  # distinct regions, no double-booking
    assert allocation.incumbent is False  # CP-SAT proved the assignment optimal


def test_the_solver_passes_an_allocate_native_request_straight_through() -> None:
    # A Studio/Bench harness that stages Allocate's own (fully constrained) request under the same
    # extras key gets the planner's native behavior — no lossy round-trip through Mind's DTO.
    from tests.allocate.factories import anchor_request

    solver = MindAllocationSolver(backend="trivial-stub")
    batch = solver.decide({}, DecisionContext(extras={ALLOCATION_REQUEST_KEY: anchor_request()}))

    assert {a.agent_id for a in batch.actions} == {
        "prospector-rover-1",
        "excavator-1",
        "hauler-1",
    }
    assert solver.allocation() is None  # a native solve has no Mind-side delegation DTO to report


def test_the_solver_is_a_defined_noop_without_a_request() -> None:
    solver = MindAllocationSolver(backend="trivial-stub")
    assert solver.decide({}, DecisionContext()) == ActionBatch()
    allocation = solver.allocation()
    assert allocation is not None and allocation.assignments == ()


def test_an_unknown_backend_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="unknown solver backend"):
        MindAllocationSolver(backend="gurobi")


def test_an_infeasible_delegation_degrades_to_an_empty_batch_not_an_exception() -> None:
    # A region no agent can reach (an impossible capability) makes the delegated solve INFEASIBLE.
    # The tier must emit no actions rather than a role nobody can execute — Mind's executive then
    # degrades through the tier's declared fallback (mind.md §3, principle 4).
    from unittest.mock import patch

    from astro_mine.allocate import (
        Allocation,
        AllocationProvenance,
        AllocationStatus,
    )
    from astro_mine.mind.mission.allocate.adapter import assemble_request

    solver = MindAllocationSolver(backend="trivial-stub")
    request = assemble_request(_observations("rover-0"), _mission_decomposition(_region(10.0)))
    infeasible = Allocation(
        status=AllocationStatus.INFEASIBLE,
        plan=None,
        provenance=AllocationProvenance(ir_version="0.1.0", backend="trivial-stub"),
    )
    with patch.object(solver._planner, "solve", return_value=infeasible):
        batch = solver.decide(
            _observations("rover-0"),
            DecisionContext(seed=3, extras={ALLOCATION_REQUEST_KEY: request}),
        )

    assert batch == ActionBatch()
    allocation = solver.allocation()
    assert allocation is not None
    assert allocation.assignments == ()
    assert allocation.provenance.solver == PLUGIN_NAME and allocation.provenance.seed == 3
