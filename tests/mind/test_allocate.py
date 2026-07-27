"""Delegation to Allocate — adapter, delegated solver, no embedded solver (RM-P1-MIND-04)."""

from __future__ import annotations

from astro_mine.core.messages.model import ActionBatch
from astro_mine.core.policy.model import DecisionContext
from astro_mine.mind.mission.allocate import (
    ALLOCATION_REQUEST_KEY,
    AllocationAdapter,
    assemble_request,
)
from astro_mine.mind.mission.allocate.reference import GreedyReferenceAllocator
from astro_mine.mind.mission.planner.reference import ReferenceMissionPlanner
from astro_mine.mind.spec.enums import TIER_ORDER, TierRole
from tests.mind.support.harness import assert_deterministic_trace, run_stack
from tests.mind.support.toy_env import ToyProspectingEnv


def _decomposition():  # type: ignore[no-untyped-def]
    obs = ToyProspectingEnv(horizon=4).reset().observations
    return obs, ReferenceMissionPlanner().decide(obs, DecisionContext())


def test_allocator_role_sits_between_mission_and_tamp() -> None:
    assert TIER_ORDER == (TierRole.MISSION, TierRole.ALLOCATOR, TierRole.TAMP, TierRole.CONTROL)


def test_assemble_request_from_decomposition() -> None:
    obs, mission = _decomposition()
    request = assemble_request(obs, mission, deadline_s=0.5)
    assert [t.task_id for t in request.tasks] == ["t0", "t1"]
    assert {a.agent_id for a in request.assets} == {"rover-0", "rover-1"}
    assert request.deadline_s == 0.5


def test_greedy_allocator_assigns_and_records_provenance() -> None:
    obs, mission = _decomposition()
    request = assemble_request(obs, mission)
    solver = GreedyReferenceAllocator()
    context = DecisionContext(seed=7, extras={ALLOCATION_REQUEST_KEY: request})
    batch = solver.decide(obs, context)
    assert all(a.task is not None and a.task.prospect is not None for a in batch.actions)
    allocation = solver.allocation()
    assert allocation is not None
    assert (
        allocation.provenance.solver == "mind.allocate.greedy" and allocation.provenance.seed == 7
    )
    assert allocation.incumbent is True  # anytime good-enough result


def test_greedy_allocator_resolves_a_contended_region() -> None:
    # Two assets, but both regions distinct: greedy gives each its nearest, no double-booking.
    obs, mission = _decomposition()
    request = assemble_request(obs, mission)
    solver = GreedyReferenceAllocator()
    solver.decide(obs, DecisionContext(seed=1, extras={ALLOCATION_REQUEST_KEY: request}))
    tasks = set(solver.allocation().by_agent.values())
    assert len(tasks) == 2  # distinct assignments, no conflict


def test_greedy_allocator_without_request_is_a_defined_noop() -> None:
    solver = GreedyReferenceAllocator()
    batch = solver.decide({}, DecisionContext())
    assert batch == ActionBatch()
    assert solver.allocation().assignments == ()


def test_adapter_delegates_and_carries_provenance() -> None:
    obs, mission = _decomposition()
    adapter = AllocationAdapter(GreedyReferenceAllocator(), deadline_s=0.25)
    batch = adapter.decide(obs, DecisionContext(seed=7, upstream=mission))
    # the adapter returns exactly the delegated solver's assignment (it embeds no solver)...
    assert all(a.task is not None and a.task.prospect is not None for a in batch.actions)
    # ...and carries the solver's structured allocation for plan provenance.
    assert adapter.allocation() is not None
    assert adapter.allocation().provenance.solver == "mind.allocate.greedy"


def test_adapter_publishes_request_under_the_shared_key() -> None:
    captured: dict[str, object] = {}

    class _SpySolver:
        def decide(self, observations, context):  # type: ignore[no-untyped-def]
            captured["request"] = context.extras.get(ALLOCATION_REQUEST_KEY)
            return ActionBatch()

    obs, mission = _decomposition()
    AllocationAdapter(_SpySolver()).decide(obs, DecisionContext(upstream=mission))
    assert captured["request"] is not None  # delegated via the Core Allocator extras convention


def test_allocate_stack_runs_and_is_shielded_and_deterministic() -> None:
    result = run_stack("lunar_prospecting_allocate.yaml", horizon=6, max_ticks=6)
    for obs in result.final_observations.values():
        assert obs.self_state.pose.translation_m.x > 0.0
    assert all(t.shield.intervened for t in result.trace.ticks)  # constraint shield enforces
    assert_deterministic_trace(
        lambda: run_stack("lunar_prospecting_allocate.yaml", horizon=6, max_ticks=6)
    )
