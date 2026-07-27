"""The allocation sub-interface honors the Core Policy/Planner contract (RM-P1-ALLOC-01).

Consumer-driven contract test: ``AllocationPlanner`` satisfies the Core
:class:`~astro_mine.core.policy.protocol.Policy` protocol and its ``decide`` returns a
Sim-consumable :class:`~astro_mine.core.messages.model.ActionBatch` — asserted through Core's
own :func:`~astro_mine.core.policy.conformance.check_policy`. Also exercises the fully
implemented Allocation → ActionBatch adapter.
"""

from __future__ import annotations

import pytest

from astro_mine.allocate import REQUEST_KEY, AllocationPlanner, ConstraintContext
from astro_mine.allocate.api.planner import CONSTRAINT_CONTEXT_KEY
from astro_mine.core.messages.enums import ActionKind, TaskKind
from astro_mine.core.policy.conformance import check_policy
from astro_mine.core.policy.model import DecisionContext
from astro_mine.core.policy.protocol import Allocator, Policy
from tests.allocate.factories import anchor_request, infeasible_request


def test_planner_satisfies_the_core_policy_protocol() -> None:
    planner = AllocationPlanner()
    assert isinstance(planner, Policy)
    # Allocator is a nominal sub-interface marker (shares `decide`); structurally identical.
    assert issubclass(AllocationPlanner, Policy) or isinstance(planner, Policy)
    assert Allocator.__mro__[1] is Policy


def test_check_policy_passes_with_a_request_in_extras() -> None:
    planner = AllocationPlanner()
    ctx = DecisionContext(
        extras={REQUEST_KEY: anchor_request(), CONSTRAINT_CONTEXT_KEY: ConstraintContext()}
    )
    batch = check_policy(planner, {}, ctx)  # must not raise
    assert len(batch.actions) == 3
    assert all(a.kind is ActionKind.TASK for a in batch.actions)


def test_check_policy_passes_with_no_request_to_allocate() -> None:
    # An empty ActionBatch is a valid no-op and must pass the Sim-consumable contract.
    batch = check_policy(AllocationPlanner(), {}, DecisionContext())
    assert batch.actions == []


def test_decide_rejects_a_non_request_in_extras() -> None:
    ctx = DecisionContext(extras={REQUEST_KEY: "not-a-request"})
    with pytest.raises(TypeError, match="must be an AllocationRequest"):
        AllocationPlanner().decide({}, ctx)


def test_decide_on_an_infeasible_request_yields_an_empty_batch() -> None:
    ctx = DecisionContext(extras={REQUEST_KEY: infeasible_request()})
    batch = AllocationPlanner().decide({}, ctx)
    assert batch.actions == []


def test_adapter_maps_scheduled_tasks_to_custom_task_directives() -> None:
    planner = AllocationPlanner()
    batch = planner.decide({}, DecisionContext(extras={REQUEST_KEY: anchor_request()}))
    by_agent = {a.agent_id: a for a in batch.actions}
    assert set(by_agent) == {"prospector-rover-1", "excavator-1", "hauler-1"}
    excavate = by_agent["excavator-1"]
    assert excavate.kind is ActionKind.TASK
    assert excavate.task is not None
    assert excavate.task.task_kind is TaskKind.CUSTOM  # assignment-level directive
    assert excavate.task.directive == "excavate-crater-a"
    assert excavate.task.params["allocated_kind"] == "excavate"
    assert excavate.task.params["asset_id"] == "excavator-1"
    assert excavate.sim_time_s == 1800.0
    assert excavate.task.deadline_s == 1800.0


def test_actions_are_emitted_in_deterministic_asset_then_start_order() -> None:
    planner = AllocationPlanner()
    batch = planner.decide({}, DecisionContext(extras={REQUEST_KEY: anchor_request()}))
    agents = [a.agent_id for a in batch.actions]
    assert agents == sorted(agents)  # asset-sorted (one task each here)
