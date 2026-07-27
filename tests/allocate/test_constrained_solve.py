"""Constraint-aware solve on the anchor scenario (RM-P1-ALLOC-03 acceptance).

The composed constraints yield a feasible plan **or** an explicit infeasibility certificate, and any
feasible plan re-checks against the augmented IR with the independent verifier — the Guard
recheckable oracle (allocate.md §9/§10; LUNAR-FR-004).
"""

from __future__ import annotations

import pytest

from astro_mine.allocate import (
    AllocationPlanner,
    AllocationStatus,
    ConstraintConfig,
    ConstraintContext,
    compile_with_constraints,
    ir_from_wire,
    ir_to_wire,
    verify_feasible,
)
from astro_mine.allocate.api.planner import (
    CONSTRAINT_CONFIG_KEY,
    CONSTRAINT_CONTEXT_KEY,
    COST_TABLE_KEY,
    REQUEST_KEY,
)
from astro_mine.allocate.constraints.config import CommsPolicy, TerrainPolicy
from astro_mine.core.messages.enums import ActionKind
from astro_mine.core.policy.model import DecisionContext
from astro_mine.core.units import J2000_EPOCH
from tests.allocate import constraint_factories as F
from tests.allocate.factories import anchor_request

_HAUL_GATED = ConstraintConfig(
    comms=CommsPolicy(relay_required_task_ids=frozenset({"haul-to-plant"}), epoch0=J2000_EPOCH)
)
_ANCHOR_COSTS = F.cost_table(
    {
        ("prospect-crater-a", "prospector-rover-1"): (600.0, 1.0e6),
        ("excavate-crater-a", "excavator-1"): (900.0, 3.0e6),
        ("haul-to-plant", "hauler-1"): (600.0, 1.0e6),
        ("haul-to-plant", "prospector-rover-1"): (600.0, 1.0e6),
    }
)


def _feasible_context() -> ConstraintContext:
    return F.context(
        world=F.FakeWorld(slope_deg=4.0),
        contacts=F.contact_plan(
            {"prospector-rover-1": (5000.0, 9000.0), "hauler-1": (5000.0, 9000.0)}
        ),
        resource=F.FakeField(mean=0.4, variance=0.02),
    )


def test_anchor_composed_constraints_yield_a_feasible_verified_plan() -> None:
    request = anchor_request()
    ctx = _feasible_context()
    alloc = AllocationPlanner().solve(request, context=ctx, config=_HAUL_GATED, costs=_ANCHOR_COSTS)

    assert alloc.status is AllocationStatus.FEASIBLE
    assert alloc.plan is not None
    # The plan re-checks against the (independently recompiled) augmented IR (acceptance oracle).
    comp = compile_with_constraints(request, ctx, config=_HAUL_GATED, costs=_ANCHOR_COSTS)
    assert verify_feasible(alloc, comp.ir)
    # Provenance pins the request, config, and cost-table content hashes.
    assert _HAUL_GATED.content_hash() in alloc.provenance.input_hashes
    assert _ANCHOR_COSTS.content_hash() in alloc.provenance.input_hashes


def test_haul_start_respects_the_comms_window() -> None:
    request = anchor_request()
    alloc = AllocationPlanner().solve(
        request, context=_feasible_context(), config=_HAUL_GATED, costs=_ANCHOR_COSTS
    )
    assert alloc.plan is not None
    haul = next(st for sched in alloc.plan for st in sched.tasks if st.task_id == "haul-to-plant")
    assert haul.start_s >= 5000.0  # inside the 5000..9000 contact window, not the 3600 window start


def test_no_contact_window_long_enough_to_relay_is_an_explicit_certificate() -> None:
    request = anchor_request()
    # Every eligible haul asset sees only a 300 s window but the haul needs 600 s → infeasible.
    ctx = F.context(
        contacts=F.contact_plan(
            {"prospector-rover-1": (5000.0, 5300.0), "hauler-1": (5000.0, 5300.0)}
        )
    )
    alloc = AllocationPlanner().solve(request, context=ctx, config=_HAUL_GATED, costs=_ANCHOR_COSTS)

    assert alloc.status is AllocationStatus.INFEASIBLE
    assert alloc.plan is None
    cert = alloc.infeasibility_certificate
    assert cert is not None
    assert cert.explanation is not None
    assert "no contact window long enough to relay" in cert.explanation
    assert "haul-to-plant" in cert.task_ids


def test_slope_keepout_makes_the_request_infeasible_with_a_certificate() -> None:
    request = anchor_request()
    ctx = F.context(world=F.FakeWorld(slope_deg=40.0))  # over the 25° default limit
    alloc = AllocationPlanner().solve(request, context=ctx)
    assert alloc.status is AllocationStatus.INFEASIBLE
    cert = alloc.infeasibility_certificate
    assert cert is not None
    assert cert.explanation is not None
    assert "kept out" in cert.explanation


def test_energy_budget_shortfall_is_infeasible() -> None:
    request = anchor_request()
    # The excavate energy cost (2.0e7 J) exceeds the excavator's 1.2e7 J budget → no feasible asset.
    costs = F.cost_table({("excavate-crater-a", "excavator-1"): (900.0, 2.0e7)})
    alloc = AllocationPlanner().solve(request, context=F.context(), costs=costs)
    assert alloc.status is AllocationStatus.INFEASIBLE
    cert = alloc.infeasibility_certificate
    assert cert is not None
    assert "excavate-crater-a" in cert.task_ids


def test_augmented_ir_round_trips_through_the_versioned_wire_form() -> None:
    request = anchor_request()
    comp = compile_with_constraints(
        request, _feasible_context(), config=_HAUL_GATED, costs=_ANCHOR_COSTS
    )
    assert ir_from_wire(ir_to_wire(comp.ir)) == comp.ir


def test_decide_threads_the_constraint_context_and_emits_scheduled_actions() -> None:
    request = anchor_request()
    ctx = DecisionContext(
        extras={
            REQUEST_KEY: request,
            CONSTRAINT_CONTEXT_KEY: _feasible_context(),
            CONSTRAINT_CONFIG_KEY: _HAUL_GATED,
            COST_TABLE_KEY: _ANCHOR_COSTS,
        }
    )
    batch = AllocationPlanner().decide({}, ctx)
    assert len(batch.actions) == 3
    assert all(a.kind is ActionKind.TASK for a in batch.actions)


def test_decide_emits_no_actions_when_constraints_make_the_request_infeasible() -> None:
    request = anchor_request()
    ctx = DecisionContext(
        extras={
            REQUEST_KEY: request,
            CONSTRAINT_CONTEXT_KEY: F.context(world=F.FakeWorld(slope_deg=40.0)),
            CONSTRAINT_CONFIG_KEY: ConstraintConfig(
                terrain=TerrainPolicy(default_max_slope_deg=20.0)
            ),
        }
    )
    assert AllocationPlanner().decide({}, ctx).actions == []


def test_decide_rejects_a_non_constraint_context_in_extras() -> None:
    ctx = DecisionContext(
        extras={REQUEST_KEY: anchor_request(), CONSTRAINT_CONTEXT_KEY: "not-a-context"}
    )
    with pytest.raises(TypeError, match="must be a ConstraintContext"):
        AllocationPlanner().decide({}, ctx)
