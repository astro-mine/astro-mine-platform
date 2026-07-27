"""Binding-constraint reporting traces the result to its upstream truth (RM-P1-ALLOC-06).

``LUNAR-UX-004``: an Allocate decision surfaced to an operator MUST carry binding-constraint
explanations — which power floor / comms window / slope limit / time window bound the result. Each
binding is computed over the solver-neutral IR and carries a ``source`` tracing it back to Link /
Worlds / Fleet / the request; the structural exactly-one cover is excluded as uninformative.
"""

from __future__ import annotations

import importlib.util

import pytest

from astro_mine.allocate import (
    AllocationIR,
    AllocationPlanner,
    AllocationStatus,
    AssetSchedule,
    ConstraintSense,
    compile_request,
)
from astro_mine.allocate.enums import ConstraintKind
from astro_mine.allocate.explain import binding_constraints, binding_source, plan_variable_values
from tests.allocate import constraint_factories as F
from tests.allocate.factories import anchor_request

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ortools") is None, reason="OR-Tools (cp-sat backend) not installed"
)

_EPS = 1.0e-9


def _reevaluate_slack(plan: list[AssetSchedule], ir: AllocationIR, constraint_id: str) -> float:
    values = plan_variable_values(plan, ir)
    c = next(c for c in ir.constraints if c.id == constraint_id)
    lhs = sum(t.coefficient * values[t.var_ref] for t in c.terms)
    if c.sense is ConstraintSense.LE:
        return c.rhs - lhs
    if c.sense is ConstraintSense.GE:
        return lhs - c.rhs
    return abs(lhs - c.rhs)


def test_source_map_covers_every_constraint_family() -> None:
    assert binding_source("keepout::t::a") == "worlds:traversability"
    assert binding_source("power::a") == "fleet:power-budget"
    assert binding_source("comms_lo::t") == "link:contact-window"
    assert binding_source("comms_hi::t") == "link:contact-window"
    assert binding_source("twin_lo::t") == "request:time-window"
    assert binding_source("prec::a->b") == "request:precedence"
    assert binding_source("mystery::x") is None


def test_skeleton_binding_reports_tight_time_windows_and_excludes_cover() -> None:
    request = anchor_request()
    ir = compile_request(request)
    alloc = AllocationPlanner(backend="cp-sat").solve(request)

    assert alloc.plan is not None
    ids = {b.constraint_id for b in alloc.binding_constraints}
    # Every task starts at its window's earliest bound, so twin_lo is tight for each.
    assert ids == {f"twin_lo::{t.task_id}" for t in request.tasks}
    # The structural exactly-one cover is never reported (always tight, uninformative).
    assert not any(b.constraint_id.startswith("cover::") for b in alloc.binding_constraints)
    for b in alloc.binding_constraints:
        assert b.kind is ConstraintKind.TIME_WINDOW
        assert b.source == "request:time-window"
        # A reported binding really is tight against the returned plan.
        assert abs(_reevaluate_slack(alloc.plan, ir, b.constraint_id)) <= _EPS


def test_tight_power_budget_is_a_binding_constraint_from_fleet() -> None:
    # The excavate energy cost equals the excavator's energy budget → power::excavator-1 is tight.
    from astro_mine.allocate.constraints.compose import compile_with_constraints

    costs = F.cost_table({("excavate-crater-a", "excavator-1"): (900.0, 1.2e7)})
    comp_ir = compile_with_constraints(anchor_request(), F.context(), costs=costs).ir
    alloc = AllocationPlanner(backend="cp-sat").solve(
        anchor_request(), context=F.context(), costs=costs
    )
    assert alloc.status is AllocationStatus.OPTIMAL
    assert alloc.plan is not None

    power = next(b for b in alloc.binding_constraints if b.constraint_id == "power::excavator-1")
    assert power.kind is ConstraintKind.LINEAR
    assert power.source == "fleet:power-budget"
    assert abs(_reevaluate_slack(alloc.plan, comp_ir, power.constraint_id)) <= _EPS


def test_terrain_keepout_is_a_binding_constraint_from_worlds() -> None:
    # A slope-limited asset is kept out of a task's location; the task goes elsewhere, but the
    # keep-out (an equality assign=0, tight) is reported so the operator sees the slope limit that
    # shaped the plan (LUNAR-UX-004 — "which slope limit bound the result").
    from astro_mine.allocate import (
        AllocationRequest,
        AssetRef,
        Task,
        ValueEstimate,
    )
    from astro_mine.core.messages.enums import TaskKind
    from astro_mine.core.messages.model import Vec3, Volume
    from astro_mine.core.sadf import CapabilityTag

    request = AllocationRequest(
        request_id="keepout-binding",
        tasks=[
            Task(
                task_id="survey",
                kind=TaskKind.PROSPECT,
                location=Volume(
                    frame="MOON_ME",
                    center_m=Vec3(x=0.0, y=0.0, z=0.0),
                    dimensions_m=Vec3(x=1.0, y=1.0, z=1.0),
                ),
                required_capabilities=[CapabilityTag.MOBILITY_WHEELED],
                value=ValueEstimate(mean=5.0),
            )
        ],
        assets=[
            AssetRef(asset_id="steep-limited", capability_tags=[CapabilityTag.MOBILITY_WHEELED]),
            AssetRef(asset_id="all-terrain", capability_tags=[CapabilityTag.MOBILITY_WHEELED]),
        ],
    )
    ctx = F.context(
        world=F.FakeWorld(slope_deg=40.0),
        assets={
            "steep-limited": F.sadf_asset("steep-limited", max_slope_deg=10.0),
            "all-terrain": F.sadf_asset("all-terrain", max_slope_deg=60.0),
        },
    )
    alloc = AllocationPlanner(backend="cp-sat").solve(request, context=ctx)
    assert alloc.status is AllocationStatus.OPTIMAL

    keepout = next(
        b for b in alloc.binding_constraints if b.constraint_id == "keepout::survey::steep-limited"
    )
    assert keepout.kind is ConstraintKind.LINEAR
    assert keepout.source == "worlds:traversability"


def test_binding_constraints_helper_is_a_pure_function_of_plan_and_ir() -> None:
    request = anchor_request()
    ir = compile_request(request)
    alloc = AllocationPlanner(backend="cp-sat").solve(request)
    # The helper called directly agrees with what the planner attached to the plan.
    direct = binding_constraints(alloc.plan or [], ir)
    assert [b.model_dump() for b in direct] == [b.model_dump() for b in alloc.binding_constraints]


def test_plan_variable_values_maps_assignment_and_start() -> None:
    request = anchor_request()
    ir = compile_request(request)
    alloc = AllocationPlanner(backend="cp-sat").solve(request)
    values = plan_variable_values(alloc.plan or [], ir)
    # Prospect is placed on the only eligible asset at t=0.
    assert values["assign::prospect-crater-a::prospector-rover-1"] == 1.0
    assert values["start::prospect-crater-a"] == 0.0
    # Every declared variable resolves to a value.
    assert set(values) == {v.id for v in ir.variables}
