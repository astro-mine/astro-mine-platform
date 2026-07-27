"""Unit tests for the RM-P1-ALLOC-03 constraint builders (power, comms-window, terrain, value).

Each builder is exercised directly against the Core-contract fakes: it maps its upstream input into
IR constraints/keep-outs, and the mapping is asserted precisely (allocate.md §3/§5).
"""

from __future__ import annotations

from astro_mine.allocate import compile_request
from astro_mine.allocate.constraints import (
    ConstraintConfig,
    CostTable,
    build_comms_constraints,
    build_power_constraints,
    build_terrain_constraints,
    compile_with_constraints,
    refine_value_objective,
)
from astro_mine.allocate.constraints.config import CommsPolicy, PowerPolicy, TerrainPolicy
from astro_mine.allocate.constraints.terrain import keepout_constraint_id
from astro_mine.allocate.enums import ConstraintKind, ConstraintSense
from astro_mine.allocate.model.ir.utils import assignment_pairs
from astro_mine.core.units import J2000_EPOCH
from astro_mine.core.world.model import IlluminationState
from tests.allocate import constraint_factories as F
from tests.allocate.factories import anchor_request

# --- terrain -------------------------------------------------------------------------------


def test_terrain_slope_keeps_out_the_pair_over_the_asset_limit() -> None:
    request = anchor_request()
    base_ir = compile_request(request)
    ctx = F.context(
        world=F.FakeWorld(slope_deg=40.0),
        assets={"excavator-1": F.sadf_asset("excavator-1", max_slope_deg=25.0)},
    )
    result = build_terrain_constraints(
        request, base_ir, ctx, config=ConstraintConfig(), costs=CostTable()
    )

    # The excavate task's only eligible asset is over its 25° slope limit → forbidden.
    assert ("excavate-crater-a", "excavator-1") in result.forbidden
    assert any(f.code == "terrain.slope_keepout" for f in result.findings)


def test_terrain_shadow_keepout_only_when_illumination_required() -> None:
    request = anchor_request()
    base_ir = compile_request(request)
    ctx = F.context(world=F.FakeWorld(slope_deg=2.0, illumination=IlluminationState.SHADOW))

    lit_ok = build_terrain_constraints(
        request, base_ir, ctx, config=ConstraintConfig(), costs=CostTable()
    )
    assert lit_ok.forbidden == frozenset()  # require_illuminated defaults False → shadow tolerated

    require_light = ConstraintConfig(terrain=TerrainPolicy(require_illuminated=True))
    kept_out = build_terrain_constraints(
        request, base_ir, ctx, config=require_light, costs=CostTable()
    )
    assert ("prospect-crater-a", "prospector-rover-1") in kept_out.forbidden
    assert any(f.code == "terrain.shadow_keepout" for f in kept_out.findings)


def test_terrain_bearing_capacity_keepout() -> None:
    request = anchor_request()
    base_ir = compile_request(request)
    # Regolith bears 10 kPa; the excavator loads 30 kPa → it would sink → kept out.
    ctx = F.context(
        world=F.FakeWorld(slope_deg=2.0, bearing_capacity_pa=1.0e4),
        assets={"excavator-1": F.sadf_asset("excavator-1", ground_pressure_pa=3.0e4)},
    )
    result = build_terrain_constraints(
        request, base_ir, ctx, config=ConstraintConfig(), costs=CostTable()
    )
    assert ("excavate-crater-a", "excavator-1") in result.forbidden
    assert any(f.code == "terrain.bearing_keepout" for f in result.findings)


def test_terrain_resolves_durations_from_cost_table_and_flags_missing() -> None:
    request = anchor_request()
    base_ir = compile_request(request)
    ctx = F.context(world=F.FakeWorld(slope_deg=2.0))
    costs = F.cost_table({("prospect-crater-a", "prospector-rover-1"): (900.0, None)})
    result = build_terrain_constraints(
        request, base_ir, ctx, config=ConstraintConfig(), costs=costs
    )
    assert result.durations[("prospect-crater-a", "prospector-rover-1")] == 900.0
    # A pair with no cost entry falls back to 0.0 and marks the build degraded.
    assert result.durations[("excavate-crater-a", "excavator-1")] == 0.0
    assert "cost.missing_duration" in result.degraded


# --- comms ---------------------------------------------------------------------------------


def test_comms_gate_forbids_when_no_window_long_enough() -> None:
    request = anchor_request()
    base_ir = compile_request(request)
    # Haul needs a 600 s relay window but the only window is 300 s → no fit → task infeasible.
    contacts = F.contact_plan(
        {"prospector-rover-1": (4000.0, 4300.0), "hauler-1": (4000.0, 4300.0)}
    )
    ctx = F.context(contacts=contacts)
    config = ConstraintConfig(
        comms=CommsPolicy(relay_required_task_ids=frozenset({"haul-to-plant"}), epoch0=J2000_EPOCH)
    )
    durations = {
        ("haul-to-plant", "prospector-rover-1"): 600.0,
        ("haul-to-plant", "hauler-1"): 600.0,
    }

    result = build_comms_constraints(
        request, base_ir, ctx, config=config, durations=durations, forbidden=frozenset()
    )
    assert ("haul-to-plant", "prospector-rover-1") in result.forbidden
    assert ("haul-to-plant", "hauler-1") in result.forbidden
    assert any("no contact window long enough to relay" in f.detail for f in result.findings)


def test_comms_gate_narrows_start_window_when_window_fits() -> None:
    request = anchor_request()
    base_ir = compile_request(request)
    contacts = F.contact_plan(
        {"prospector-rover-1": (5000.0, 9000.0), "hauler-1": (5000.0, 9000.0)}
    )
    ctx = F.context(contacts=contacts)
    config = ConstraintConfig(
        comms=CommsPolicy(relay_required_task_ids=frozenset({"haul-to-plant"}), epoch0=J2000_EPOCH)
    )
    durations = {
        ("haul-to-plant", "prospector-rover-1"): 600.0,
        ("haul-to-plant", "hauler-1"): 600.0,
    }

    result = build_comms_constraints(
        request, base_ir, ctx, config=config, durations=durations, forbidden=frozenset()
    )
    assert result.forbidden == frozenset()
    lo = next(c for c in result.constraints if c.id == "comms_lo::haul-to-plant")
    hi = next(c for c in result.constraints if c.id == "comms_hi::haul-to-plant")
    assert lo.sense is ConstraintSense.GE and lo.rhs == 5000.0
    # start must leave room for the 600 s duration before the window closes at 9000.
    assert hi.sense is ConstraintSense.LE and hi.rhs == 8400.0


def test_comms_relay_gated_with_no_contact_data_degrades_and_forbids() -> None:
    request = anchor_request()
    base_ir = compile_request(request)
    config = ConstraintConfig(
        comms=CommsPolicy(relay_required_task_ids=frozenset({"haul-to-plant"}), epoch0=J2000_EPOCH)
    )
    result = build_comms_constraints(
        request, base_ir, F.context(), config=config, durations={}, forbidden=frozenset()
    )
    assert "comms.no_contact_data" in result.degraded
    assert ("haul-to-plant", "hauler-1") in result.forbidden


# --- power ---------------------------------------------------------------------------------


def test_power_emits_one_energy_budget_constraint_per_asset() -> None:
    request = anchor_request()
    base_ir = compile_request(request)
    pairs = assignment_pairs(base_ir)
    costs = F.cost_table(
        {
            ("prospect-crater-a", "prospector-rover-1"): (None, 2.0e6),
            ("excavate-crater-a", "excavator-1"): (None, 5.0e6),
        }
    )
    result = build_power_constraints(
        request,
        F.context(),
        config=ConstraintConfig(),
        costs=costs,
        pairs=pairs,
        durations={},
        forbidden=frozenset(),
    )
    excavator = next(c for c in result.constraints if c.id == "power::excavator-1")
    assert excavator.kind is ConstraintKind.LINEAR and excavator.sense is ConstraintSense.LE
    assert excavator.rhs == 1.2e7  # excavator-1 energy budget from the anchor AssetRef
    (term,) = excavator.terms
    assert term.coefficient == 5.0e6  # the excavate energy cost from the table


def test_power_reserves_the_housekeeping_floor_over_the_horizon() -> None:
    request = anchor_request()
    base_ir = compile_request(request)
    pairs = assignment_pairs(base_ir)
    ctx = F.context(assets={"excavator-1": F.sadf_asset("excavator-1", floor_w=100.0)})
    config = ConstraintConfig(power=PowerPolicy(horizon_s=1000.0))
    result = build_power_constraints(
        request,
        ctx,
        config=config,
        costs=CostTable(),
        pairs=pairs,
        durations={},
        forbidden=frozenset(),
    )
    excavator = next(c for c in result.constraints if c.id == "power::excavator-1")
    # 1.2e7 capacity minus 100 W * 1000 s reserved = 1.2e7 - 1.0e5 available.
    assert excavator.rhs == 1.2e7 - 1.0e5


# --- value ---------------------------------------------------------------------------------


def test_value_scales_objective_by_posterior_and_flags_deterministic_mode() -> None:
    request = anchor_request()
    base_ir = compile_request(request)
    ctx = F.context(resource=F.FakeField(mean=0.25, variance=0.04))
    result = refine_value_objective(request, base_ir, ctx, config=ConstraintConfig())

    # Only the prospect task carries a resource_target_ref → only its term is rescaled by 0.25.
    prospect_term = next(
        t
        for t in result.objective_terms
        if t.var_ref == "assign::prospect-crater-a::prospector-rover-1"
    )
    base_term = next(
        t
        for t in base_ir.objective_terms
        if t.var_ref == "assign::prospect-crater-a::prospector-rover-1"
    )
    assert prospect_term.coefficient == base_term.coefficient * 0.25
    assert "value.deterministic_mean" in result.degraded
    assert result.metadata["value_variance::prospect-crater-a"] == repr(0.04)


def test_value_is_a_no_op_without_a_resource_field() -> None:
    request = anchor_request()
    base_ir = compile_request(request)
    result = refine_value_objective(request, base_ir, F.context(), config=ConstraintConfig())
    assert result.objective_terms == tuple(base_ir.objective_terms)
    assert result.degraded == ()


# --- compose -------------------------------------------------------------------------------


def test_compose_folds_input_hashes_into_ir_metadata_and_keeps_referential_integrity() -> None:
    request = anchor_request()
    config = ConstraintConfig(
        comms=CommsPolicy(relay_required_task_ids=frozenset({"haul-to-plant"}), epoch0=J2000_EPOCH)
    )
    costs = F.cost_table({("haul-to-plant", "hauler-1"): (600.0, 1.0e6)})
    ctx = F.context(
        world=F.FakeWorld(slope_deg=3.0),
        contacts=F.contact_plan(
            {"prospector-rover-1": (5000.0, 9000.0), "hauler-1": (5000.0, 9000.0)}
        ),
        resource=F.FakeField(),
    )
    comp = compile_with_constraints(request, ctx, config=config, costs=costs)
    assert comp.ir.metadata["constraint_config_hash"] == config.content_hash()
    assert comp.ir.metadata["cost_table_hash"] == costs.content_hash()
    # A keep-out/window/power constraint references only existing variables (validator passed).
    var_ids = {v.id for v in comp.ir.variables}
    for c in comp.ir.constraints:
        assert all(t.var_ref in var_ids for t in c.terms)


def test_keepout_and_comms_never_collide_on_a_constraint_id() -> None:
    request = anchor_request()
    config = ConstraintConfig(
        terrain=TerrainPolicy(default_max_slope_deg=10.0),
        comms=CommsPolicy(relay_required_task_ids=frozenset({"haul-to-plant"}), epoch0=J2000_EPOCH),
    )
    ctx = F.context(
        world=F.FakeWorld(slope_deg=40.0),  # forces slope keep-outs
        contacts=F.contact_plan({"hauler-1": (4000.0, 4200.0)}),  # forces a comms keep-out
    )
    comp = compile_with_constraints(request, ctx, config=config)
    ids = [c.id for c in comp.ir.constraints]
    assert len(ids) == len(set(ids))  # unique — the AllocationIR validator would also reject dupes
    assert any(
        c.id == keepout_constraint_id("prospect-crater-a", "prospector-rover-1")
        for c in comp.ir.constraints
    )


def test_terrain_falls_back_to_the_declared_nominal_duration_and_says_so() -> None:
    # The cost table is the measured, asset-specific truth. When it is silent, the builder falls
    # back to the task's *declared* nominal `duration_s` — a declared fallback, flagged degraded,
    # never an invented physical number (allocate.md §6). With neither, the duration is an honest 0.
    request = anchor_request(
        tasks=[
            t.model_copy(update={"duration_s": 750.0}) if t.task_id == "prospect-crater-a" else t
            for t in anchor_request().tasks
        ]
    )
    comp = compile_with_constraints(request, F.context(), costs=CostTable())

    assert comp.report.durations[("prospect-crater-a", "prospector-rover-1")] == 750.0
    assert "cost.nominal_duration" in comp.report.degraded
    # A task that declares no duration either is still an honest zero-length point.
    assert comp.report.durations[("excavate-crater-a", "excavator-1")] == 0.0
    assert "cost.missing_duration" in comp.report.degraded


def test_a_cost_table_entry_supersedes_the_declared_nominal_duration() -> None:
    # `haul-to-plant` is eligible on two wheeled assets. The cost table measures only one of them —
    # the *measured* duration wins for that pair, and the declared nominal fills the other.
    request = anchor_request(
        tasks=[
            t.model_copy(update={"duration_s": 750.0}) if t.task_id == "haul-to-plant" else t
            for t in anchor_request().tasks
        ]
    )
    comp = compile_with_constraints(
        request,
        F.context(),
        costs=F.cost_table({("haul-to-plant", "hauler-1"): (600.0, None)}),
    )

    assert comp.report.durations[("haul-to-plant", "hauler-1")] == 600.0
    assert comp.report.durations[("haul-to-plant", "prospector-rover-1")] == 750.0
