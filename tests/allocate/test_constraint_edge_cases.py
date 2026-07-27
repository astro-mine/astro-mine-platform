"""Edge/fallback branches of the RM-P1-ALLOC-03 builders (degraded modes, missing inputs).

These paths keep the builders honest when upstream truth is partial: energy from SADF storage or a
duration fallback, a missing budget, a housekeeping floor that swamps capacity, a degenerate terrain
normal, and a comms pair already kept out upstream.
"""

from __future__ import annotations

from astro_mine.allocate import (
    AllocationRequest,
    AssetRef,
    ConstraintConfig,
    Task,
    ValueEstimate,
    compile_request,
)
from astro_mine.allocate.constraints.comms import build_comms_constraints
from astro_mine.allocate.constraints.config import CommsPolicy, PowerPolicy
from astro_mine.allocate.constraints.power import build_power_constraints
from astro_mine.allocate.constraints.terrain import slope_deg
from astro_mine.allocate.model.ir.utils import assignment_pairs
from astro_mine.core.messages.enums import TaskKind
from astro_mine.core.units import J2000_EPOCH
from tests.allocate import constraint_factories as F
from tests.allocate.factories import anchor_request


def _one_task_one_asset(*, budgets: dict[str, float] | None = None) -> AllocationRequest:
    return AllocationRequest(
        request_id="edge",
        tasks=[Task(task_id="t0", kind=TaskKind.PROSPECT, value=ValueEstimate(mean=5.0))],
        assets=[AssetRef(asset_id="a0", budgets=budgets or {})],
    )


def test_power_capacity_falls_back_to_sadf_storage_when_no_request_budget() -> None:
    request = _one_task_one_asset()  # no energy_j in AssetRef.budgets
    base_ir = compile_request(request)
    ctx = F.context(assets={"a0": F.sadf_asset("a0", storage_j=4.0e6)})
    result = build_power_constraints(
        request,
        ctx,
        config=ConstraintConfig(),
        costs=F.cost_table({("t0", "a0"): (None, 1.0e6)}),
        pairs=assignment_pairs(base_ir),
        durations={},
        forbidden=frozenset(),
    )
    (constraint,) = result.constraints
    assert constraint.rhs == 4.0e6  # summed SADF storage capacity_j


def test_power_derives_energy_from_duration_when_no_energy_cost() -> None:
    request = _one_task_one_asset(budgets={"energy_j": 1.0e7})
    base_ir = compile_request(request)
    config = ConstraintConfig(power=PowerPolicy(default_mode_power_w=200.0))
    result = build_power_constraints(
        request,
        F.context(),
        config=config,
        costs=F.cost_table({}),
        pairs=assignment_pairs(base_ir),
        durations={("t0", "a0"): 500.0},
        forbidden=frozenset(),
    )
    assert result.energy_costs[("t0", "a0")] == 500.0 * 200.0  # duration * mode power
    assert "power.energy_from_duration" in result.degraded


def test_power_asset_without_a_budget_is_skipped_with_a_finding() -> None:
    request = _one_task_one_asset()  # neither request budget nor SADF storage
    base_ir = compile_request(request)
    result = build_power_constraints(
        request,
        F.context(),
        config=ConstraintConfig(),
        costs=F.cost_table({}),
        pairs=assignment_pairs(base_ir),
        durations={},
        forbidden=frozenset(),
    )
    assert result.constraints == ()
    assert any(f.code == "power.no_budget" for f in result.findings)
    assert "power.no_budget" in result.degraded


def test_power_housekeeping_floor_exceeding_capacity_flags_a_finding() -> None:
    request = _one_task_one_asset(budgets={"energy_j": 1.0e4})
    base_ir = compile_request(request)
    ctx = F.context(assets={"a0": F.sadf_asset("a0", floor_w=100.0)})
    config = ConstraintConfig(power=PowerPolicy(horizon_s=1000.0))  # reserves 1.0e5 J > 1.0e4 J cap
    result = build_power_constraints(
        request,
        ctx,
        config=config,
        costs=F.cost_table({}),
        pairs=assignment_pairs(base_ir),
        durations={},
        forbidden=frozenset(),
    )
    assert result.energy_capacity["a0"] < 0.0
    assert any(f.code == "power.floor_exceeds_capacity" for f in result.findings)


def test_slope_of_a_degenerate_normal_is_flat() -> None:
    assert slope_deg((0.0, 0.0, 0.0), (0.0, 0.0, -1.62)) == 0.0


def test_comms_skips_a_pair_already_kept_out_upstream() -> None:
    request = anchor_request()
    base_ir = compile_request(request)
    contacts = F.contact_plan({"hauler-1": (5000.0, 9000.0)})
    config = ConstraintConfig(
        comms=CommsPolicy(relay_required_task_ids=frozenset({"haul-to-plant"}), epoch0=J2000_EPOCH)
    )
    # prospector-rover-1 is kept out upstream (terrain); only hauler-1 is assessed for a window.
    result = build_comms_constraints(
        request,
        base_ir,
        F.context(contacts=contacts),
        config=config,
        durations={("haul-to-plant", "hauler-1"): 600.0},
        forbidden=frozenset({("haul-to-plant", "prospector-rover-1")}),
    )
    assert ("haul-to-plant", "hauler-1") in result.pair_windows
    assert ("haul-to-plant", "prospector-rover-1") not in result.pair_windows
