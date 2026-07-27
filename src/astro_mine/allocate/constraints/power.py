"""``build_power_constraints`` — Fleet SADF budgets → energy-budget constraints (RM-P1-ALLOC-03).

Emits, per asset, the *feasibility* energy budget: the total energy of the tasks assigned to an
asset, plus the housekeeping energy reserved over the horizon, must not exceed the asset's
deliverable energy. Energy costs come from the cached cost table (or a declared, degraded-flagged
fallback of duration * a mode power draw); capacity and the housekeeping floor come from the
request's ``AssetRef.budgets`` and the [Fleet](fleet.md) SADF
:class:`~astro_mine.core.sadf.model.PowerBudget` handle in the
:class:`~astro_mine.allocate.ConstraintContext`.

This is a **feasibility** power floor, *not* the safety power floor — Guard independently re-checks
power floors and keep-out at execution; Allocate is not the safety authority (allocate.md §6). Each
budget is a single linear constraint over the existing assignment variables, so any solver backend
lowers it and :func:`~astro_mine.allocate.verify_feasible` re-checks it structurally.
"""

from __future__ import annotations

from astro_mine.allocate.api.model import AllocationRequest, AssetRef, ConstraintContext
from astro_mine.allocate.constraints.config import ConstraintConfig, PowerPolicy
from astro_mine.allocate.constraints.costs import CostTable
from astro_mine.allocate.constraints.result import ConstraintFinding, Pair, PowerResult
from astro_mine.allocate.enums import ConstraintKind, ConstraintSense
from astro_mine.allocate.model.ir.compile import assignment_var_id
from astro_mine.allocate.model.ir.model import Constraint, ConstraintTerm
from astro_mine.core.sadf import Asset

__all__ = ["build_power_constraints", "power_constraint_id"]


def power_constraint_id(asset_id: str) -> str:
    """The stable id of an asset's energy-budget constraint."""
    return f"power::{asset_id}"


def _energy_capacity(asset_ref: AssetRef, sadf: Asset | None, policy: PowerPolicy) -> float | None:
    """The asset's deliverable energy (J): the request budget key, else the SADF storage sum."""
    budget = asset_ref.budgets.get(policy.energy_budget_key)
    if budget is not None:
        return budget
    if sadf is not None and sadf.power is not None and sadf.power.storage:
        return sum(s.capacity_j for s in sadf.power.storage)
    return None


def _reserved_energy(sadf: Asset | None, policy: PowerPolicy) -> float:
    """Housekeeping energy (J) reserved over the horizon from the SADF power floor."""
    if not policy.reserve_floor or policy.horizon_s <= 0.0 or sadf is None or sadf.power is None:
        return 0.0
    floor_w = sadf.power.floor_w
    return floor_w * policy.horizon_s if floor_w is not None else 0.0


def _pair_energy(
    task_id: str,
    asset_id: str,
    costs: CostTable,
    durations: dict[Pair, float],
    policy: PowerPolicy,
    degraded: set[str],
) -> float:
    """Energy (J) for one asset doing one task: the cost-table entry, else duration * mode power."""
    entry = costs.lookup(task_id, asset_id)
    if entry is not None and entry.energy_j is not None:
        return entry.energy_j
    duration = durations.get((task_id, asset_id))
    if duration is not None and duration > 0.0:
        degraded.add("power.energy_from_duration")
        return duration * policy.default_mode_power_w
    degraded.add("power.no_energy_cost")
    return 0.0


def build_power_constraints(
    request: AllocationRequest,
    context: ConstraintContext,
    *,
    config: ConstraintConfig,
    costs: CostTable,
    pairs: dict[str, list[str]],
    durations: dict[Pair, float],
    forbidden: frozenset[Pair],
) -> PowerResult:
    """Derive one energy-budget constraint per asset from Fleet SADF budgets and the cost table."""
    policy = config.power
    asset_refs = {a.asset_id: a for a in request.assets}

    findings: list[ConstraintFinding] = []
    degraded: set[str] = set()
    energy_costs: dict[Pair, float] = {}
    energy_capacity: dict[str, float] = {}
    constraints: list[Constraint] = []

    # Tasks that can land on each asset (eligible, not kept out) — the terms of its budget.
    tasks_by_asset: dict[str, list[str]] = {}
    for task_id in sorted(pairs):
        for asset_id in pairs[task_id]:
            if (task_id, asset_id) in forbidden:
                continue
            tasks_by_asset.setdefault(asset_id, []).append(task_id)

    for asset_id in sorted(tasks_by_asset):
        asset_ref = asset_refs[asset_id]
        sadf = context.assets.get(asset_id)
        capacity = _energy_capacity(asset_ref, sadf, policy)
        if capacity is None:
            degraded.add("power.no_budget")
            findings.append(
                ConstraintFinding(
                    code="power.no_budget",
                    detail=f"{asset_id} declares no {policy.energy_budget_key} budget/SADF storage",
                    asset_id=asset_id,
                )
            )
            continue

        available = capacity - _reserved_energy(sadf, policy)
        energy_capacity[asset_id] = available

        terms: list[ConstraintTerm] = []
        for task_id in sorted(tasks_by_asset[asset_id]):
            energy = _pair_energy(task_id, asset_id, costs, durations, policy, degraded)
            energy_costs[(task_id, asset_id)] = energy
            terms.append(
                ConstraintTerm(var_ref=assignment_var_id(task_id, asset_id), coefficient=energy)
            )

        constraints.append(
            Constraint(
                id=power_constraint_id(asset_id),
                kind=ConstraintKind.LINEAR,
                terms=terms,
                sense=ConstraintSense.LE,
                rhs=available,
            )
        )
        if available < 0.0:
            findings.append(
                ConstraintFinding(
                    code="power.floor_exceeds_capacity",
                    detail=(
                        f"{asset_id} housekeeping reservation exceeds its energy capacity "
                        f"(available {available:.0f} J)"
                    ),
                    asset_id=asset_id,
                    constraint_id=power_constraint_id(asset_id),
                )
            )

    return PowerResult(
        constraints=tuple(constraints),
        findings=tuple(findings),
        degraded=tuple(sorted(degraded)),
        energy_costs=energy_costs,
        energy_capacity=energy_capacity,
    )
