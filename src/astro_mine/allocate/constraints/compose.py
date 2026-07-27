"""``compile_with_constraints`` — compose the RM-P1-ALLOC-03 builders into one IR (RM-P1-ALLOC-03).

The single entry point that lowers an :class:`~astro_mine.allocate.AllocationRequest` **plus its
upstream constraint truth** (a :class:`~astro_mine.allocate.ConstraintContext`) into an augmented,
byte-stable :class:`~astro_mine.allocate.AllocationIR`: the RM-P1-ALLOC-01 structural skeleton, plus
the terrain keep-out, comms-window, and power-budget constraint families, with the objective refined
by Prospect resource value. Every collection is re-emitted in stable sorted order so a fixed
``(request, context, config, cost-table)`` compiles to a byte-identical IR — the determinism the
golden-plan gate (RM-P1-ALLOC-07) depends on. The config and cost-table content hashes are folded
into the IR metadata so the IR's own content hash pins the exact modeling policy and cost inputs.

Ordering matters: terrain resolves per-pair durations first (comms fits tasks into windows using
them; power sums energy using them), then comms gates on contact windows, then power budgets over
the surviving pairs, then the objective is refined — value scaled by resource abundance, info-gain
composed alongside it, and finally the per-pair **cost** family charged against both, priced from
the very durations and energies the earlier builders resolved. The ``assign(task, asset) = 0``
keep-out constraints are emitted here, once, over the union of every builder's forbidden pairs.

The **scheduling** constraints (per-asset ``NO_OVERLAP``: one asset does one task at a time) are
re-emitted here too, replacing the skeleton's — the skeleton reserves each task's *declared nominal*
``duration_s`` for every asset alike, whereas here the terrain builder has resolved the **per-pair**
duration (a tracked excavator and a wheeled rover do not cross the same slope in the same time), and
a kept-out pair contributes no interval at all. Same family, same IR, strictly better data.
"""

from __future__ import annotations

from astro_mine.allocate.api.model import AllocationRequest, ConstraintContext
from astro_mine.allocate.constraints.comms import build_comms_constraints
from astro_mine.allocate.constraints.config import ConstraintConfig
from astro_mine.allocate.constraints.cost_objective import refine_cost_objective
from astro_mine.allocate.constraints.costs import CostTable
from astro_mine.allocate.constraints.infogain import refine_infogain_objective
from astro_mine.allocate.constraints.power import build_power_constraints
from astro_mine.allocate.constraints.result import (
    ConstraintCompilation,
    ConstraintFinding,
    ConstraintReport,
    Pair,
)
from astro_mine.allocate.constraints.terrain import build_terrain_constraints, keepout_constraint_id
from astro_mine.allocate.constraints.value import refine_value_objective
from astro_mine.allocate.enums import ConstraintKind, ConstraintSense
from astro_mine.allocate.model.ir.compile import (
    assignment_var_id,
    compile_request,
    no_overlap_constraints,
)
from astro_mine.allocate.model.ir.model import AllocationIR, Constraint, ConstraintTerm
from astro_mine.allocate.model.ir.schedule import SCHEDULING_KINDS
from astro_mine.allocate.model.ir.utils import assignment_pairs

__all__ = ["compile_with_constraints"]


def _finding_key(f: ConstraintFinding) -> tuple[str, str, str, str]:
    return (f.code, f.task_id or "", f.asset_id or "", f.detail)


def _keepout_constraints(forbidden: frozenset[Pair]) -> list[Constraint]:
    """One ``assign(task, asset) = 0`` LINEAR constraint per forbidden pair (sorted order)."""
    return [
        Constraint(
            id=keepout_constraint_id(task_id, asset_id),
            kind=ConstraintKind.LINEAR,
            terms=[ConstraintTerm(var_ref=assignment_var_id(task_id, asset_id), coefficient=1.0)],
            sense=ConstraintSense.EQ,
            rhs=0.0,
        )
        for task_id, asset_id in sorted(forbidden)
    ]


def compile_with_constraints(
    request: AllocationRequest,
    context: ConstraintContext,
    *,
    config: ConstraintConfig | None = None,
    costs: CostTable | None = None,
) -> ConstraintCompilation:
    """Compile a request + its constraint context into an augmented, byte-stable IR + report."""
    config = config or ConstraintConfig()
    costs = costs or CostTable()

    base_ir = compile_request(request)
    pairs = assignment_pairs(base_ir)

    terrain = build_terrain_constraints(request, base_ir, context, config=config, costs=costs)
    durations = dict(terrain.durations)

    comms = build_comms_constraints(
        request,
        base_ir,
        context,
        config=config,
        durations=durations,
        forbidden=terrain.forbidden,
    )

    forbidden = terrain.forbidden | comms.forbidden

    power = build_power_constraints(
        request,
        context,
        config=config,
        costs=costs,
        pairs=pairs,
        durations=durations,
        forbidden=forbidden,
    )

    # ROI first (Prospect abundance scales the value coefficient), then info-gain composed onto the
    # same objective (active perception traded against extraction) — no new solver paradigm — and
    # finally the per-pair cost charged against both: what this asset spends doing this task. The
    # cost family runs last because it prices the durations and energies the terrain and power
    # builders resolved, and it is the term that makes two feasible assignments score differently
    # at all (issue #22).
    value = refine_value_objective(request, base_ir, context, config=config)
    infogain = refine_infogain_objective(
        request,
        base_ir,
        value.objective_terms,
        context,
        config=config,
        weights=request.objective.weights,
    )
    cost = refine_cost_objective(
        base_ir,
        infogain.objective_terms,
        config=config,
        durations=durations,
        energy_costs=power.energy_costs,
        weights=request.objective.weights,
        forbidden=forbidden,
    )

    constraints = [
        # The skeleton's nominal-duration scheduling constraints are superseded below by the same
        # family emitted over the terrain-resolved per-pair durations.
        *(c for c in base_ir.constraints if c.kind not in SCHEDULING_KINDS),
        *comms.constraints,
        *power.constraints,
        *_keepout_constraints(forbidden),
        *no_overlap_constraints(pairs, durations, forbidden=forbidden),
    ]
    constraints.sort(key=lambda c: c.id)

    metadata = dict(base_ir.metadata)
    metadata.update(value.metadata)
    metadata.update(infogain.metadata)
    metadata.update(cost.metadata)
    metadata["constraint_config_hash"] = config.content_hash()
    metadata["cost_table_hash"] = costs.content_hash()

    ir = AllocationIR(
        ir_version=base_ir.ir_version,
        variables=base_ir.variables,
        constraints=constraints,
        objective_terms=list(cost.objective_terms),
        objective_sense=base_ir.objective_sense,
        metadata=metadata,
    )

    findings = tuple(
        sorted(
            (
                *terrain.findings,
                *comms.findings,
                *power.findings,
                *value.findings,
                *infogain.findings,
                *cost.findings,
            ),
            key=_finding_key,
        )
    )
    degraded = tuple(
        sorted(
            set(terrain.degraded)
            | set(comms.degraded)
            | set(power.degraded)
            | set(value.degraded)
            | set(infogain.degraded)
            | set(cost.degraded)
        )
    )

    report = ConstraintReport(
        findings=findings,
        degraded=degraded,
        forbidden=forbidden,
        pair_windows=comms.pair_windows,
        durations=durations,
        energy_costs=power.energy_costs,
        energy_capacity=power.energy_capacity,
        pair_costs=cost.pair_costs,
    )
    return ConstraintCompilation(ir=ir, report=report, objective_terms=tuple(cost.objective_terms))
