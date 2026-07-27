"""``refine_cost_objective`` — the per-**pair** cost family: value minus what it cost (ALLOC-04).

The objective family that makes a plan's *assignment* matter. Task value is a per-**task** quantity
— a crater's ice is worth what it is worth whoever digs it — and the assignment cover is
exactly-one, so an objective built from value alone gives **every feasible plan the identical
score**: the solver has nothing to optimize, and the optimality gap it reports is zero by
construction rather than by quality. That is a gauge reading zero because it is unplugged, and it
cannot detect a worse solver (astro-mine-allocate#22, found by #21).

This builder adds the term that varies with *who does what*: the SI cost of **this** asset doing
**this** task — the energy it draws and the time it spends — priced into value units by the declared
:class:`~astro_mine.allocate.constraints.CostPolicy` and **subtracted** from the value it earns. Two
assignments of the same task to different assets now score differently, so a cheaper plan is a
better plan, a worse plan is measurably worse, and the optimality gap becomes a real quality signal
with teeth.

Allocate re-derives no physics here (allocate.md §6): the per-pair energy and duration are exactly
the ones the power and terrain builders already resolved from the cached
:class:`~astro_mine.allocate.constraints.CostTable`, not a traversal model invented in this module.

The family composes onto the *same* linear objective as
:func:`~astro_mine.allocate.constraints.refine_value_objective` and
:func:`~astro_mine.allocate.constraints.refine_infogain_objective` — no new solver paradigm — so
CP-SAT optimizes ``w_roi * roi + w_info * evpi - w_cost * cost`` directly, and every backend lowers
it unchanged. It is **opt-in**: with no ``CostPolicy`` declared it emits no terms at all, so a
request that never asks for it compiles to exactly the IR it did before. Each term is tagged
``objective_family::{term_id} = "cost"`` in the IR metadata, so the RM-P1-ALLOC-06 objective
decomposition reports the value-vs-info-vs-cost split for free.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from astro_mine.allocate.constraints.config import ConstraintConfig
from astro_mine.allocate.constraints.result import ConstraintFinding, Pair
from astro_mine.allocate.enums import ObjectiveSense, VariableSemantic
from astro_mine.allocate.model.ir.model import AllocationIR, ObjectiveTerm

__all__ = ["CostObjectiveResult", "cost_term_id", "refine_cost_objective"]


def cost_term_id(assignment_var_id: str) -> str:
    """The stable id of the cost objective term charged against one assignment variable."""
    return f"cost::{assignment_var_id}"


@dataclass(frozen=True, slots=True)
class CostObjectiveResult:
    """The cost builder's contribution: the combined objective + the priced-in cost per pair.

    ``objective_terms`` is the full objective — the terms it was handed, plus one negative cost term
    per assignment pair whose cost the policy prices. ``pair_costs`` records what each pair was
    charged (in value units), and ``metadata`` carries each term's ``objective_family`` tag and the
    per-pair charge, so a plan's objective decomposition can be audited back to its inputs.
    """

    objective_terms: tuple[ObjectiveTerm, ...]
    findings: tuple[ConstraintFinding, ...] = ()
    degraded: tuple[str, ...] = ()
    pair_costs: Mapping[Pair, float] = field(default_factory=dict)
    metadata: dict[str, str] = field(default_factory=dict)


def refine_cost_objective(
    base_ir: AllocationIR,
    objective_terms: tuple[ObjectiveTerm, ...],
    *,
    config: ConstraintConfig,
    durations: Mapping[Pair, float],
    energy_costs: Mapping[Pair, float],
    weights: Mapping[str, float],
    forbidden: frozenset[Pair] = frozenset(),
) -> CostObjectiveResult:
    """Charge each assignment pair its priced cost against the objective it earns."""
    policy = config.cost
    if policy is None:
        return CostObjectiveResult(objective_terms=objective_terms)

    w_cost = float(weights.get("cost", 1.0))
    # A cost is a cost whichever way the objective is optimized: it is subtracted from a MAXIMIZE
    # objective and added to a MINIMIZE one, so a cheaper assignment is always the better one.
    sign = -1.0 if base_ir.objective_sense is ObjectiveSense.MAXIMIZE else 1.0

    combined = list(objective_terms)
    # Only the cost terms are tagged here: the value/info-gain families tag their own (compose
    # merges every builder's metadata), so this builder never second-guesses another's family.
    metadata: dict[str, str] = {}
    pair_costs: dict[Pair, float] = {}
    degraded: set[str] = set()
    findings: list[ConstraintFinding] = []

    for var in base_ir.variables:
        if var.semantic is not VariableSemantic.ASSIGNMENT or not var.task_ref or not var.asset_ref:
            continue
        pair = (var.task_ref, var.asset_ref)
        # A kept-out pair's variable is pinned to 0, so its cost can never be realized; pricing it
        # would only add a term that is always zero.
        if pair in forbidden:
            continue

        energy = energy_costs.get(pair)
        duration = durations.get(pair)
        if energy is None and policy.energy_price_per_j > 0.0:
            # The power builder prices energy only for assets that declare an energy budget; an
            # asset without one is charged for its *time* alone rather than fabricating a joule.
            degraded.add("cost.no_energy_cost")
        cost = policy.energy_price_per_j * (energy or 0.0) + policy.time_price_per_s * (
            duration or 0.0
        )
        pair_costs[pair] = cost

        term_id = cost_term_id(var.id)
        combined.append(ObjectiveTerm(id=term_id, var_ref=var.id, coefficient=sign * w_cost * cost))
        metadata[f"objective_family::{term_id}"] = "cost"
        metadata[f"pair_cost::{var.task_ref}::{var.asset_ref}"] = repr(cost)

    if degraded:
        findings.append(
            ConstraintFinding(
                code="cost.no_energy_cost",
                detail=(
                    "some pairs carry no energy cost (their asset declares no energy budget, so "
                    "the power builder priced none) and are charged for their duration alone"
                ),
            )
        )

    combined.sort(key=lambda o: o.id)
    return CostObjectiveResult(
        objective_terms=tuple(combined),
        findings=tuple(findings),
        degraded=tuple(sorted(degraded)),
        pair_costs=pair_costs,
        metadata=metadata,
    )
