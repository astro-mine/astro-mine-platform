"""Objective decomposition over the solver-neutral IR (RM-P1-ALLOC-06).

*Why this objective value* — the realized objective broken into its per-family contributions,
including the info-gain-vs-ROI split from RM-P1-ALLOC-04 (allocate.md §10). Each IR objective
term is tagged with its family in the IR ``metadata`` (``objective_family::{term_id}`` — ``roi``
/ ``info_gain`` from the info-gain builder, ``cost`` from the per-pair cost builder, or the default
``value`` for the structural skeleton); this groups the terms by family and sums each family's
``coefficient * variable_value`` under the plan. The ``cost`` family's contribution is **negative**
under a MAXIMIZE objective — it is what the plan paid, and the reason two feasible plans score
differently at all.

The value mapping is :func:`~astro_mine.allocate.explain.binding.plan_variable_values` — the same
mapping the feasibility verifier evaluates through — so ``ObjectiveDecomposition.total`` equals
both the sum of the contributions and the plan's ``realized_objective`` by construction (the
property RM-P1-ALLOC-06 asserts). No backend-specific state leaks into the explanation.
"""

from __future__ import annotations

from astro_mine.allocate.api.model import (
    AssetSchedule,
    ObjectiveContribution,
    ObjectiveDecomposition,
)
from astro_mine.allocate.explain.binding import plan_variable_values
from astro_mine.allocate.model.ir.model import AllocationIR

__all__ = ["decompose_objective"]

#: The family an objective term with no ``objective_family`` metadata tag belongs to — the
#: structural skeleton's per-task value (the info-gain builder tags ``roi`` / ``info_gain``).
_DEFAULT_FAMILY = "value"


def decompose_objective(
    plan: list[AssetSchedule], ir: AllocationIR, *, by_task: bool = False
) -> ObjectiveDecomposition:
    """Decompose ``plan``'s realized objective into its per-family contributions.

    Groups the IR objective terms by their ``objective_family`` metadata tag (default ``value``)
    and sums each group's ``coefficient * variable_value`` under the plan. With ``by_task`` the
    breakdown is per (family, task); otherwise one contribution per family. The contributions are
    sorted (family, task) for determinism, and ``total`` is their sum — equal to the plan's
    realized objective by construction.
    """
    values = plan_variable_values(plan, ir)
    task_of_var = {v.id: v.task_ref for v in ir.variables}

    grouped: dict[tuple[str, str | None], float] = {}
    for term in ir.objective_terms:
        family = ir.metadata.get(f"objective_family::{term.id}", _DEFAULT_FAMILY)
        task_id = task_of_var.get(term.var_ref) if by_task else None
        contribution = term.coefficient * values[term.var_ref]
        grouped[(family, task_id)] = grouped.get((family, task_id), 0.0) + contribution

    contributions = [
        ObjectiveContribution(family=family, value=value, task_id=task_id)
        for (family, task_id), value in sorted(
            grouped.items(), key=lambda kv: (kv[0][0], kv[0][1] or "")
        )
    ]
    return ObjectiveDecomposition(
        total=sum(c.value for c in contributions), contributions=contributions
    )
