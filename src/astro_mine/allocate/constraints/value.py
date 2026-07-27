"""``refine_value_objective`` — Prospect resource value with uncertainty (RM-P1-ALLOC-03).

Ingests [Prospect](prospect.md)'s resource-value field as an objective input: a task that acts on a
resource target has its objective coefficient scaled by the posterior **mean abundance** at its
location, read through the Core uncertainty-first
:class:`~astro_mine.core.resource.protocol.ResourceField` contract (distributions, not point
guesses). Because the RM-P1-ALLOC-01 objective is a single linear scalarization, collapsing the
posterior to its mean discards the variance — a legitimate but *degraded* mode (robust/stochastic
formulations over the full distribution are P1-late/P2). The build flags it, and the discarded
variance is recorded in the IR metadata so a plan is honest about the uncertainty it ignored.

No sibling import: the value truth arrives only through the Core ``ResourceField`` in the
:class:`~astro_mine.allocate.ConstraintContext` (allocate.md §6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from astro_mine.allocate.api.model import AllocationRequest, ConstraintContext
from astro_mine.allocate.constraints.config import ConstraintConfig
from astro_mine.allocate.constraints.result import ConstraintFinding
from astro_mine.allocate.enums import VariableSemantic
from astro_mine.allocate.model.ir.model import AllocationIR, ObjectiveTerm

__all__ = ["ValueResult", "refine_value_objective"]


@dataclass(frozen=True, slots=True)
class ValueResult:
    """The value builder's contribution: refined objective terms + recorded uncertainty."""

    objective_terms: tuple[ObjectiveTerm, ...]
    findings: tuple[ConstraintFinding, ...] = ()
    degraded: tuple[str, ...] = ()
    #: ``value_variance::{task_id}`` → repr(variance) metadata folded into the IR (auditability).
    metadata: dict[str, str] = field(default_factory=dict)


def refine_value_objective(
    request: AllocationRequest,
    base_ir: AllocationIR,
    context: ConstraintContext,
    *,
    config: ConstraintConfig,
) -> ValueResult:
    """Scale objective coefficients by Prospect posterior abundance and record the uncertainty."""
    if context.resource is None:
        return ValueResult(objective_terms=tuple(base_ir.objective_terms))

    field_ = context.resource
    tasks = {t.task_id: t for t in request.tasks}
    # task_id -> assignment var ids, to map an objective term back to its task.
    task_of_var = {
        v.id: v.task_ref
        for v in base_ir.variables
        if v.semantic is VariableSemantic.ASSIGNMENT and v.task_ref
    }

    findings: list[ConstraintFinding] = []
    degraded: set[str] = set()
    metadata: dict[str, str] = {}
    refined: list[ObjectiveTerm] = []

    for term in base_ir.objective_terms:
        task_id = task_of_var.get(term.var_ref)
        task = tasks.get(task_id) if task_id else None
        if task is None or task.location is None or task.resource_target_ref is None:
            refined.append(term)
            continue

        center = task.location.center_m
        dist = field_.posterior((center.x, center.y, center.z))
        refined.append(
            ObjectiveTerm(
                id=term.id, var_ref=term.var_ref, coefficient=term.coefficient * dist.mean
            )
        )
        metadata[f"value_variance::{task_id}"] = repr(dist.variance)
        degraded.add("value.deterministic_mean")
        findings.append(
            ConstraintFinding(
                code="value.ingested",
                detail=(
                    f"{task_id} value scaled by posterior abundance mean {dist.mean:.4g} "
                    f"(variance {dist.variance:.4g} discarded — deterministic mode)"
                ),
                task_id=task_id,
            )
        )

    refined.sort(key=lambda o: o.id)
    return ValueResult(
        objective_terms=tuple(refined),
        findings=tuple(findings),
        degraded=tuple(sorted(degraded)),
        metadata=metadata,
    )
