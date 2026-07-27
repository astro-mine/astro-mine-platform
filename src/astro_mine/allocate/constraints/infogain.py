"""``refine_infogain_objective`` — active-perception info-gain vs extraction ROI (RM-P1-ALLOC-04).

The planning-to-learn knob (charter §8; allocate.md §11): an objective family that trades **active
perception** — information gained about the resource field — against **extraction ROI**, deciding
when the swarm should prospect to reduce uncertainty versus commit to extraction. It composes into
the *same* solver-neutral objective as
:func:`~astro_mine.allocate.constraints.refine_value_objective` — no new solver paradigm — so the
CP-SAT backend optimizes the combined
``w_roi * roi + w_info * evpi`` per assignment directly (allocate.md §4).

Information value comes from [Prospect](prospect.md) as **EVPI** (expected value of perfect
information, already denominated in the ROI value units), consumed by **injection** through
:attr:`~astro_mine.allocate.ConstraintContext.info_values` — never a sibling import (allocate.md §6;
RM-P1-ALLOC-04 confirmed decision). EVPI reflects Prospect's *distributional* value (mean **and**
variance), satisfying "distributional, not a point estimate". Absent an injected EVPI, the builder
may fall back to a **posterior-variance proxy** read through the Core ``ResourceField`` contract, a
degraded mode it flags (``"infogain.deterministic"``, parallel to value.py's
``"value.deterministic_mean"``).

The trade weights are the Core ``Objective.weights`` (``"roi"`` / ``"info_gain"``, both defaulting
to 1.0). Info-gain is **opt-in**: it contributes terms only when a weight or an injected value asks
for it, so a request that never mentions info-gain optimizes exactly the ROI objective it did
before. Each objective term is tagged by family in the IR ``metadata``
(``objective_family::{term_id}``) — the additive, byte-stable seam the objective-decomposition
explainability (RM-P1-ALLOC-06) reads the info-gain-vs-ROI split back through.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from astro_mine.allocate.api.model import AllocationRequest, ConstraintContext, Task
from astro_mine.allocate.constraints.config import ConstraintConfig
from astro_mine.allocate.constraints.result import ConstraintFinding
from astro_mine.allocate.enums import VariableSemantic
from astro_mine.allocate.model.ir.model import AllocationIR, ObjectiveTerm

__all__ = ["InfoGainResult", "refine_infogain_objective"]


@dataclass(frozen=True, slots=True)
class InfoGainResult:
    """The info-gain builder's contribution: the combined objective + provenance/honesty notes.

    ``objective_terms`` is the full objective — the ROI terms (weighted by ``w_roi``, tagged
    ``roi``) plus one info-gain term per assignment whose task carries an information value
    (weighted by ``w_info``, tagged ``info_gain``). ``metadata`` records each term's
    ``objective_family`` and the ingested ``info_value`` per task (the RM-P1-ALLOC-06 seam).
    """

    objective_terms: tuple[ObjectiveTerm, ...]
    findings: tuple[ConstraintFinding, ...] = ()
    degraded: tuple[str, ...] = ()
    metadata: dict[str, str] = field(default_factory=dict)


def refine_infogain_objective(
    request: AllocationRequest,
    base_ir: AllocationIR,
    roi_terms: tuple[ObjectiveTerm, ...],
    context: ConstraintContext,
    *,
    config: ConstraintConfig,
    weights: Mapping[str, float],
) -> InfoGainResult:
    """Fold an info-gain term alongside each ROI term, weighted by the Core objective weights."""
    w_roi = float(weights.get("roi", 1.0))
    w_info = float(weights.get("info_gain", 1.0))
    active = "info_gain" in weights or context.info_values is not None

    tasks = {t.task_id: t for t in request.tasks}
    task_of_var = {
        v.id: v.task_ref
        for v in base_ir.variables
        if v.semantic is VariableSemantic.ASSIGNMENT and v.task_ref
    }

    combined: list[ObjectiveTerm] = []
    metadata: dict[str, str] = {}
    findings: list[ConstraintFinding] = []
    degraded: set[str] = set()

    for term in roi_terms:
        combined.append(
            ObjectiveTerm(id=term.id, var_ref=term.var_ref, coefficient=term.coefficient * w_roi)
        )
        metadata[f"objective_family::{term.id}"] = "roi"

    if active:
        for term in roi_terms:
            task_id = task_of_var.get(term.var_ref)
            task = tasks.get(task_id) if task_id else None
            if task is None:
                continue
            info = _info_value(task, context, findings, degraded)
            if info is None:
                continue
            info_id = f"infogain::{term.var_ref}"
            combined.append(
                ObjectiveTerm(id=info_id, var_ref=term.var_ref, coefficient=w_info * info)
            )
            metadata[f"objective_family::{info_id}"] = "info_gain"
            metadata[f"info_value::{task_id}"] = repr(info)

    combined.sort(key=lambda o: o.id)
    return InfoGainResult(
        objective_terms=tuple(combined),
        findings=tuple(findings),
        degraded=tuple(sorted(degraded)),
        metadata=metadata,
    )


def _info_value(
    task: Task,
    context: ConstraintContext,
    findings: list[ConstraintFinding],
    degraded: set[str],
) -> float | None:
    """The information value of prospecting ``task`` — injected EVPI, a variance proxy, or None.

    Primary: an injected per-task EVPI (:attr:`ConstraintContext.info_values`) — a distributional
    value (mean and variance), the RM-P1-ALLOC-04 confirmed path. Fallback: the posterior variance
    at the task's location read through the Core ``ResourceField`` — a degraded uncertainty proxy,
    flagged ``"infogain.deterministic"`` (a raw variance is not a true expected-value-of-perfect-
    information; parallel to value.py collapsing a distribution to its mean).
    """
    if context.info_values is not None:
        evpi = context.info_values.get(task.task_id)
        if evpi is not None:
            findings.append(
                ConstraintFinding(
                    code="infogain.ingested",
                    detail=(
                        f"{task.task_id} info-gain from injected EVPI {evpi:.4g} "
                        "(distributional Prospect value)"
                    ),
                    task_id=task.task_id,
                )
            )
            return evpi

    if context.resource is not None and task.location is not None and task.resource_target_ref:
        center = task.location.center_m
        dist = context.resource.posterior((center.x, center.y, center.z))
        degraded.add("infogain.deterministic")
        findings.append(
            ConstraintFinding(
                code="infogain.variance_proxy",
                detail=(
                    f"{task.task_id} info-gain proxied by posterior variance {dist.variance:.4g} "
                    "(no injected EVPI — degraded)"
                ),
                task_id=task.task_id,
            )
        )
        return dist.variance

    return None
