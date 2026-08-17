# SPDX-License-Identifier: Apache-2.0
"""Deterministic, no-LLM structured intent authoring (studio.md §3 ``intent/forms``).

The MVP intent front door: turn an :class:`~astro_mine.studio.models.IntentDraft`
(goal + region + assets + constraints + objectives) into a Core-validated
``ObjectiveSpec``. This path is **always available** and never requires a model — it is
the guarantee against which the optional LLM adapter (RM-P1-STUDIO-05) is tested.

The projection is total and deterministic: each target product becomes a soft, weighted
success criterion (a rolling window when it states a rate); each hard constraint becomes
a *required* criterion with a pass/fail threshold. An objective with no criteria at all
is rejected by the Core ``ObjectiveSpec`` schema (``success_criteria`` min length 1) —
i.e. malformed intent fails loudly at construction and is never persisted.
"""

from __future__ import annotations

from astro_mine.core.objective import (
    EvaluationWindow,
    MetricBinding,
    ObjectiveDocument,
    ObjectiveSpec,
    SuccessCriterion,
    WindowKind,
)
from astro_mine.core.units import require_crs

from ..models import IntentDraft


def build_objective(draft: IntentDraft) -> ObjectiveDocument:
    """Project an ``IntentDraft`` into a Core ``ObjectiveDocument``.

    Raises a Core/pydantic validation error if the draft yields no success criteria
    (the boundary reject), and fails loudly if the region CRS is not an explicit
    planetary CRS (``require_crs``)."""
    require_crs(draft.region.crs)

    criteria: list[SuccessCriterion] = []
    for product in draft.products:
        window = (
            EvaluationWindow(kind=WindowKind.ROLLING, duration_s=product.rate_window_s)
            if product.rate_window_s is not None
            else None
        )
        criteria.append(
            SuccessCriterion(
                id=product.criterion_id,
                binding=MetricBinding(
                    metric=product.metric,
                    unit=product.unit,
                    direction=product.direction,
                    target=product.target,
                    tolerance=product.tolerance,
                    evaluation_window=window,
                ),
                required=True,
                weight=product.weight,
            )
        )
    for constraint in draft.constraints:
        criteria.append(
            SuccessCriterion(
                id=constraint.criterion_id,
                binding=MetricBinding(
                    metric=constraint.metric,
                    unit=constraint.unit,
                    direction=constraint.direction,
                    target=constraint.threshold,
                    tolerance=0.0,
                    threshold=constraint.threshold,
                ),
                required=True,
            )
        )

    labels = dict(draft.labels)
    labels.setdefault("region.name", draft.region.name)
    labels.setdefault("region.body", draft.region.crs.body)

    spec = ObjectiveSpec(
        id=draft.id,
        name=draft.name,
        description=draft.description,
        scenario_ref=draft.scenario_ref,
        success_criteria=criteria,
        labels=labels,
    )
    return ObjectiveDocument(objective_version="0.1", objective=spec)
