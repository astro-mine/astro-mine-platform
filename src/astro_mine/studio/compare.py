"""The comparison view: a Pareto front that shows what it does not know (RM-P1-STUDIO-06).

studio.md §2 principle 7 is unambiguous — "**Uncertainty is shown, not hidden.** … the comparison UI
surfaces [error bounds] rather than presenting single point estimates as truth." A surrogate-pruned
or low-fidelity evaluation carries a per-metric bound alongside its estimate, and the front end
draws it (Plotly ``error_y``, or a band).

The one design decision here is what to do when a metric has **no** recorded bound. A missing
bound is not a zero bound: `None` means *we never measured the dispersion*, and rendering that
as a point estimate with a hairline error bar would assert a precision nobody claimed. So the model
carries
``uncertainty: float | None``, the flag :attr:`ComparisonView.has_complete_uncertainty` says whether
every rendered point has one, and the UI is expected to mark the missing ones rather than draw them
as exact.

Studio computes nothing (studio.md §2 principle 1): every number here was produced by Bench and
arrives through the injected ``Scorer`` seam. This module reshapes a ``TradeStudy`` for display.
"""

from __future__ import annotations

from ._base import FrozenStudioModel
from .models import TradeStudy

__all__ = [
    "ComparisonCandidate",
    "ComparisonView",
    "MetricEstimate",
    "build_comparison",
]


class MetricEstimate(FrozenStudioModel):
    """One metric's score, and how well it is known.

    ``uncertainty`` is ``None`` when the evaluation recorded no dispersion for this metric — a
    different statement from ``0.0``, and the UI must not conflate them.
    """

    value: float
    uncertainty: float | None = None


class ComparisonCandidate(FrozenStudioModel):
    """One evaluated candidate, as the trade-off plots read it."""

    candidate_id: str
    seed: int
    aggregate: float
    passed: bool
    on_pareto_front: bool
    metrics: dict[str, MetricEstimate]


class ComparisonView(FrozenStudioModel):
    """A trade study reshaped for the Pareto scatter and parallel-coordinates plots."""

    study_id: str
    objective_hash: str
    backend: str
    #: What produced the metric values. Carried onto the view because the chart is otherwise
    #: pixel-identical whether a stand-in or real physics scored it, and the surface must say which.
    evaluator: str
    #: Sorted, so the parallel-coordinates axes are stable between requests.
    metrics: list[str]
    candidates: list[ComparisonCandidate]
    pareto_front: list[str]

    @property
    def front_is_degenerate(self) -> bool:
        """Whether the front contains every candidate — no candidate dominates any other.

        Not a result. It is what the stand-in evaluator produces by construction, and a surface
        that reports "N candidates, N on the front" without saying so presents a tautology as a
        finding."""
        return bool(self.candidates) and len(self.pareto_front) == len(self.candidates)

    @property
    def has_complete_uncertainty(self) -> bool:
        """Whether every point carries a bound. When false, the UI must say which do not."""
        return all(
            estimate.uncertainty is not None
            for candidate in self.candidates
            for estimate in candidate.metrics.values()
        )


def build_comparison(study: TradeStudy) -> ComparisonView:
    """Project a frozen trade study onto the comparison UI's shape.

    The Pareto front is the study's own — Studio does not re-rank it here, because the front is part
    of the reproducible artifact (``TradeStudy.pareto_front``), not a view-time computation.
    """
    metrics: set[str] = set()
    for evaluated in study.evaluated:
        metrics.update(evaluated.score.metric_scores)
    axes = sorted(metrics)
    front = set(study.pareto_front)

    candidates = [
        ComparisonCandidate(
            candidate_id=evaluated.candidate.id,
            seed=evaluated.seed,
            aggregate=evaluated.score.aggregate,
            passed=evaluated.score.passed,
            on_pareto_front=evaluated.candidate.id in front,
            metrics={
                metric: MetricEstimate(
                    value=evaluated.score.metric_scores[metric],
                    uncertainty=evaluated.score.metric_uncertainty.get(metric),
                )
                for metric in axes
                if metric in evaluated.score.metric_scores
            },
        )
        for evaluated in study.evaluated
    ]

    return ComparisonView(
        study_id=study.id,
        objective_hash=study.objective_hash,
        backend=study.backend,
        evaluator=study.evaluator,
        metrics=axes,
        candidates=candidates,
        pareto_front=list(study.pareto_front),
    )
