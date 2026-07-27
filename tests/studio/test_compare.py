"""RM-P1-STUDIO-06 — the comparison view surfaces uncertainty rather than hiding it."""

from __future__ import annotations

from astro_mine.studio.compare import build_comparison
from astro_mine.studio.models import (
    CandidateScore,
    DesignCandidate,
    EvaluatedCandidate,
    TradeStudy,
)
from astro_mine.studio.orchestrate import LOCAL_STAND_IN_EVALUATOR_ID
from astro_mine.studio.provenance import capture_provenance


def _evaluated(
    candidate_id: str,
    scores: dict[str, float],
    uncertainty: dict[str, float] | None = None,
    *,
    seed: int = 1,
) -> EvaluatedCandidate:
    return EvaluatedCandidate(
        candidate=DesignCandidate(id=candidate_id, swarm=[]),
        score=CandidateScore(
            objective_hash="sha256:obj",
            metric_scores=scores,
            metric_uncertainty=uncertainty or {},
            aggregate=sum(scores.values()),
            passed=True,
        ),
        seed=seed,
        world_ref="sha256:world",
        provenance=capture_provenance(input_hashes=["sha256:in"], seed=seed),
    )


def _study(*evaluated: EvaluatedCandidate, front: list[str] | None = None) -> TradeStudy:
    return TradeStudy(
        id="ts-1",
        objective_hash="sha256:obj",
        backend="random",
        evaluator=LOCAL_STAND_IN_EVALUATOR_ID,
        seeds=[1],
        evaluated=list(evaluated),
        pareto_front=front if front is not None else [evaluated[0].candidate.id],
        provenance=capture_provenance(input_hashes=["sha256:in"], seed=1),
    )


def test_carries_the_bound_alongside_the_point_estimate() -> None:
    study = _study(_evaluated("a", {"water": 10.0}, {"water": 1.5}))
    view = build_comparison(study)

    estimate = view.candidates[0].metrics["water"]
    assert estimate.value == 10.0
    assert estimate.uncertainty == 1.5
    assert view.has_complete_uncertainty


def test_a_missing_bound_is_none_not_zero() -> None:
    """`None` means the dispersion was never measured; `0.0` would claim it is exactly known."""
    study = _study(_evaluated("a", {"water": 10.0}))
    view = build_comparison(study)

    assert view.candidates[0].metrics["water"].uncertainty is None
    assert not view.has_complete_uncertainty


def test_partially_bounded_studies_are_reported_as_such() -> None:
    study = _study(
        _evaluated("a", {"water": 10.0, "power": 2.0}, {"water": 1.0}),
        _evaluated("b", {"water": 8.0, "power": 3.0}, {"water": 0.5, "power": 0.2}),
    )
    view = build_comparison(study)
    assert not view.has_complete_uncertainty
    assert view.candidates[0].metrics["power"].uncertainty is None
    assert view.candidates[1].metrics["power"].uncertainty == 0.2


def test_axes_are_sorted_and_stable() -> None:
    """Parallel-coordinates axes must not reorder between requests."""
    study = _study(_evaluated("a", {"zeta": 1.0, "alpha": 2.0, "mu": 3.0}))
    assert build_comparison(study).metrics == ["alpha", "mu", "zeta"]


def test_the_front_is_the_studys_own_not_recomputed() -> None:
    """The Pareto front is part of the reproducible artifact, not a view-time computation."""
    study = _study(
        _evaluated("a", {"water": 1.0}),
        _evaluated("b", {"water": 99.0}),
        front=["a"],  # deliberately not the better-scoring candidate
    )
    view = build_comparison(study)

    assert view.pareto_front == ["a"]
    assert [c.on_pareto_front for c in view.candidates] == [True, False]


def test_a_candidate_missing_a_metric_omits_it_rather_than_imputing_zero() -> None:
    study = _study(
        _evaluated("a", {"water": 10.0, "power": 1.0}),
        _evaluated("b", {"water": 8.0}),
    )
    view = build_comparison(study)

    assert view.metrics == ["power", "water"]
    assert set(view.candidates[1].metrics) == {"water"}


def test_a_study_with_no_candidates_yields_an_empty_view() -> None:
    """Vacuously complete: there is nothing whose uncertainty could be missing."""
    study = TradeStudy(
        id="ts-empty",
        objective_hash="sha256:obj",
        backend="random",
        evaluator=LOCAL_STAND_IN_EVALUATOR_ID,
        seeds=[1],
        evaluated=[],
        pareto_front=[],
        provenance=capture_provenance(input_hashes=["sha256:in"], seed=1),
    )
    view = build_comparison(study)

    assert view.metrics == [] and view.candidates == [] and view.pareto_front == []
    assert view.has_complete_uncertainty
