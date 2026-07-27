"""The promotion gate: coverage/calibration + error budget (RM-P1-SURR-03; surrogate.md §10, §11).

A calibrated report is admitted; an over-confident one (empirical coverage << nominal) and an
over-budget one (error beyond the required tolerance) are refused — the gate a retrained model must
clear before its weights enter Sim.
"""

from __future__ import annotations

from astro_mine.surrogate import (
    ChannelError,
    ChannelKind,
    ContinuousMetrics,
    CoveragePoint,
    SubstitutionPolicy,
    TailBehavior,
)
from astro_mine.surrogate.eval import PromotionCriteria, evaluate_promotion
from tests.surrogate.factories import granular_report, illumination_report


def _continuous_channel(
    name: str, *, nominal: float, empirical: float, rmse: float
) -> ChannelError:
    return ChannelError(
        channel=name,
        kind=ChannelKind.CONTINUOUS,
        continuous=ContinuousMetrics(
            unit="N",
            rmse=rmse,
            coverage=[CoveragePoint(nominal=nominal, empirical=empirical)],
            tail=TailBehavior(p95_abs_error=rmse, p99_abs_error=rmse * 2, max_abs_error=rmse * 3),
        ),
    )


def test_a_calibrated_continuous_report_passes() -> None:
    result = evaluate_promotion(granular_report(), PromotionCriteria())
    assert result.passed
    assert result.reasons == ()
    assert all(v.passed for v in result.coverage_verdicts)
    assert all(v.passed for v in result.budget_verdicts)


def test_a_calibrated_categorical_report_passes_via_reliability_and_accuracy() -> None:
    result = evaluate_promotion(illumination_report(), PromotionCriteria())
    assert result.passed
    # The categorical channel is gated on reliability (calibration) and 1 - accuracy (budget).
    budget = {v.channel: v for v in result.budget_verdicts}
    assert budget["visibility"].measured_error == 1.0 - 0.97


def test_an_over_confident_report_fails_the_coverage_gate() -> None:
    report = granular_report(
        channels=[_continuous_channel("reaction_force_n", nominal=0.9, empirical=0.5, rmse=1.5)]
    )
    result = evaluate_promotion(report, PromotionCriteria())
    assert not result.passed
    assert any("over-confident" in reason for reason in result.reasons)


def test_an_over_budget_report_fails_the_budget_gate() -> None:
    # rmse 1.5 / recommended 2.0 both exceed a required tolerance of 1.0.
    criteria = PromotionCriteria(error_budget={"reaction_force_n": 1.0})
    result = evaluate_promotion(granular_report(), criteria)
    assert not result.passed
    assert any("over-budget" in reason for reason in result.reasons)
    verdict = next(v for v in result.budget_verdicts if v.channel == "reaction_force_n")
    assert verdict.required_budget == 1.0
    assert not verdict.passed


def test_a_channel_without_a_declared_budget_is_only_coverage_gated() -> None:
    report = granular_report(
        channels=[
            _continuous_channel("reaction_force_n", nominal=0.9, empirical=0.9, rmse=1.5),
            _continuous_channel("torque_nm", nominal=0.9, empirical=0.9, rmse=0.4),
        ],
        substitution_policy=SubstitutionPolicy(recommended_error_budget={"reaction_force_n": 3.0}),
    )
    result = evaluate_promotion(report, PromotionCriteria())
    assert result.passed
    # No budget verdict for the un-budgeted channel; coverage still checked for both.
    assert {v.channel for v in result.budget_verdicts} == {"reaction_force_n"}
    assert {v.channel for v in result.coverage_verdicts} == {"reaction_force_n", "torque_nm"}


def test_coverage_slack_absorbs_finite_sample_noise() -> None:
    # empirical 0.78 for nominal 0.9: below 0.9*0.9=0.81 but within the default 0.05 slack (>=0.76).
    report = granular_report(
        channels=[_continuous_channel("reaction_force_n", nominal=0.9, empirical=0.78, rmse=1.5)]
    )
    assert evaluate_promotion(report, PromotionCriteria()).passed
    # A stricter, zero-slack gate rejects it.
    assert not evaluate_promotion(report, PromotionCriteria(coverage_slack=0.0)).passed
