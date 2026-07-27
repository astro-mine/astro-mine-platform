"""The promotion gate — coverage/calibration + error budget as first-class objects (RM-P1-SURR-03).

A retrained surrogate enters Sim **only** through the automated validation gate (surrogate.md §10,
§11 "Offline retrain + gated promotion ... weights enter Sim only through the validation gate"; the
issue's acceptance criterion). This module extracts that predicate — previously implicit in the
calibration tests — into first-class objects so the retrain harness (and a human) can read *why* a
model was or was not admitted:

- **Coverage/calibration** — every channel's calibration curve must hold: a nominal ``p`` interval's
  empirical coverage must be ``>= min_coverage_ratio * p - coverage_slack`` (a 90% interval covers
  ~90%, surrogate.md §10). An over-confident surrogate (empirical << nominal) fails.
- **Error budget** — the surrogate's declared ``recommended_error_budget`` and its measured error
  (continuous ``rmse``; categorical ``1 - accuracy``) must both be within the consumer's required
  per-channel budget. A model whose error exceeds the required tolerance cannot be promoted.

Pure Core + Pydantic types (:class:`~astro_mine.surrogate.report.ErrorReport`) — no numpy, no torch,
no ``[datasets]`` extra: the gate reads the calibrated report a surrogate already carries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from astro_mine.surrogate.enums import ChannelKind
from astro_mine.surrogate.report import ChannelError, CoveragePoint, ErrorReport

__all__ = [
    "BudgetVerdict",
    "CoverageVerdict",
    "GateResult",
    "PromotionCriteria",
    "evaluate_promotion",
]


@dataclass(frozen=True)
class PromotionCriteria:
    """The thresholds a retrained surrogate must clear to be admitted into Sim.

    ``min_coverage_ratio`` times the nominal coverage (minus ``coverage_slack`` for finite-sample
    noise) is the empirical coverage each calibration point must reach. ``error_budget`` maps an
    output channel to the maximum error tolerance the consumer requires; a channel absent from the
    map is gated against the surrogate's own ``recommended_error_budget`` (a self-consistency
    check that always holds for a calibrated report).
    """

    min_coverage_ratio: float = 0.9
    coverage_slack: float = 0.05
    error_budget: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CoverageVerdict:
    """One calibration point's verdict: does empirical coverage clear the required floor."""

    channel: str
    nominal: float
    empirical: float
    required: float
    passed: bool


@dataclass(frozen=True, slots=True)
class BudgetVerdict:
    """One channel's error-budget verdict: declared budget + measured error vs the required."""

    channel: str
    measured_error: float
    recommended_budget: float
    required_budget: float
    passed: bool


@dataclass(frozen=True, slots=True)
class GateResult:
    """The gate outcome: the pass/fail plus every verdict and the human-readable failure reasons.

    ``passed`` is ``True`` only if every coverage and budget verdict passed. ``reasons`` names each
    failing verdict so a gate failure is actionable (why the model was not promoted) — empty when
    ``passed`` is ``True``.
    """

    passed: bool
    coverage_verdicts: tuple[CoverageVerdict, ...]
    budget_verdicts: tuple[BudgetVerdict, ...]
    reasons: tuple[str, ...]


def _calibration_points(channel: ChannelError) -> list[CoveragePoint]:
    """A channel's calibration curve: interval coverage (continuous) or reliability (else)."""
    if channel.kind is ChannelKind.CONTINUOUS:
        assert channel.continuous is not None  # kind invariant (report.ChannelError validator)
        return list(channel.continuous.coverage)
    assert channel.categorical is not None
    return list(channel.categorical.reliability)


def _measured_error(channel: ChannelError) -> float:
    """The headline error the budget gates on: continuous ``rmse``; categorical error rate."""
    if channel.kind is ChannelKind.CONTINUOUS:
        assert channel.continuous is not None
        return channel.continuous.rmse
    assert channel.categorical is not None
    return 1.0 - channel.categorical.accuracy


def evaluate_promotion(report: ErrorReport, criteria: PromotionCriteria) -> GateResult:
    """Evaluate ``report`` against ``criteria`` — the coverage + error-budget promotion predicate.

    Produces a :class:`CoverageVerdict` per calibration point and a :class:`BudgetVerdict` per
    channel, and passes only if every verdict passes. This is the gate the retrain harness consults
    before admitting a model: a gate failure means the retrained weights never enter Sim
    (continual data collection is still allowed — only promotion is gated).
    """
    coverage_verdicts: list[CoverageVerdict] = []
    budget_verdicts: list[BudgetVerdict] = []
    reasons: list[str] = []

    for channel in report.channels:
        for point in _calibration_points(channel):
            required = criteria.min_coverage_ratio * point.nominal - criteria.coverage_slack
            passed = point.empirical >= required
            coverage_verdicts.append(
                CoverageVerdict(
                    channel=channel.channel,
                    nominal=point.nominal,
                    empirical=point.empirical,
                    required=required,
                    passed=passed,
                )
            )
            if not passed:
                reasons.append(
                    f"channel {channel.channel!r} nominal {point.nominal:g} coverage: empirical "
                    f"{point.empirical:g} below required {required:g} (over-confident)"
                )

        recommended = report.substitution_policy.recommended_error_budget.get(channel.channel)
        if recommended is None:
            continue
        required_budget = criteria.error_budget.get(channel.channel, recommended)
        measured = _measured_error(channel)
        passed = measured <= required_budget and recommended <= required_budget
        budget_verdicts.append(
            BudgetVerdict(
                channel=channel.channel,
                measured_error=measured,
                recommended_budget=recommended,
                required_budget=required_budget,
                passed=passed,
            )
        )
        if not passed:
            reasons.append(
                f"channel {channel.channel!r} error budget: measured {measured:g} / recommended "
                f"{recommended:g} exceeds required {required_budget:g} (over-budget)"
            )

    passed = all(v.passed for v in coverage_verdicts) and all(v.passed for v in budget_verdicts)
    return GateResult(
        passed=passed,
        coverage_verdicts=tuple(coverage_verdicts),
        budget_verdicts=tuple(budget_verdicts),
        reasons=tuple(reasons),
    )
