"""RM-P0-PROSPECT-07 — the calibration gate: credible-interval coverage in CI.

Proves the deliverable and acceptance criteria (prospect.md §10, §12; LUNAR-DR-005):

- a held-out coverage test runs here (the gate), and a **calibrated** belief passes it — the CI
  guard that the belief's stated uncertainty is honest end to end;
- a **deliberately over-confident** belief fails it (and an under-confident one too — the budget is
  two-sided);
- the held-out split is **reproducible** and the coverage budget is **documented** (`DEFAULT_*`);
- reading the sealed truth to build the split is **capability-gated**, and the belief the harness
  produces is agent-safe (carries no ground-truth handle).

The calibrated reference is the **prior belief** (no observations): because the sealed ground truth
is a per-cell draw from that same prior, the prior's credible intervals cover it at the nominal rate
by construction — a robust, non-flaky calibrated baseline. The over/under-confident cases wrap that
same belief and change *only* its stated uncertainty, so the gate's verdict keys on honesty alone.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from astro_mine.core.units import Epoch
from astro_mine.prospect.calibration import (
    DEFAULT_COVERAGE_TOLERANCE,
    DEFAULT_LEVELS,
    HeldOutTruth,
    build_calibration_case,
    check_calibration,
)
from astro_mine.prospect.field import BaseResourceField, FieldGrid, Position
from astro_mine.prospect.isolation import GROUND_TRUTH_ACCESS, IsolationError, assert_isolated
from astro_mine.prospect.priors import load_prior

# A grid large enough that a few-hundred-point held-out set drives the binomial sampling noise of an
# empirical coverage (~sqrt(p(1-p)/n)) well under the budget.
_GRID = FieldGrid(
    min_x_m=-5_000.0, min_y_m=-5_000.0, max_x_m=5_000.0, max_y_m=5_000.0, n_rows=32, n_cols=32
)
_CAPS = (GROUND_TRUTH_ACCESS,)
_PRIOR = load_prior(grid=_GRID)
_N_HOLDOUT = 400


def _calibrated_case(*, seed: int = 11, n_train: int = 0) -> tuple[BaseResourceField, HeldOutTruth]:
    """A calibrated ``(belief, held_out)`` case: the prior belief vs. held-out truth drawn from it.

    ``noise_sigma`` is large relative to the prior's spread, so even with observations the belief
    stays near the (calibrated) prior. ``n_train=0`` (the default) is the pure prior belief.
    """
    belief, held_out = build_calibration_case(
        _PRIOR,
        seed=seed,
        n_train=n_train,
        n_holdout=_N_HOLDOUT,
        noise_sigma=0.05,
        capabilities=_CAPS,
    )
    return belief, held_out


class _ScaledField(BaseResourceField):
    """A belief whose stated predictive std is scaled by ``scale`` around its mean.

    Same mean and predictions; dishonestly narrow (``scale < 1`` → over-confident) or wide
    (``scale > 1`` → under-confident) credible intervals. The cleanest "deliberately mis-calibrated
    belief": only the *uncertainty* changes, so a coverage failure can come from nothing else.
    """

    def __init__(self, inner: BaseResourceField, scale: float) -> None:
        super().__init__(inner.metadata)
        self._inner = inner
        self._scale = scale

    def mean(self, position: Position, *, epoch: Epoch | None = None) -> float:
        return self._inner.mean(position, epoch=epoch)

    def variance(self, position: Position, *, epoch: Epoch | None = None) -> float:
        return self._inner.variance(position, epoch=epoch) * self._scale * self._scale

    def quantile(self, position: Position, q: float, *, epoch: Epoch | None = None) -> float:
        mean = self._inner.mean(position, epoch=epoch)
        return mean + self._scale * (self._inner.quantile(position, q, epoch=epoch) - mean)

    def sample(
        self,
        position: Position,
        *,
        n: int = 1,
        seed: int | None = None,
        epoch: Epoch | None = None,
    ) -> tuple[float, ...]:
        return self._inner.sample(position, n=n, seed=seed, epoch=epoch)


# --- the gate: a calibrated belief passes ----------------------------------------------------


def test_calibrated_belief_passes_the_gate() -> None:
    belief, held_out = _calibrated_case()
    report = check_calibration(belief, held_out)
    assert report.passed
    assert report.max_deviation < DEFAULT_COVERAGE_TOLERANCE
    assert report.max_deviation < 0.08  # comfortable margin, not a knife-edge pass
    assert report.n == _N_HOLDOUT
    assert report.levels == DEFAULT_LEVELS
    assert len(report.coverage) == len(DEFAULT_LEVELS)


# --- the gate has teeth: a mis-calibrated belief fails (acceptance) ---------------------------


def test_overconfident_belief_fails_the_gate() -> None:
    belief, held_out = _calibrated_case()
    overconfident = _ScaledField(belief, 0.01)  # intervals collapse toward the mean
    report = check_calibration(overconfident, held_out)
    assert not report.passed
    # Coverage collapses at every level — the truth almost never lands in the tiny interval.
    assert max(report.coverage) < 0.2


def test_underconfident_belief_fails_the_gate() -> None:
    belief, held_out = _calibrated_case()
    underconfident = _ScaledField(belief, 5.0)  # intervals balloon
    report = check_calibration(underconfident, held_out)
    assert not report.passed
    # Over-coverage: even the 50% interval swallows nearly everything.
    assert report.coverage[0] > 0.8


# --- reliability diagram ----------------------------------------------------------------------


def test_reliability_curve_is_monotone_and_bounded() -> None:
    belief, held_out = _calibrated_case()
    report = check_calibration(belief, held_out)
    coverage = report.coverage
    assert all(0.0 <= c <= 1.0 for c in coverage)
    # Wider nominal intervals can only cover at least as much (levels are ascending).
    assert all(a <= b + 1e-12 for a, b in itertools.pairwise(coverage))


def test_report_reliability_pairs_nominal_with_empirical() -> None:
    belief, held_out = _calibrated_case()
    report = check_calibration(belief, held_out)
    assert report.reliability == tuple(zip(report.levels, report.coverage, strict=True))


def test_check_calibration_is_deterministic() -> None:
    belief, held_out = _calibrated_case()
    assert check_calibration(belief, held_out) == check_calibration(belief, held_out)


# --- the held-out split: reproducible, gated, agent-safe --------------------------------------


def test_build_calibration_case_is_reproducible() -> None:
    a_belief, a_held = build_calibration_case(
        _PRIOR, seed=3, n_train=64, n_holdout=200, noise_sigma=0.05, capabilities=_CAPS
    )
    b_belief, b_held = build_calibration_case(
        _PRIOR, seed=3, n_train=64, n_holdout=200, noise_sigma=0.05, capabilities=_CAPS
    )
    assert np.array_equal(a_held.values, b_held.values)
    assert a_held.positions == b_held.positions
    assert a_belief.content_hash == b_belief.content_hash

    _c_belief, c_held = build_calibration_case(
        _PRIOR, seed=4, n_train=64, n_holdout=200, noise_sigma=0.05, capabilities=_CAPS
    )
    assert not np.array_equal(a_held.values, c_held.values)


def test_build_calibration_case_conditions_on_training_observations() -> None:
    prior_belief, _ = build_calibration_case(
        _PRIOR, seed=5, n_train=0, n_holdout=100, noise_sigma=0.05, capabilities=_CAPS
    )
    assert prior_belief.log == ()  # no observations → the belief is exactly the prior

    conditioned, held_out = build_calibration_case(
        _PRIOR, seed=5, n_train=48, n_holdout=100, noise_sigma=0.05, capabilities=_CAPS
    )
    assert len(conditioned.log) == 48  # the training cells became belief observations
    assert len(held_out) == 100


def test_build_calibration_case_requires_ground_truth_access() -> None:
    # The harness is privileged; without GROUND_TRUTH_ACCESS no truth is read (LUNAR-DR-005).
    with pytest.raises(IsolationError, match="ground_truth_access"):
        build_calibration_case(
            _PRIOR, seed=1, n_train=8, n_holdout=50, noise_sigma=0.05, capabilities=()
        )


def test_calibration_belief_carries_no_ground_truth_handle() -> None:
    # Built from sealed truth, yet the belief is agent-safe — no GroundTruthField is reachable.
    _, _ = _calibrated_case()
    prior_belief, _ = _calibrated_case(n_train=0)
    conditioned, _ = _calibrated_case(n_train=32)
    assert_isolated(prior_belief)
    assert_isolated(conditioned)


# --- fail-loud guards -------------------------------------------------------------------------


def test_build_calibration_case_rejects_a_bad_split() -> None:
    n_cells = _GRID.n_rows * _GRID.n_cols
    with pytest.raises(ValueError, match="exceeds the grid"):
        build_calibration_case(
            _PRIOR, seed=1, n_train=n_cells, n_holdout=1, noise_sigma=0.05, capabilities=_CAPS
        )
    with pytest.raises(ValueError, match="n_holdout > 0"):
        build_calibration_case(
            _PRIOR, seed=1, n_train=10, n_holdout=0, noise_sigma=0.05, capabilities=_CAPS
        )
    with pytest.raises(ValueError, match="n_train >= 0"):
        build_calibration_case(
            _PRIOR, seed=1, n_train=-1, n_holdout=10, noise_sigma=0.05, capabilities=_CAPS
        )


def test_check_calibration_rejects_bad_arguments() -> None:
    belief, held_out = _calibrated_case()
    with pytest.raises(ValueError, match="tolerance must be positive"):
        check_calibration(belief, held_out, tolerance=0.0)
    with pytest.raises(ValueError, match="at least one"):
        check_calibration(belief, held_out, levels=())
    with pytest.raises(ValueError, match="open interval"):
        check_calibration(belief, held_out, levels=(0.5, 1.0))
    with pytest.raises(ValueError, match="held_out is empty"):
        check_calibration(belief, HeldOutTruth(positions=(), values=np.zeros(0)))
    with pytest.raises(ValueError, match="one value per position"):
        check_calibration(
            belief, HeldOutTruth(positions=((0.0, 0.0, 0.0),), values=np.zeros((2, 2)))
        )
