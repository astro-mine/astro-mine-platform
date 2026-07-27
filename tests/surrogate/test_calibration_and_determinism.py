"""The §10 validation gates: calibrated coverage, error budget, determinism (RM-P1-SURR-02).

Calibration/coverage and the error budget gate promotion — an over-confident surrogate fails.
Determinism is the reproducibility contract: same seed + fixture reproduce the surrogate in-process
(torch CPU is not bit-portable across builds, so this is a tolerance gate, not a bit-exact golden).
"""

from __future__ import annotations

import numpy as np
import pytest

from astro_mine.surrogate.models.dataset import load_dem_dataset


def test_ninety_percent_intervals_cover_about_ninety_percent(surrogate) -> None:
    # The calibration/coverage gate: a calibrated 90% interval contains truth ~90% of the time.
    for channel in surrogate.error_report.channels:
        point = channel.continuous.coverage[0]
        assert point.nominal == 0.9
        assert point.empirical >= 0.8  # finite-sample slack; an over-confident model fails this


def test_error_budget_is_published_and_positive(surrogate) -> None:
    budget = surrogate.error_report.substitution_policy.recommended_error_budget
    assert set(budget) == {"pos_x", "pos_z", "vel_x", "vel_z"}
    assert all(v > 0.0 for v in budget.values())


def test_the_budget_is_a_max_over_the_rollout_horizon_sim_grades_at(surrogate) -> None:
    """The budget must bound the statistic Sim enforces (a **max**, surrogate#21) over the
    **horizon** it grades at (a rollout, surrogate#23) — not an RMSE, and not a single step.

    Sim re-validates with ``abs(surrogate - reference).max()`` over the bed, after the surrogate
    has rolled up to ``revalidate_every`` steps on its own predictions (Sim re-anchors that often).
    Two prior budgets each broke on exactly one of these axes and each blocked astro-mine-bench#31:

    - ``2 x RMSE`` — a *central* statistic where Sim enforces a *max*. An RMSE bounds nothing; half
      the 90-particle bed exceeds it by construction (#21).
    - ``1.5 x single-step max`` — the right statistic at the *wrong horizon*. It held on step 1 and
      was breached a few steps into the rollout, because a step surrogate drifts on its own output
      (#23).
    """
    from astro_mine.surrogate.models.excavation import _BUDGET_HORIZON_STEPS, _rollout_budget

    sp = surrogate.error_report.substitution_policy

    # The horizon is declared, and it is the one the budget was calibrated at.
    assert sp.budget_horizon_steps == _BUDGET_HORIZON_STEPS

    # And the budget *is* the rollout budget — pins the calibration so a quiet revert to a
    # single-step or an RMSE budget is caught, not just tolerated.
    expected = _rollout_budget(
        surrogate._models,
        surrogate._dataset,
        surrogate._normalizer,
        horizon=_BUDGET_HORIZON_STEPS,
    )
    for channel, declared in sp.recommended_error_budget.items():
        assert declared == pytest.approx(expected[channel])

    for channel in surrogate.error_report.channels:
        tail = channel.continuous.tail
        declared = sp.recommended_error_budget[channel.channel]
        # A rollout max is at least the single-step max, so the budget covers the one-step worst
        # case too — never below what Sim would see even on a fresh bed.
        assert declared >= tail.max_abs_error
        # ...and it remains a max, not a mean: 2 x RMSE sits *below* the single-step max on a bed
        # this size, so a revert to a central statistic is caught here (#21).
        assert 2.0 * channel.continuous.rmse < tail.max_abs_error


def test_every_channel_ships_a_finite_bounded_error(surrogate) -> None:
    for channel in surrogate.error_report.channels:
        metrics = channel.continuous
        assert metrics is not None
        assert np.isfinite(metrics.rmse) and metrics.rmse >= 0.0
        assert (
            metrics.tail.max_abs_error >= metrics.tail.p99_abs_error >= metrics.tail.p95_abs_error
        )


def test_same_seed_builds_reproduce_within_tolerance() -> None:
    from astro_mine.surrogate.models import TrainConfig, build_excavation_surrogate

    cfg = TrainConfig(hidden=16, message_passing_steps=1, epochs=5, ensemble_size=1)
    a = build_excavation_surrogate(config=cfg, seed=7)
    b = build_excavation_surrogate(config=cfg, seed=7)
    ds = load_dem_dataset()
    query = {
        "position": ds.states[0, 0, :, :2],
        "velocity": ds.states[0, 0, :, 2:],
        "tool_x": np.array([ds.tool_x[0, 0]]),
        "config": ds.params[0],
    }
    pa, pb = a.predict(query), b.predict(query)
    assert np.allclose(pa.fields["position"], pb.fields["position"])
    assert np.allclose(pa.field_uncertainty["velocity"], pb.field_uncertainty["velocity"])
    # the ErrorReport's per-channel RMSEs reproduce too
    ra = [c.continuous.rmse for c in a.error_report.channels]
    rb = [c.continuous.rmse for c in b.error_report.channels]
    assert np.allclose(ra, rb)
