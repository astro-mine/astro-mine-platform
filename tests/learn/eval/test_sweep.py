"""Seed sweeps report variance + reject single-seed results (RM-P1-LEARN-06)."""

from __future__ import annotations

import pytest

from astro_mine.learn import make_reference_policy, make_swarm_env
from astro_mine.learn.bench import EvalReport, ReferenceReport
from astro_mine.learn.eval.sweep import SweepReport, sample_efficiency, seed_sweep
from tests.learn.fakes import FakeSwarmWorld, build_assets


def _factory():
    return make_swarm_env(FakeSwarmWorld(), build_assets())


def _empty_eval() -> EvalReport:
    return EvalReport(
        seeds=(), returns=(), mean_return=0.0, std_return=0.0, throughput_steps_per_s=0.0
    )


def test_seed_sweep_reports_mean_variance_and_cost() -> None:
    policy = make_reference_policy(_factory().agent_specs, seed=0)
    report = seed_sweep(policy, _factory, seeds=(100, 101, 102), steps=16)
    assert isinstance(report, SweepReport)
    assert report.seeds == (100, 101, 102)
    assert len(report.returns) == 3
    assert report.std_return >= 0.0
    assert report.eval_throughput_steps_per_s > 0.0
    assert report.wall_clock_s >= 0.0
    assert report.sample_efficiency is None  # no training reference supplied
    assert isinstance(report.report, EvalReport)


def test_single_seed_is_rejected() -> None:
    policy = make_reference_policy(_factory().agent_specs, seed=0)
    with pytest.raises(ValueError, match="single-seed result is an anti-pattern"):
        seed_sweep(policy, _factory, seeds=(1,), steps=8)
    # Distinct-seed count, not length, is what matters.
    with pytest.raises(ValueError, match="anti-pattern"):
        seed_sweep(policy, _factory, seeds=(5, 5), steps=8)


def test_min_seeds_is_configurable() -> None:
    policy = make_reference_policy(_factory().agent_specs, seed=0)
    # A single seed is allowed only when the caller explicitly lowers the floor.
    report = seed_sweep(policy, _factory, seeds=(1,), steps=8, min_seeds=1)
    assert report.seeds == (1,)


def test_sample_efficiency_from_reference_learning_curve() -> None:
    policy = make_reference_policy(_factory().agent_specs, seed=0)
    reference = ReferenceReport(
        algorithm="ippo",
        learning_curve=(0.0, 1.0, 2.0),  # trapezoid area = 0.5 + 1.5 = 2.0
        train_throughput_steps_per_s=10.0,
        evaluation=_empty_eval(),
    )
    assert sample_efficiency(reference) == pytest.approx(2.0)
    report = seed_sweep(policy, _factory, seeds=(1, 2), steps=8, reference=reference)
    assert report.sample_efficiency == pytest.approx(2.0)


def test_sample_efficiency_degenerate_curves() -> None:
    empty = ReferenceReport(
        algorithm="x",
        learning_curve=(),
        train_throughput_steps_per_s=0.0,
        evaluation=_empty_eval(),
    )
    assert sample_efficiency(empty) == 0.0
    single = ReferenceReport(
        algorithm="x",
        learning_curve=(3.0,),
        train_throughput_steps_per_s=0.0,
        evaluation=_empty_eval(),
    )
    assert sample_efficiency(single) == 3.0
