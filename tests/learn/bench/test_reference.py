"""Reference-score + honest-eval harness (RM-P1-LEARN-03; LEARN-06 seam).

:func:`evaluate` is covered Torch-free (a seeded reference policy drives it); the full
:func:`reference_score` needs the ``[rllib]`` extra to train a baseline. A ``slow``-marked
placeholder documents the out-of-CI real anchor-scenario reference (needs Sim).
"""

from __future__ import annotations

import pytest

from astro_mine.learn import make_reference_policy, make_swarm_env
from astro_mine.learn.bench import EvalReport, evaluate
from astro_mine.learn.envs import CommsModel, CommsModelConfig, DropConfig
from tests.learn.fakes import FakeSwarmWorld, build_assets


def _factory():
    return make_swarm_env(FakeSwarmWorld(), build_assets())


def test_evaluate_reports_held_out_seed_variance() -> None:
    # Torch-free: a seeded reference policy over a seed sweep (the honest-eval seam).
    policy = make_reference_policy(_factory().agent_specs, seed=0)
    report = evaluate(policy, _factory, seeds=(100, 101, 102), steps=16)
    assert isinstance(report, EvalReport)
    assert report.seeds == (100, 101, 102)
    assert len(report.returns) == 3
    assert report.std_return >= 0.0
    assert report.throughput_steps_per_s > 0.0


def test_evaluate_is_reproducible() -> None:
    policy = make_reference_policy(_factory().agent_specs, seed=0)
    first = evaluate(policy, _factory, seeds=(5, 6), steps=8)
    second = evaluate(policy, _factory, seeds=(5, 6), steps=8)
    assert first.returns == second.returns


def test_evaluate_records_comms_stress_ledger() -> None:
    def comms_factory():
        cfg = CommsModelConfig(drop=DropConfig(probability=0.5))
        return make_swarm_env(FakeSwarmWorld(), build_assets(), comms_model=CommsModel(cfg))

    policy = make_reference_policy(comms_factory().agent_specs, seed=0)
    report = evaluate(policy, comms_factory, seeds=(1,), steps=12)
    # The comms-stress denominator: a per-seed comms-budget ledger is recorded.
    assert len(report.comms_stress) == 1
    assert report.comms_stress[0]  # non-empty ledger under a degraded channel


@pytest.mark.ray
def test_reference_score_trains_and_evaluates() -> None:
    pytest.importorskip("torch")
    from astro_mine.learn.algos import TrainConfig, default_registry
    from astro_mine.learn.bench import reference_score

    config = TrainConfig(iterations=2, rollout_steps=8, hidden_sizes=(16, 16))
    report = reference_score(
        default_registry().get("ippo"), _factory, config, eval_seeds=(100, 101)
    )
    assert report.algorithm == "ippo"
    assert len(report.learning_curve) == 2
    assert report.train_throughput_steps_per_s > 0.0
    assert len(report.evaluation.returns) == 2


@pytest.mark.slow
def test_real_anchor_scenario_reference_is_out_of_ci() -> None:
    # The real lunar-polar-prospecting reference score needs a Sim-backed env (Sim is
    # un-importable from Learn) and an overnight run; it is a DOCUMENTED out-of-CI artifact
    # (learn.md §8, §12). CI runs only the FakeSwarmWorld smoke reference above.
    pytest.skip("real anchor-scenario reference score requires astro_mine.sim (RM-P1-LEARN-04+)")
