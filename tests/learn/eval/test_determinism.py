"""The comms-stress curve determinism gate (CX-REPRO; RM-P1-LEARN-06).

Same policy + grid + held-out seeds ⇒ byte-identical returns/ledgers **and** an identical
manifest hash; a different seed changes the curve. The Torch-free gate runs in CI on
FakeSwarmWorld; the cross-algorithm comparison over real trained baselines is ``ray``-marked
(runs in CI with the [rllib] extra); the real anchor-scenario Sim-backed sweep is ``slow``
(deselected, documented out of CI — Sim is un-importable from Learn).
"""

from __future__ import annotations

import pytest

from astro_mine.learn import make_reference_policy, make_swarm_env
from astro_mine.learn.envs import CommsModel, CommsModelConfig
from astro_mine.learn.eval import partition
from astro_mine.learn.eval.comms_stress import (
    CommsStressGrid,
    comms_stress_curve,
    comms_stress_curves,
)
from tests.learn.fakes import FakeSwarmWorld, build_assets


def _world(cfg: CommsModelConfig):
    return make_swarm_env(FakeSwarmWorld(), build_assets(), comms_model=CommsModel(cfg))


def _specs():
    return _world(CommsModelConfig()).agent_specs


def _ref_curve(grid, split):
    # Fresh policy(seed=0) each call: the reference policy's sampling RNG is stateful, so a
    # rebuilt policy reproduces the exact call sequence — the determinism contract.
    policy = make_reference_policy(_specs(), seed=0)
    return comms_stress_curve(policy, _world, grid, split, steps=16)


def _fingerprint(table):
    return [
        (
            r.algorithm,
            r.stress_axis,
            r.stress_value,
            r.seed,
            r.episode_return,
            r.delivery_ratio,
            r.offered,
            r.delivered,
            r.comms_config_hash,
        )
        for r in table.rows
    ]


def test_same_policy_grid_seeds_reproduce_curve_and_manifest_hash() -> None:
    grid = CommsStressGrid(drop_probabilities=(0.0, 0.4, 0.8))
    split = partition(base_seed=7, n_train=4, n_eval=3)
    first = _ref_curve(grid, split)
    second = _ref_curve(grid, split)
    assert first.manifest_hash == second.manifest_hash
    assert _fingerprint(first) == _fingerprint(second)


def test_a_different_seed_changes_the_curve_and_the_hash() -> None:
    grid = CommsStressGrid(drop_probabilities=(0.0, 0.4, 0.8))
    a = _ref_curve(grid, partition(7, 4, 3))
    b = _ref_curve(grid, partition(11, 4, 3))
    # A different held-out split ⇒ a different manifest (hash) and a different realization.
    assert a.manifest_hash != b.manifest_hash
    assert _fingerprint(a) != _fingerprint(b)


@pytest.mark.ray
def test_curves_are_comparable_across_trained_baselines() -> None:
    pytest.importorskip("torch")
    from astro_mine.learn import TrainConfig, default_registry

    cfg = TrainConfig(seed=1, iterations=1, rollout_steps=8, hidden_sizes=(16, 16))
    registry = default_registry()
    policies = {}
    for tag in ("ippo", "mappo", "qmix"):
        trainer = registry.get(tag).make_trainer(_world(CommsModelConfig()), cfg)
        trainer.train_iteration()
        policies[tag] = trainer.policy()

    grid = CommsStressGrid(drop_probabilities=(0.0, 0.5))
    split = partition(base_seed=2, n_train=4, n_eval=2)
    table = comms_stress_curves(policies, _world, grid, split, steps=8)

    # ippo/mappo/qmix scored on the identical axes + seeds — a comparable cross-algorithm curve.
    assert {r.algorithm for r in table.rows} == {"ippo", "mappo", "qmix"}
    coords = {
        algo: sorted((r.stress_value, r.seed) for r in table.rows if r.algorithm == algo)
        for algo in policies
    }
    assert coords["ippo"] == coords["mappo"] == coords["qmix"]
    # The determinism gate holds over trained baselines too (identical manifest on rerun).
    again = comms_stress_curves(policies, _world, grid, split, steps=8)
    assert again.manifest_hash == table.manifest_hash


@pytest.mark.slow
def test_real_anchor_scenario_comms_stress_sweep_is_out_of_ci() -> None:
    # The real lunar-polar-prospecting comms-stress sweep needs a Sim-backed Core Environment
    # (Sim is un-importable from Learn) and an overnight run; it is a DOCUMENTED out-of-CI
    # artifact (learn.md §8, §10). CI runs the Torch-free + ray-trained curves above.
    pytest.skip("real anchor-scenario comms-stress sweep requires astro_mine.sim (RM-P1-LEARN-04+)")
