"""An exported ONNX PolicyPackage is scored through the same rollout as a live policy (LEARN-06).

Needs the [export] extra (onnxruntime) + Torch to train/export; guarded by importorskip so the
no-extra job skips it. Not ray/gpu.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from astro_mine.learn import TrainConfig, default_registry, make_swarm_env
from astro_mine.learn.envs import CommsModel, CommsModelConfig
from astro_mine.learn.eval import partition
from astro_mine.learn.eval.comms_stress import CommsStressGrid, comms_stress_curve
from astro_mine.learn.eval.onnx import OnnxGraph, onnx_policy_id, onnx_policy_under_test
from astro_mine.learn.export import export_policy_packages
from tests.learn.fakes import FakeSwarmWorld, build_assets

_CFG = TrainConfig(seed=1, iterations=1, rollout_steps=8, hidden_sizes=(16, 16))


def _train_and_export():
    env = make_swarm_env(FakeSwarmWorld(), build_assets())
    trainer = default_registry().get("ippo").make_trainer(env, _CFG)
    trainer.train_iteration()
    exports = export_policy_packages(trainer.export(), version="0.1.0")
    graphs = {
        agent: OnnxGraph(package=exp.document.policy_package, onnx_bytes=exp.onnx_bytes)
        for agent, exp in exports.items()
    }
    return graphs, env.agent_specs


def _world(cfg: CommsModelConfig):
    return make_swarm_env(FakeSwarmWorld(), build_assets(), comms_model=CommsModel(cfg))


def test_onnx_package_is_scored_through_the_comms_stress_curve() -> None:
    graphs, specs = _train_and_export()
    policy = onnx_policy_under_test(graphs, specs)
    policy_id = onnx_policy_id(graphs)
    assert policy_id.startswith("sha256:")
    # onnx_policy_id is a deterministic function of the scored artifacts.
    assert onnx_policy_id(graphs) == policy_id

    grid = CommsStressGrid(drop_probabilities=(0.0, 0.5))
    split = partition(base_seed=4, n_train=4, n_eval=2)
    table = comms_stress_curve(
        policy, _world, grid, split, algorithm="ippo_onnx", policy_id=policy_id, steps=8
    )
    # The exported package scored through the same rollout: 2 drop points x 2 seeds.
    assert len(table.rows) == 4
    assert {row.policy_id for row in table.rows} == {policy_id}
    assert {row.algorithm for row in table.rows} == {"ippo_onnx"}


def test_onnx_policy_acts_for_every_agent() -> None:
    graphs, specs = _train_and_export()
    policy = onnx_policy_under_test(graphs, specs)
    env = _world(CommsModelConfig())
    obs, _infos = env.reset(seed=0)
    from astro_mine.learn.algos.policy import flatten_obs

    flat = {a: flatten_obs(obs[a], env.observation_space(a)) for a in env.agents}
    actions = policy.act(flat)
    # Every live agent gets an action sample from its own ONNX graph.
    assert set(actions) == set(env.agents)
    for sample in actions.values():
        assert "kind" in sample
