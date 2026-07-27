"""Comms-stress curves: swept, degrading with drop, comparable across algorithms (LEARN-06)."""

from __future__ import annotations

from collections import defaultdict

import pyarrow.parquet as pq
import pytest

from astro_mine.learn import make_reference_policy, make_swarm_env
from astro_mine.learn.envs import CommsModel, CommsModelConfig
from astro_mine.learn.eval import HeldOutSplit, ParquetSink, partition
from astro_mine.learn.eval.comms_stress import (
    CommsStressGrid,
    build_curve_manifest,
    comms_stress_curve,
    comms_stress_curves,
)
from tests.learn.fakes import FakeSwarmWorld, build_assets


def _world(cfg: CommsModelConfig):
    return make_swarm_env(FakeSwarmWorld(), build_assets(), comms_model=CommsModel(cfg))


def _specs():
    return _world(CommsModelConfig()).agent_specs


def test_grid_enumerates_every_swept_axis() -> None:
    grid = CommsStressGrid(drop_probabilities=(0.0, 0.5), delay_ticks=(1, 4), budgets=(2.0,))
    points = grid.points()
    assert [p.axis for p in points] == ["drop", "drop", "delay", "delay", "budget"]
    assert points[0].config.drop.probability == 0.0
    assert points[2].config.delay.kind == "fixed" and points[2].config.delay.ticks == 1
    assert points[4].config.bandwidth.per_agent_bits_per_tick == 2.0
    # JSON-Schema-emitting, like CommsModelConfig.
    assert grid.model_json_schema()["title"] == "CommsStressGrid"


def test_delivery_ratio_degrades_as_drop_rises() -> None:
    grid = CommsStressGrid(drop_probabilities=(0.0, 0.3, 0.9))
    split = partition(base_seed=3, n_train=4, n_eval=3)
    policy = make_reference_policy(_specs(), seed=0)
    table = comms_stress_curve(policy, _world, grid, split, algorithm="ref", steps=16)

    # Aggregate delivery_ratio per drop value: sum(delivered) / sum(offered).
    offered: dict[float, int] = defaultdict(int)
    delivered: dict[float, int] = defaultdict(int)
    for row in table.rows:
        assert row.stress_axis == "drop"
        assert row.split == "held_out"
        offered[row.stress_value] += row.offered
        delivered[row.stress_value] += row.delivered

    def ratio(value: float) -> float:
        return 1.0 if offered[value] == 0 else delivered[value] / offered[value]

    assert ratio(0.0) == 1.0  # identity channel: nothing degraded
    assert offered[0.9] > 0 and offered[0.3] > 0  # a degraded channel actually offers links
    assert ratio(0.9) < ratio(0.3) < 1.0  # more drop ⇒ fewer delivered


def test_curves_are_comparable_across_algorithms() -> None:
    # Torch-free cross-algorithm structure: two seeded reference policies on the identical
    # grid + split + seeds yield aligned, algorithm-keyed rows.
    grid = CommsStressGrid(drop_probabilities=(0.0, 0.5))
    split = partition(base_seed=1, n_train=4, n_eval=2)
    policies = {
        "ref_a": make_reference_policy(_specs(), seed=0),
        "ref_b": make_reference_policy(_specs(), seed=1),
    }
    table = comms_stress_curves(policies, _world, grid, split, steps=12)

    by_algo = defaultdict(list)
    for row in table.rows:
        by_algo[row.algorithm].append((row.stress_axis, row.stress_value, row.seed))
    assert set(by_algo) == {"ref_a", "ref_b"}
    # Identical (axis, value, seed) coordinates per algorithm — a comparable curve.
    assert sorted(by_algo["ref_a"]) == sorted(by_algo["ref_b"])
    # policy_id defaults to live:<name>.
    assert {r.policy_id for r in table.rows} == {"live:ref_a", "live:ref_b"}
    # 2 policies x 2 drop points x 2 seeds.
    assert len(table.rows) == 2 * 2 * 2


def test_single_held_out_seed_is_rejected_by_the_curve() -> None:
    grid = CommsStressGrid(drop_probabilities=(0.5,))
    split = HeldOutSplit(train_seeds=frozenset({1, 2}), held_out_seeds=(100,))
    policy = make_reference_policy(_specs(), seed=0)
    with pytest.raises(ValueError, match="anti-pattern"):
        comms_stress_curve(policy, _world, grid, split, steps=8)


def test_curve_is_emitted_through_a_sink(tmp_path) -> None:
    grid = CommsStressGrid(drop_probabilities=(0.0, 0.5))
    split = partition(base_seed=5, n_train=4, n_eval=2)
    policy = make_reference_policy(_specs(), seed=0)
    out = tmp_path / "curve.parquet"
    table = comms_stress_curve(policy, _world, grid, split, steps=8, sink=ParquetSink(out))
    # The sink wrote the returned table to Parquet, resolvable back by Bench.
    assert out.exists()
    read = pq.read_table(out)
    assert read.num_rows == len(table.rows)
    assert set(read.column("manifest_hash").to_pylist()) == {table.manifest_hash}


def test_build_curve_manifest_captures_the_reproducibility_key() -> None:
    grid = CommsStressGrid(drop_probabilities=(0.0, 0.5))
    split = partition(base_seed=2, n_train=4, n_eval=2)
    manifest = build_curve_manifest(
        policy_ids={"ref": "live:ref"}, grid=grid, split=split, seeds=split.held_out_seeds, steps=16
    )
    assert manifest["kind"] == "comms_stress_curve"
    assert manifest["policies"] == {"ref": "live:ref"}
    assert manifest["split"]["held_out_seeds"] == list(split.held_out_seeds)
    assert manifest["core_interfaces"]["policy"] == "0.1.0"
