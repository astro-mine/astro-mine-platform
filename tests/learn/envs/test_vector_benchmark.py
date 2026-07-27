"""The rollout-throughput benchmark (RM-P1-LEARN-04 AC; learn.md §8) — needs the [rllib] extra.

"Every baseline ships a reproducible throughput benchmark" (learn.md §8; conventions.md §8
"measure before optimizing"), and the AC requires the batched-vs-sequential delta to be
*measured and recorded*. The recorded numbers live in the README; this is the harness that
produces them, so it has to be exercised — a benchmark nobody runs is a benchmark that rots.

Tiny sizes here (the point is the harness, not the number). The real measurement is
``python -m astro_mine.learn.envs.vector.benchmark --num-envs 256 --env-factory …``, and the
``gpu``-marked test in ``test_vector_batched.py`` re-runs it on a real accelerator.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from astro_mine.learn import make_swarm_env
from astro_mine.learn.envs.vector import jax_available
from astro_mine.learn.envs.vector.benchmark import (
    BenchmarkResult,
    ThroughputReport,
    baseline_step,
    rollout_throughput,
)
from tests.learn.fakes import FakeSwarmWorld, build_assets


def _env_factory():
    return make_swarm_env(FakeSwarmWorld(horizon=64), build_assets())


def test_throughput_report_computes_steps_per_second() -> None:
    report = ThroughputReport(
        backend="jax", num_envs=8, steps=4, env_steps=96, wall_clock_s=0.5, device="gpu"
    )
    assert report.steps_per_s == pytest.approx(192.0)
    payload = report.to_dict()
    assert payload["backend"] == "jax"
    assert payload["device"] == "gpu"
    assert payload["env_steps_per_s"] == pytest.approx(192.0)
    assert payload["speedup_vs_sequential"] is None
    # A zero-duration measurement must not divide by zero.
    assert (
        ThroughputReport(
            backend="cpu", num_envs=1, steps=1, env_steps=1, wall_clock_s=0.0
        ).steps_per_s
        == 0.0
    )


def test_baseline_step_drives_the_real_policy() -> None:
    # The benchmark must measure a REAL rollout step (a genuine batched Torch forward), not a
    # numpy stub — otherwise the number it reports is not a training rollout's.
    from astro_mine.learn.train.executor import BatchedStep

    step = baseline_step(_env_factory, algorithm="mappo", hidden=(16, 16))
    assert isinstance(step, BatchedStep)


def test_benchmark_measures_both_backends() -> None:
    result = rollout_throughput(
        _env_factory, baseline_step(_env_factory, hidden=(16, 16)), num_envs=2, steps=2
    )
    assert isinstance(result, BenchmarkResult)
    payload = result.to_dict()

    # The sequential CPU loop is always measured — it is the baseline the delta is against.
    sequential = payload["sequential_cpu"]
    assert sequential["backend"] == "cpu"
    assert sequential["num_envs"] == 2
    assert sequential["env_steps"] > 0
    assert sequential["env_steps_per_s"] > 0

    assert payload["jax_available"] is jax_available()
    if not jax_available():  # pragma: no cover - the [jax] extra IS in the CI sync
        assert payload["batched"] is None
        assert any("[jax] extra is not installed" in note for note in payload["notes"])
        return

    # With [jax], the batched kernel is measured too, and the delta recorded.
    batched = payload["batched"]
    assert batched["backend"] == "jax"
    assert batched["env_steps"] > 0
    assert batched["speedup_vs_sequential"] is not None
    assert batched["speedup_vs_sequential"] > 0
    # The device is reported honestly: a "GPU-vectorized" number measured on the XLA-CPU
    # fallback is a throughput claim nobody should believe, so the report says which it was.
    assert batched["device"] in {"cpu", "gpu", "tpu"}
    if batched["device"] == "cpu":
        assert any("BATCHING alone" in note for note in payload["notes"])
