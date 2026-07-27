"""SweepSpec -- deterministic expansion of grid / random / halton parameter spaces."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from astro_mine.cloud.packaging import ImageRef
from astro_mine.cloud.submission.jobspec import JobSpec
from astro_mine.cloud.submission.sweepspec import SWEEP_INDEX_ENV, SweepSpec

IMAGE = ImageRef.parse("ghcr.io/astro-mine/x@sha256:" + "ef" * 32)
BASE = JobSpec(image=IMAGE, command=["run"], env={"KEEP": "1"}, seed=100)


def test_grid_is_the_cartesian_product() -> None:
    sweep = SweepSpec(base=BASE, grid={"lr": [0.1, 0.2], "bs": [16, 32]})
    variants = sweep.expand()
    assert sweep.size() == 4
    assert len(variants) == 4
    # each variant keeps the base env, injects params as strings, and stamps its index
    first = variants[0]
    assert first.env["KEEP"] == "1"
    assert first.env["lr"] == "0.1"
    assert first.env["bs"] == "16"
    assert first.env[SWEEP_INDEX_ENV] == "0"


def test_grid_derives_a_distinct_reproducible_seed_per_variant() -> None:
    sweep = SweepSpec(base=BASE, grid={"lr": [0.1, 0.2, 0.3]})
    seeds = [v.seed for v in sweep.expand()]
    assert seeds == [100, 101, 102]


def test_grid_with_no_base_seed_leaves_seed_none() -> None:
    base = JobSpec(image=IMAGE, command=["run"])
    variants = SweepSpec(base=base, grid={"lr": [0.1, 0.2]}).expand()
    assert [v.seed for v in variants] == [None, None]


def test_random_is_deterministic_for_a_fixed_seed() -> None:
    sweep = SweepSpec(base=BASE, method="random", ranges={"lr": (0.0, 1.0)}, samples=5, seed=7)
    assert [v.env["lr"] for v in sweep.expand()] == [v.env["lr"] for v in sweep.expand()]
    assert sweep.size() == 5


def test_halton_is_deterministic_and_spread() -> None:
    sweep = SweepSpec(
        base=BASE, method="halton", ranges={"a": (0.0, 1.0), "b": (0.0, 10.0)}, samples=8
    )
    points = [v.env["a"] for v in sweep.expand()]
    assert points == [v.env["a"] for v in sweep.expand()]  # reproducible
    assert len(set(points)) == 8  # low-discrepancy: no repeats


def test_grid_validation() -> None:
    with pytest.raises(ValidationError, match="non-empty `grid`"):
        SweepSpec(base=BASE, grid={})
    with pytest.raises(ValidationError, match="no values"):
        SweepSpec(base=BASE, grid={"lr": []})
    with pytest.raises(ValidationError, match="not `ranges`"):
        SweepSpec(base=BASE, grid={"lr": [1]}, ranges={"x": (0.0, 1.0)})


def test_sampling_validation() -> None:
    with pytest.raises(ValidationError, match="requires `ranges`"):
        SweepSpec(base=BASE, method="random", samples=3)
    with pytest.raises(ValidationError, match="positive `samples`"):
        SweepSpec(base=BASE, method="random", ranges={"x": (0.0, 1.0)})
    with pytest.raises(ValidationError, match="low < high"):
        SweepSpec(base=BASE, method="random", ranges={"x": (1.0, 0.0)}, samples=2)
    with pytest.raises(ValidationError, match="not `grid`"):
        SweepSpec(base=BASE, method="random", ranges={"x": (0.0, 1.0)}, samples=2, grid={"y": [1]})


def test_halton_dimension_limit() -> None:
    ranges = {f"p{i}": (0.0, 1.0) for i in range(25)}
    with pytest.raises(ValidationError, match="dimensions"):
        SweepSpec(base=BASE, method="halton", ranges=ranges, samples=2)
