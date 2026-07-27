"""Held-out / training seed separation is enforced by construction (RM-P1-LEARN-06)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from astro_mine.learn.eval import HeldOutSplit, partition
from astro_mine.learn.eval.split import _HELD_OUT_SEED_SALT, _TRAIN_SEED_SALT
from astro_mine.learn.train.executor import derive_seeds


def test_partition_yields_disjoint_reproducible_sets() -> None:
    split = partition(base_seed=7, n_train=8, n_eval=4)
    assert len(split.train_seeds) == 8
    assert len(split.held_out_seeds) == 4
    # Disjoint by construction.
    assert not (split.train_seeds & set(split.held_out_seeds))
    # Reproducible: same (base, sizes) ⇒ same seeds.
    again = partition(base_seed=7, n_train=8, n_eval=4)
    assert again.train_seeds == split.train_seeds
    assert again.held_out_seeds == split.held_out_seeds
    # A different base draws a different split.
    other = partition(base_seed=8, n_train=8, n_eval=4)
    assert other.held_out_seeds != split.held_out_seeds


def test_held_out_seeds_are_namespaced_away_from_distributed_slice_seeds() -> None:
    # derive_seeds salts distributed rollout slices as SeedSequence([seed, index]); the
    # held-out draw uses a distinct salt namespace, so a training run's per-worker seeds can
    # never collide with a held-out seed.
    split = partition(base_seed=0, n_train=16, n_eval=8)
    slice_seeds = set(derive_seeds(0, 16))
    assert not (set(split.held_out_seeds) & slice_seeds)
    assert _TRAIN_SEED_SALT != _HELD_OUT_SEED_SALT


def test_overlapping_split_cannot_be_constructed() -> None:
    with pytest.raises(ValidationError, match="separated from training seeds"):
        HeldOutSplit(train_seeds=frozenset({1, 2, 3}), held_out_seeds=(3, 4))


def test_empty_or_duplicate_held_out_rejected() -> None:
    with pytest.raises(ValidationError, match="at least one held-out seed"):
        HeldOutSplit(train_seeds=frozenset({1}), held_out_seeds=())
    with pytest.raises(ValidationError, match="must be unique"):
        HeldOutSplit(train_seeds=frozenset({1}), held_out_seeds=(9, 9))


def test_assert_holds_out_rejects_a_trained_on_seed() -> None:
    split = HeldOutSplit(train_seeds=frozenset({1, 2}), held_out_seeds=(100, 101))
    split.assert_holds_out(1, 2)  # training seeds are fine
    with pytest.raises(ValueError, match="would not be held out"):
        split.assert_holds_out(100)


def test_partition_rejects_nonpositive_counts() -> None:
    with pytest.raises(ValueError, match="count must be >= 1"):
        partition(base_seed=1, n_train=0, n_eval=1)
