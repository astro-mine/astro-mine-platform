"""Immutable, content-addressed dataset store round-trips (RM-P1-SURR-03; surrogate.md §5).

Zarr (particle arrays) + Parquet (tabular config) write→read, with the content hash stable across
the storage layer, immutability enforced (no overwrite), and a fail-closed integrity check on read
(conventions.md §5: datasets are immutable and content-addressed).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from astro_mine.surrogate.datagen import (
    DatasetRef,
    SamplingPolicy,
    generate_dataset,
    read_dataset,
    reference_rollout_oracle,
    split_dataset,
    write_dataset,
)
from astro_mine.surrogate.models.dataset import DemDataset
from astro_mine.surrogate.report import Bound

_BOUNDS = {
    "density": Bound(low=1400.0, high=1600.0),
    "friction": Bound(low=0.4, high=0.7),
    "restitution": Bound(low=0.2, high=0.4),
    "tool_speed": Bound(low=0.05, high=0.08),
}


def _dataset(seed: int = 0, n: int = 6) -> DemDataset:
    policy = SamplingPolicy(parameter_bounds=_BOUNDS, n_initial=n, pool_size=8, n_per_round=2)
    return generate_dataset(policy, reference_rollout_oracle, seed=seed)


def test_write_read_round_trip_preserves_arrays_and_hash(tmp_path) -> None:
    dataset = _dataset()
    ref = write_dataset(dataset, tmp_path, name="dem-al", version="0.1.0")
    assert isinstance(ref, DatasetRef)
    assert ref.content_hash == dataset.content_hash()

    back = read_dataset(ref)
    # Content hash is stable regardless of Zarr chunking / Parquet encoding.
    assert back.content_hash() == dataset.content_hash()
    assert np.array_equal(back.states, dataset.states)
    assert np.array_equal(back.tool_x, dataset.tool_x)
    assert np.array_equal(back.params, dataset.params)
    assert back.param_names == dataset.param_names
    assert back.dt_s == dataset.dt_s


def test_ref_records_distinct_train_and_val_split_hashes(tmp_path) -> None:
    ref = write_dataset(_dataset(), tmp_path, name="dem-al", version="0.1.0")
    assert ref.train_split_hash != ref.val_split_hash
    assert ref.train_split_hash != ref.content_hash
    train, validation = split_dataset(_dataset())
    assert ref.train_split_hash == train.content_hash()
    assert ref.val_split_hash == validation.content_hash()


def test_identical_arrays_hash_identically_but_different_data_differs(tmp_path) -> None:
    a = _dataset(seed=0)
    b = _dataset(seed=0)
    c = _dataset(seed=1)
    assert a.content_hash() == b.content_hash()  # deterministic
    assert a.content_hash() != c.content_hash()  # different data


def test_overwrite_is_refused_immutability(tmp_path) -> None:
    dataset = _dataset()
    write_dataset(dataset, tmp_path, name="dem-al", version="0.1.0")
    with pytest.raises(FileExistsError):
        write_dataset(dataset, tmp_path, name="dem-al", version="0.1.0")
    # A new version is allowed alongside the prior (never overwritten).
    other = write_dataset(dataset, tmp_path, name="dem-al", version="0.2.0")
    assert other.version == "0.2.0"


def test_read_is_fail_closed_on_a_tampered_content_hash(tmp_path) -> None:
    ref = write_dataset(_dataset(), tmp_path, name="dem-al", version="0.1.0")
    tampered = replace(ref, content_hash="sha256:" + "00" * 32)
    with pytest.raises(ValueError, match="tampered or truncated"):
        read_dataset(tampered)


def test_split_dataset_refuses_a_single_config() -> None:
    one = _dataset(n=2)
    one = DemDataset(
        states=one.states[:1],
        tool_x=one.tool_x[:1],
        params=one.params[:1],
        dt_s=one.dt_s,
        bed_width_m=one.bed_width_m,
        tool_height_m=one.tool_height_m,
        feature_names=one.feature_names,
        param_names=one.param_names,
    )
    with pytest.raises(ValueError, match="cannot split"):
        split_dataset(one)


def test_the_store_does_not_put_the_zarr_encoding_into_the_dataset_address(tmp_path) -> None:
    """The dataset's address must come from its arrays, never from how Zarr encoded them (#19).

    This is the property the zarr 2 -> 3 bump risked. If the content hash tracked the on-disk
    chunking or codec, a dependency bump would silently re-address every published fixture — and the
    trust region pinned against it (surrogate#17).

    Note what is *not* asserted: a literal digest. The fixture is produced by a numeric rollout, so
    its bytes — and so its hash — are machine-sensitive (CI and a dev box legitimately disagree). A
    hard-coded expectation would test the FPU, not the store. What is invariant, and what is checked
    here, is that the address is a pure function of the arrays: it is computed *before* Zarr ever
    sees them, survives the encode/decode, and is what the fail-closed reader verifies against.
    """
    import zarr

    assert zarr.__version__.startswith("3."), "the workspace is on one Zarr major now (#19)"

    dataset = _dataset(n=4)
    from_arrays = dataset.content_hash()

    ref = write_dataset(dataset, tmp_path / "ds", name="probe", version="0.0.1")
    assert ref.content_hash == from_arrays

    # read_dataset is fail-closed on the hash, so returning at all proves the arrays survived the
    # Zarr round-trip byte-for-byte; re-deriving the address from the *decoded* arrays proves the
    # encoding contributed nothing to it.
    assert read_dataset(ref).content_hash() == from_arrays
