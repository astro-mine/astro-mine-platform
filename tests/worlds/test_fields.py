"""The chunked N-D Zarr field store (worlds.md §5's format table; issue #39).

Engine-level tests for :mod:`astro_mine.worlds.fields`: the write/read round-trip, the chunking,
the consolidated metadata index, the fail-loud schema guard, and — the load-bearing one — that
:func:`~astro_mine.worlds.fields.zarr_store_hash` is a *content* hash of the store as it lands on
disk, so a tampered chunk moves it (and hence the world hash).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import zarr

from astro_mine.worlds.fields import (
    FIELD_STORE_SCHEMA,
    ZARR_MEDIA_TYPE,
    FieldArray,
    default_chunks,
    read_field_zarr,
    write_field_zarr,
    zarr_store_hash,
    zarr_version,
)


def _horizon(height: int = 12, width: int = 10, n_azimuth: int = 8) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.uniform(0.0, 30.0, size=(height, width, n_azimuth)).astype(np.float32)


def test_round_trip_preserves_bytes_units_and_dims(tmp_path: Path) -> None:
    values = _horizon()
    store = write_field_zarr(
        tmp_path / "horizon.zarr",
        [FieldArray(name="horizon", values=values, units="degree", dims=("y", "x", "azimuth"))],
        attrs={"layer": "illumination/horizon"},
    )
    arrays, attrs = read_field_zarr(store.path)

    # float32 round-trips EXACTLY — this is what keeps illumination_hash stable across a
    # persist/reload (the store is lossless, not a lossy cache).
    assert np.array_equal(arrays["horizon"], values)
    assert arrays["horizon"].dtype == np.float32
    assert attrs["schema"] == FIELD_STORE_SCHEMA
    assert attrs["layer"] == "illumination/horizon"
    assert store.arrays == {"horizon": values.shape}

    group = zarr.open_group(store=str(store.path), mode="r")
    assert group["horizon"].attrs["units"] == "degree"
    assert list(group["horizon"].attrs["dims"]) == ["y", "x", "azimuth"]


def test_store_is_chunked_for_range_reads(tmp_path: Path) -> None:
    """The map plane tiles; the trailing profile axis stays whole (one cell = one chunk read)."""
    values = _horizon(height=300, width=300, n_azimuth=16)
    assert default_chunks(values.shape) == (128, 128, 16)
    store = write_field_zarr(
        tmp_path / "h.zarr",
        [FieldArray(name="horizon", values=values, units="degree", dims=("y", "x", "azimuth"))],
        attrs={},
    )
    group = zarr.open_group(store=str(store.path), mode="r")
    assert group["horizon"].chunks == (128, 128, 16)
    # More than one chunk file actually landed — it is genuinely chunked, not one blob.
    chunk_files = [p for p in store.path.rglob("c/*/*/*") if p.is_file()]
    assert len(chunk_files) > 1


def test_default_chunks_never_exceed_the_array() -> None:
    assert default_chunks((4, 5, 3)) == (4, 5, 3)
    assert default_chunks((2, 400)) == (2, 400)  # non-3D: a single chunk


def test_consolidated_metadata_index_is_written(tmp_path: Path) -> None:
    """worlds.md §5: "field Zarr arrays keep a consolidated metadata index"."""
    store = write_field_zarr(
        tmp_path / "h.zarr",
        [FieldArray(name="horizon", values=_horizon(), units="degree", dims=("y", "x", "az"))],
        attrs={},
    )
    root = zarr.open_group(store=str(store.path), mode="r")
    assert root.metadata.consolidated_metadata is not None
    assert "horizon" in root.metadata.consolidated_metadata.metadata


def test_store_hash_is_deterministic_and_content_addressed(tmp_path: Path) -> None:
    values = _horizon()
    args = (
        [FieldArray(name="horizon", values=values, units="degree", dims=("y", "x", "azimuth"))],
    )
    first = write_field_zarr(tmp_path / "a.zarr", *args, attrs={"layer": "x"})
    second = write_field_zarr(tmp_path / "b.zarr", *args, attrs={"layer": "x"})
    assert first.store_hash == second.store_hash  # two builds -> identical bytes

    changed = values.copy()
    changed[0, 0, 0] += 1.0
    third = write_field_zarr(
        tmp_path / "c.zarr",
        [FieldArray(name="horizon", values=changed, units="degree", dims=("y", "x", "azimuth"))],
        attrs={"layer": "x"},
    )
    assert third.store_hash != first.store_hash  # different content -> different hash


def test_store_hash_moves_when_a_chunk_is_tampered_with(tmp_path: Path) -> None:
    """The hash digests the bytes on disk, so corrupting a chunk is detectable (issue #39)."""
    store = write_field_zarr(
        tmp_path / "h.zarr",
        [FieldArray(name="horizon", values=_horizon(), units="degree", dims=("y", "x", "az"))],
        attrs={},
    )
    chunk = next(p for p in store.path.rglob("c/*/*/*") if p.is_file())
    chunk.write_bytes(chunk.read_bytes() + b"tampered")
    assert zarr_store_hash(store.path) != store.store_hash


def test_reading_a_foreign_store_fails_loudly(tmp_path: Path) -> None:
    """A store without our schema tag is not ours; silently trusting it is the bug to avoid."""
    foreign = tmp_path / "foreign.zarr"
    group = zarr.open_group(store=str(foreign), mode="w")
    group.create_array("horizon", shape=(2, 2), dtype="float32")
    with pytest.raises(ValueError, match="not an Astro-Mine field store"):
        read_field_zarr(foreign)


def test_write_rejects_an_empty_store_and_mismatched_dims(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one array"):
        write_field_zarr(tmp_path / "empty.zarr", [], attrs={})
    with pytest.raises(ValueError, match="dim names"):
        FieldArray(name="h", values=np.zeros((2, 2)), units="m", dims=("y",))


def test_manifest_entry_is_self_describing(tmp_path: Path) -> None:
    store = write_field_zarr(
        tmp_path / "h.zarr",
        [
            FieldArray(
                name="horizon", values=_horizon(4, 4, 3), units="degree", dims=("y", "x", "a")
            )
        ],
        attrs={},
    )
    manifest = store.to_manifest()
    assert manifest["media_type"] == ZARR_MEDIA_TYPE
    assert manifest["store_hash"] == store.store_hash
    assert manifest["arrays"] == {"horizon": [4, 4, 3]}


def test_zarr_version_is_reported() -> None:
    assert zarr_version() == zarr.__version__
