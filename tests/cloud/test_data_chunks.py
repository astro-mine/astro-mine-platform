"""Lazy chunk-range reads -- Zarr chunk objects and byte-range datasets, no bulk copy."""

from __future__ import annotations

import json

import pytest

from astro_mine.cloud.data.chunks import ChunkRef, RangeDataset, ZarrArray, read_chunk


class RecordingStore:
    """An in-memory ObjectStore that records every access, to prove reads are surgical."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self._objects = objects
        self.calls: list[tuple[str, str, int | None, int | None]] = []

    def get(self, key: str) -> bytes:
        self.calls.append(("get", key, None, None))
        return self._objects[key]

    def get_range(self, key: str, start: int, length: int) -> bytes:
        self.calls.append(("get_range", key, start, length))
        return self._objects[key][start : start + length]

    def size(self, key: str) -> int:
        return len(self._objects[key])


def test_read_chunk_whole_vs_range() -> None:
    store = RecordingStore({"blob": b"0123456789"})
    assert read_chunk(store, ChunkRef(key="blob")) == b"0123456789"
    assert read_chunk(store, ChunkRef(key="blob", offset=2, length=3)) == b"234"
    assert store.calls == [("get", "blob", None, None), ("get_range", "blob", 2, 3)]


def test_range_ref_needs_a_length() -> None:
    with pytest.raises(ValueError, match="needs a length"):
        read_chunk(RecordingStore({}), ChunkRef(key="k", offset=5))


def test_chunk_ref_is_range_flag() -> None:
    assert ChunkRef(key="k").is_range is False
    assert ChunkRef(key="k", offset=0, length=4).is_range is True


def test_range_dataset_chunk_ref_lookup() -> None:
    dataset = RangeDataset(key="f", chunks=(ChunkRef(key="f", offset=0, length=4),))
    assert dataset.chunk_ref(0).offset == 0


# --- Zarr ----------------------------------------------------------------------------

ZARRAY = json.dumps({"shape": [100, 100], "chunks": [10, 10], "dimension_separator": "."})


def test_zarr_grid_and_chunk_key() -> None:
    array = ZarrArray.from_metadata("field", ZARRAY)
    assert array.grid_shape() == (10, 10)
    assert array.chunk_ref([2, 3]).key == "field/2.3"


def test_zarr_reads_only_the_requested_chunk_object() -> None:
    objects = {f"field/{i}.{j}": f"chunk-{i}-{j}".encode() for i in range(10) for j in range(10)}
    store = RecordingStore(objects)
    array = ZarrArray.from_metadata("field", ZARRAY)
    assert array.read_chunk(store, [4, 7]) == b"chunk-4-7"
    # exactly one object fetched -- no bulk copy of the 100-chunk array
    assert store.calls == [("get", "field/4.7", None, None)]


def test_zarr_index_validation() -> None:
    array = ZarrArray.from_metadata("field", ZARRAY)
    with pytest.raises(ValueError, match="wrong rank"):
        array.chunk_ref([1])
    with pytest.raises(IndexError, match="out of range"):
        array.chunk_ref([10, 0])


def test_zarr_custom_dimension_separator() -> None:
    meta = json.dumps({"shape": [20], "chunks": [10], "dimension_separator": "/"})
    array = ZarrArray.from_metadata("a/", meta)
    assert array.chunk_ref([1]).key == "a/1"


# --- RangeDataset (COG / Parquet) ----------------------------------------------------


def test_range_dataset_streams_only_the_tile() -> None:
    data = bytes(range(64))
    store = RecordingStore({"cog.tif": data})
    dataset = RangeDataset(
        key="cog.tif",
        chunks=(
            ChunkRef(key="cog.tif", offset=0, length=16),
            ChunkRef(key="cog.tif", offset=16, length=16),
        ),
    )
    assert dataset.read_chunk(store, 1) == data[16:32]
    assert store.calls == [("get_range", "cog.tif", 16, 16)]
