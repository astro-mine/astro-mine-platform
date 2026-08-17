# SPDX-License-Identifier: Apache-2.0
"""Lazy chunk-range reads over Zarr / COG / Parquet -- never bulk-copy a dataset.

A worker streams only the chunks it needs from object storage (``cloud.md`` §2 principle 7,
§5, §8). A :class:`ChunkRef` names one chunk: either a whole chunk *object* (Zarr, whose
chunks are separate keys) or a byte *range* within a single object (a COG tile, a Parquet
row-group). :func:`read_chunk` fetches exactly that -- a whole-object get or a range get, never
the whole dataset. :class:`ZarrArray` derives the chunk layout from a Zarr ``.zarray`` header
(pure index math); :class:`RangeDataset` reads byte-range chunks from an explicit layout (a
COG IFD / Parquet footer supplies the offsets -- that format parse is the producer's, we do
the streaming).

Backlog: RM-P1-CLOUD-04 -- astro-mine-cloud#15
"""

from __future__ import annotations

import json
import math
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from collections.abc import Sequence

    from astro_mine.cloud.data.objectstore import ObjectStore

__all__ = ["ChunkRef", "RangeDataset", "ZarrArray", "read_chunk"]


class ChunkRef(BaseModel):
    """A pointer to one chunk: a whole object (``offset`` unset) or a byte range within one."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    offset: int | None = None
    length: int | None = None

    @property
    def is_range(self) -> bool:
        """Whether this ref is a byte range (vs a whole chunk object)."""
        return self.offset is not None


def read_chunk(store: ObjectStore, ref: ChunkRef) -> bytes:
    """Fetch exactly one chunk: a whole-object get, or a byte-range get -- nothing more."""
    if ref.offset is None:
        return store.get(ref.key)
    if ref.length is None:
        raise ValueError("a range ChunkRef needs a length")
    return store.get_range(ref.key, ref.offset, ref.length)


class ZarrArray(BaseModel):
    """A Zarr v2 array whose chunks are separate objects -- read one chunk at a time.

    Built from the array's ``.zarray`` metadata; :meth:`chunk_ref` maps a chunk-grid index to
    the chunk object's key (dot-joined, per the ``dimension_separator``), so a read fetches a
    single chunk object -- inherently "only the slice you need", no bulk copy.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    shape: tuple[int, ...]
    chunks: tuple[int, ...] = Field(description="chunk shape per dimension")
    dimension_separator: str = "."

    @classmethod
    def from_metadata(cls, path: str, zarray_json: bytes | str) -> ZarrArray:
        """Parse a Zarr ``.zarray`` metadata document into a :class:`ZarrArray`."""
        meta = json.loads(zarray_json)
        return cls(
            path=path.rstrip("/"),
            shape=tuple(meta["shape"]),
            chunks=tuple(meta["chunks"]),
            dimension_separator=meta.get("dimension_separator", "."),
        )

    def grid_shape(self) -> tuple[int, ...]:
        """Number of chunks along each dimension."""
        return tuple(math.ceil(s / c) for s, c in zip(self.shape, self.chunks, strict=True))

    def chunk_ref(self, index: Sequence[int]) -> ChunkRef:
        """The :class:`ChunkRef` (whole-object) for the chunk at grid *index*."""
        grid = self.grid_shape()
        if len(index) != len(grid):
            raise ValueError(f"index {tuple(index)} has wrong rank for grid {grid}")
        if any(not 0 <= i < n for i, n in zip(index, grid, strict=True)):
            raise IndexError(f"chunk index {tuple(index)} out of range for grid {grid}")
        key = self.path + "/" + self.dimension_separator.join(str(i) for i in index)
        return ChunkRef(key=key)

    def read_chunk(self, store: ObjectStore, index: Sequence[int]) -> bytes:
        """Fetch the single chunk object at grid *index* from *store*."""
        return read_chunk(store, self.chunk_ref(index))


class RangeDataset(BaseModel):
    """A single object (a COG or Parquet file) sliced into byte-range chunks by an index.

    The chunk layout -- offset/length per tile or row-group -- comes from the file's own
    header (a COG IFD, a Parquet footer). Given that layout, :meth:`chunk_ref` yields a
    range ref so a read streams just those bytes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    key: str
    chunks: tuple[ChunkRef, ...]

    def chunk_ref(self, index: int) -> ChunkRef:
        """The range :class:`ChunkRef` for chunk *index*."""
        return self.chunks[index]

    def read_chunk(self, store: ObjectStore, index: int) -> bytes:
        """Stream the bytes of chunk *index* from *store*."""
        return read_chunk(store, self.chunks[index])
