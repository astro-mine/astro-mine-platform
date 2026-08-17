# SPDX-License-Identifier: Apache-2.0
"""The chunked N-D field-layer store — Zarr (worlds.md §5; conventions.md §5).

worlds.md §3's module map files the "COG/Zarr writers" under ``ingest/``, but the Zarr store is
imported by ``terrain``/``illumination``/``spec`` — and ``ingest`` (the PDS/Diviner/LEND/M³ source
adapters) already imports ``spec`` to catalog what it ingests. Housing the store there would close
that loop into an import cycle, so it lives here as a **leaf**: it depends on nothing inside Worlds,
and everything that writes a field layer can depend on it. (``terrain/_ingest.py`` already sites the
COG writer with its product for the same reason.)

worlds.md §5's format table is normative about *which* store each product goes in::

    | Field models (illumination, thermal, gravity, regolith, dust) | **Zarr** (chunked,
      range-readable); **HDF5** for interop | The N-D physical fields consumed by Sim/Prospect |
    | Horizon maps / PSR masks | **Zarr** + **COG** | Precomputed per body region; PSR mask is
      epoch-window-derived |

COG stays the store for the *2-D rasters* it fits (source DEMs, the derived slope/aspect/
roughness terrain layers, the PSR mask) — this module is the other half: the **N-D field
layers**, written chunked so "a sim worker streams only the tiles/chunks it touches" and the
cloud precompute/serve tier (worlds.md §7) has an artifact to range-read from object storage
(RM-P0-WORLDS-01, RM-P0-WORLDS-03).

**Determinism.** The bundle's ``world_hash`` is a content hash, so a field store must be
byte-reproducible: the same array + the same pinned Zarr version write byte-identical chunk
files, and :func:`zarr_store_hash` digests the store *as it lands on disk* (sorted relative
names + bytes). Folding that digest into the world hash is what makes a **stored** horizon map
provenance-honest against a **recomputed** one — tampering with a chunk moves the world hash
even though the in-process array (and hence ``illumination_hash``) would not notice.

Each store keeps a **consolidated metadata index** (worlds.md §5 lifecycle: "field Zarr arrays
keep a consolidated metadata index"), so a reader resolves every array's shape/dtype/attrs from
one object rather than one HTTP request per array.
"""

from __future__ import annotations

import hashlib
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr
from numpy.typing import NDArray

__all__ = [
    "FIELD_STORE_SCHEMA",
    "ZARR_MEDIA_TYPE",
    "ZARR_STORE_SUFFIX",
    "FieldArray",
    "FieldStore",
    "default_chunks",
    "read_field_zarr",
    "write_field_zarr",
    "zarr_store_hash",
    "zarr_version",
]

#: The schema tag every Worlds Zarr field store carries in its group attributes.
FIELD_STORE_SCHEMA = "astro-mine-worlds/field/v0.1"

#: The STAC/OGC asset media type for a Zarr store — the catalog entry a range-reading consumer
#: (the worlds.md §7 field/tile service) resolves the chunks through.
ZARR_MEDIA_TYPE = "application/vnd+zarr"

#: The on-disk suffix of a field store directory (``illumination/horizon.zarr``).
ZARR_STORE_SUFFIX = ".zarr"

#: Chunk edge (cells) for the map-plane axes of a gridded field: a ~128x128 tile is the
#: "stream only the chunks you touch" unit of worlds.md §2 principle 5. The trailing
#: (non-map) axis — azimuth bins, diurnal phases — is kept whole so one cell's full profile
#: is a single chunk read.
_MAP_CHUNK = 128


def default_chunks(shape: Sequence[int]) -> tuple[int, ...]:
    """A range-readable chunking for a field array of ``shape``.

    3-D ``(height, width, k)`` fields (the horizon map) tile the map plane at
    :data:`_MAP_CHUNK` and keep the trailing profile axis whole; anything else is a single
    chunk (the small per-class curve stacks). Chunks never exceed the array itself.
    """
    dims = tuple(int(n) for n in shape)
    if len(dims) == 3:
        return (min(_MAP_CHUNK, dims[0]), min(_MAP_CHUNK, dims[1]), dims[2])
    return dims


@dataclass(frozen=True)
class FieldArray:
    """One named N-D array in a field store, with its units and dimension names.

    ``units`` is the explicit SI/units label conventions.md §5 requires every layer to carry;
    ``dims`` names each axis (e.g. ``("y", "x", "azimuth")``) so a consumer that never imports
    Worlds can still interpret the array.
    """

    name: str
    values: NDArray[Any]
    units: str
    dims: tuple[str, ...]
    chunks: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if len(self.dims) != self.values.ndim:
            raise ValueError(
                f"field {self.name!r}: {len(self.dims)} dim names for a {self.values.ndim}-D array"
            )


@dataclass(frozen=True)
class FieldStore:
    """A written Zarr field store: where it landed, and its content hash."""

    path: Path
    store_hash: str
    arrays: dict[str, tuple[int, ...]]  # name -> shape
    attrs: dict[str, Any]

    def to_manifest(self) -> dict[str, Any]:
        """The store's entry in the world manifest — self-describing without opening it."""
        return {
            "path": self.path.name,
            "media_type": ZARR_MEDIA_TYPE,
            "store_hash": self.store_hash,
            "arrays": {name: list(shape) for name, shape in sorted(self.arrays.items())},
        }


def zarr_version() -> str:
    """The pinned Zarr version — part of the toolchain a field store's bytes depend on."""
    return str(zarr.__version__)


def write_field_zarr(
    path: str | Path, arrays: Sequence[FieldArray], *, attrs: Mapping[str, Any]
) -> FieldStore:
    """Write ``arrays`` into a chunked Zarr group at ``path`` and hash the result.

    The group carries ``attrs`` (plus the :data:`FIELD_STORE_SCHEMA` tag); each array carries its
    ``units`` and ``dims``. A consolidated metadata index is written last (worlds.md §5), then the
    store is digested with :func:`zarr_store_hash` so the caller can fold it into a world hash.
    """
    store_path = Path(path)
    if not arrays:
        raise ValueError("a field store needs at least one array")
    group = zarr.open_group(store=str(store_path), mode="w")
    group.attrs.update({"schema": FIELD_STORE_SCHEMA, **dict(attrs)})
    shapes: dict[str, tuple[int, ...]] = {}
    for field in arrays:
        values = np.ascontiguousarray(field.values)
        chunks = field.chunks or default_chunks(values.shape)
        array = group.create_array(
            field.name, shape=values.shape, chunks=chunks, dtype=values.dtype
        )
        array[...] = values
        array.attrs.update({"units": field.units, "dims": list(field.dims)})
        shapes[field.name] = tuple(int(n) for n in values.shape)
    # Consolidated metadata is not (yet) in the Zarr v3 spec, so zarr warns; worlds.md §5
    # nonetheless requires the index, and the store stays readable without it. Silence the
    # advisory rather than let it surface as a warning in every consumer's logs.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        zarr.consolidate_metadata(group.store)
    return FieldStore(
        path=store_path,
        store_hash=zarr_store_hash(store_path),
        arrays=shapes,
        attrs=dict(group.attrs),
    )


def read_field_zarr(path: str | Path) -> tuple[dict[str, NDArray[Any]], dict[str, Any]]:
    """Read every array (and the group attributes) back out of a Zarr field store.

    Returns ``({name: array}, group_attrs)``. Fails loudly on a store that is not one of ours —
    a missing/foreign :data:`FIELD_STORE_SCHEMA` tag means the bytes are not what the caller
    thinks they are, and a silently-wrong field layer is exactly the class of bug conventions.md
    §5 makes a hard failure.
    """
    store_path = Path(path)
    group = zarr.open_group(store=str(store_path), mode="r")
    attrs = dict(group.attrs)
    schema = attrs.get("schema")
    if schema != FIELD_STORE_SCHEMA:
        raise ValueError(
            f"{store_path} is not an Astro-Mine field store "
            f"(schema {schema!r}, expected {FIELD_STORE_SCHEMA!r})"
        )
    arrays = {str(name): np.asarray(array[...]) for name, array in sorted(group.arrays())}
    return arrays, attrs


def zarr_store_hash(path: str | Path) -> str:
    """A deterministic ``sha256:`` digest over a Zarr store's files, as they land on disk.

    Sorted POSIX-relative member names + their bytes — the same walk
    :func:`~astro_mine.worlds.spec._publish.deterministic_bundle_tar` packs with, so the digest
    pins exactly the bytes that get published. Content-addressing the *store* (not just the
    in-memory array) is what makes a persisted field layer provenance-honest in ``world_hash``.
    """
    root = Path(path)
    digest = hashlib.sha256()
    for member in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(member.relative_to(root).as_posix().encode("utf-8"))
        digest.update(member.read_bytes())
    return f"sha256:{digest.hexdigest()}"
