"""Latency/bandwidth time-series products for the planners (RM-P1-LINK-12).

link.md §11 recommends delivering **both** constraint representations to the planners, from
**one** plan:

- the **contact graph** as a **Parquet edge table** — the boolean/interval view combinatorial
  [Allocate](allocate.md)/[Mind](mind.md) schedule against (contact-window constraints,
  blackout intervals);
- the **continuous latency/bandwidth cube** as a **Zarr** array — dense per-tick ``rate_bps`` /
  ``latency_s`` for fidelity-sensitive consumers and Sim, **range-readable** so a consumer
  streams only the time-slice or pair it needs (link.md §5, §8) without materializing the whole
  cube, plus its interval table as Parquet.

Both are derived from a single Core :class:`~astro_mine.core.messages.ContactPlan` via the
:class:`~astro_mine.link.products.ConnectivitySampler`, and both carry provenance
(kernel/DEM/config hashes) so a comms-denied benchmark reproduces (link.md §5). Link emits the
Core message plus these serializations — it defines no new *message* types (conventions.md §1.1).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from astro_mine.core.messages import ContactPlan
from astro_mine.core.units import Epoch, EpochWindow, TimeScale
from astro_mine.link.products._errors import LinkProductsError
from astro_mine.link.products._sampler import ConnectivitySampler

__all__ = [
    "CubeManifest",
    "CubeSlice",
    "contact_edge_table",
    "emit_time_series",
    "read_cube_pair",
    "read_cube_time_window",
    "write_contact_edge_table",
]

_EDGE_TABLE = "contact_edges.parquet"
_CUBE = "latency_bandwidth.zarr"

#: The Arrow schema-metadata / Zarr-attr key under which each product records the time scale of
#: its ``*_tdb_s`` epoch columns/axis, so a consumer reads the scale from the artifact rather than
#: inferring it from a column name (RFC-0007 Design §2; conventions.md §5).
_TIME_SCALE_KEY = "time_scale"


def _plan_time_scale(plan: ContactPlan) -> TimeScale:
    """The plan's epoch scale: its typed ``window`` scale if populated, else :data:`TimeScale.TDB`.

    The ``*_tdb_s`` columns are TDB by their naming contract; a typed ``window`` (RFC-0007) is the
    authoritative source when a producer has populated it, and is preferred here (consumers MUST
    prefer the typed field)."""
    if plan.window is not None:
        return plan.window.start.scale
    for interval in plan.intervals:
        if interval.window is not None:
            return interval.window.start.scale
    return TimeScale.TDB


@dataclass(frozen=True)
class CubeManifest:
    """What :func:`emit_time_series` wrote: the Parquet edge table + the Zarr cube.

    ``pairs`` is the ordered list of ``"a|b"`` canonical link keys indexing the cube's second
    axis; ``epoch_tdb_s`` is its (uniform) time axis, tagged ``scale`` (its
    :class:`~astro_mine.core.units.TimeScale`, RFC-0007). ``provenance`` echoes the
    kernel/DEM/config hashes stamped into the cube attrs so the product is self-describing
    (link.md §5)."""

    edge_table: Path
    cube: Path
    pairs: tuple[str, ...]
    n_time: int
    provenance: Mapping[str, str]
    scale: TimeScale = TimeScale.TDB


@dataclass(frozen=True)
class CubeSlice:
    """A range-read slice of the cube — only the requested time-window/pair chunks are loaded."""

    epoch_tdb_s: np.ndarray
    pairs: tuple[str, ...]
    rate_bps: np.ndarray
    latency_s: np.ndarray
    reachable: np.ndarray


def _pair_key(node_a: str, node_b: str) -> str:
    return f"{node_a}|{node_b}" if node_a <= node_b else f"{node_b}|{node_a}"


def contact_edge_table(plan: ContactPlan) -> pa.Table:
    """The contact graph as a Parquet-ready edge table — one row per contact interval.

    Columns: ``node_a, node_b, start_tdb_s, end_tdb_s, max_rate_bps, min_latency_s,
    mean_latency_s, margin_db, band, modcod, confidence``. This is the tabular contact-window
    view Allocate builds interval/no-overlap constraints from and Mind reads blackout intervals
    from (link.md §6, §11). The ``start_tdb_s`` / ``end_tdb_s`` epoch columns stay bare ``float64``
    (the numeric kernel Arrow consumers scan), but the table's schema metadata records their
    :class:`~astro_mine.core.units.TimeScale` under :data:`_TIME_SCALE_KEY` so the scale is read
    from the artifact, not the column name (RFC-0007; conventions.md §5)."""
    rows = plan.intervals
    table = pa.table(
        {
            "node_a": pa.array([iv.node_a for iv in rows], pa.string()),
            "node_b": pa.array([iv.node_b for iv in rows], pa.string()),
            "start_tdb_s": pa.array([iv.start_tdb_s for iv in rows], pa.float64()),
            "end_tdb_s": pa.array([iv.end_tdb_s for iv in rows], pa.float64()),
            "max_rate_bps": pa.array([iv.max_rate_bps for iv in rows], pa.float64()),
            "min_latency_s": pa.array([iv.min_latency_s for iv in rows], pa.float64()),
            "mean_latency_s": pa.array([iv.mean_latency_s for iv in rows], pa.float64()),
            "margin_db": pa.array([iv.margin_db for iv in rows], pa.float64()),
            "band": pa.array([iv.band for iv in rows], pa.string()),
            "modcod": pa.array([iv.modcod for iv in rows], pa.string()),
            "confidence": pa.array([str(iv.confidence.value) for iv in rows], pa.string()),
        }
    )
    return table.replace_schema_metadata({_TIME_SCALE_KEY: _plan_time_scale(plan).value})


def write_contact_edge_table(plan: ContactPlan, path: str | Path) -> Path:
    """Write the contact-graph edge table to ``path`` as Parquet; returns the path."""
    out = Path(path)
    pq.write_table(contact_edge_table(plan), out)  # type: ignore[no-untyped-call]
    return out


def _grid(window: EpochWindow, step_s: float) -> np.ndarray:
    if step_s <= 0.0:
        raise LinkProductsError(f"step_s must be positive, got {step_s}")
    start, end = window.start.tdb_seconds, window.end.tdb_seconds
    n = int(np.ceil((end - start) / step_s))
    return start + np.arange(max(n, 1)) * step_s


def emit_time_series(
    plan: ContactPlan,
    window: EpochWindow,
    step_s: float,
    out_dir: str | Path,
    *,
    provenance: Mapping[str, str] | None = None,
) -> CubeManifest:
    """Emit **both** planner representations from ``plan`` into ``out_dir``.

    Writes ``contact_edges.parquet`` (the contact graph) and ``latency_bandwidth.zarr`` (the
    dense per-tick ``rate_bps``/``latency_s``/``reachable`` cube, shape ``(n_time, n_pairs)``,
    chunked along time so it is range-readable). Unreachable cells are ``NaN`` rate/latency and
    ``False`` reachable. ``provenance`` (kernel/DEM/config hashes) is stamped into the cube
    attrs. Deterministic: same plan + grid + provenance ⇒ identical arrays.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    edge_path = write_contact_edge_table(plan, out / _EDGE_TABLE)

    sampler = ConnectivitySampler(plan)
    grid = _grid(window, step_s)
    pairs = tuple(sorted(_pair_key(a, b) for (a, b) in sampler.connectivity(window.start)))
    n_time, n_pairs = len(grid), len(pairs)

    rate = np.full((n_time, n_pairs), np.nan, dtype=np.float64)
    latency = np.full((n_time, n_pairs), np.nan, dtype=np.float64)
    reach = np.zeros((n_time, n_pairs), dtype=bool)
    pair_nodes = [tuple(p.split("|", 1)) for p in pairs]
    for ti, t in enumerate(grid):
        epoch = Epoch(tdb_seconds=float(t), scale=TimeScale.TDB)
        for pi, (a, b) in enumerate(pair_nodes):
            state = sampler.reachable(a, b, epoch)
            if state.reachable:
                reach[ti, pi] = True
                if state.rate_bps is not None:
                    rate[ti, pi] = state.rate_bps
                if state.latency_s is not None:
                    latency[ti, pi] = state.latency_s

    cube_path = out / _CUBE
    root = zarr.open_group(store=str(cube_path), mode="w")
    time_chunk = min(n_time, 1024)
    # `create_array` + assignment, not `create_dataset(data=...)`: Zarr 3 *removed* the latter
    # outright (AttributeError, not a deprecation warning), and Link has to be on Zarr 3 because
    # Worlds writes the v3 field stores Link reads (#32).
    for name, values, chunks in (
        ("epoch_tdb_s", grid, (min(n_time, 65536),)),
        ("rate_bps", rate, (time_chunk, n_pairs)),
        ("latency_s", latency, (time_chunk, n_pairs)),
        ("reachable", reach, (time_chunk, n_pairs)),
    ):
        array = root.create_array(name, shape=values.shape, chunks=chunks, dtype=values.dtype.str)
        array[...] = values
    prov = dict(provenance or {})
    # The epoch axis carries its TimeScale explicitly (RFC-0007; conventions.md §5): the ``scale``
    # attr sits alongside ``step_s`` so a consumer cannot mistake what the ``epoch_tdb_s`` axis is.
    scale = window.start.scale
    root.attrs["pairs"] = list(pairs)
    root.attrs["provenance"] = json.dumps(prov, sort_keys=True)
    root.attrs["step_s"] = step_s
    root.attrs[_TIME_SCALE_KEY] = scale.value
    return CubeManifest(edge_path, cube_path, pairs, n_time, prov, scale)


def _cube_pairs(root: zarr.Group) -> list[str]:
    """The cube's ordered ``"a|b"`` pair keys, narrowed out of the JSON-typed group attrs.

    Zarr 3 types ``attrs`` values as a JSON union, so the stored list needs narrowing before it can
    be treated as a sequence of strings (#32)."""
    raw = root.attrs["pairs"]
    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise LinkProductsError(f"the cube's 'pairs' attr is not a list of keys, got {type(raw)}")
    return [str(pair) for pair in raw]


def _cube_array(root: zarr.Group, name: str) -> zarr.Array[Any]:
    """Narrow ``root[name]`` to an :class:`zarr.Array` — returned **lazily**, not materialized.

    Zarr 3 ships type hints, so ``root[name]`` is an ``Array | Group`` union that cannot be sliced
    until it is narrowed; Zarr 2 was untyped and mypy simply saw ``Any`` (#32). The narrowing is
    done once here rather than cast at every read site — and it must hand back the ``Array``, not
    its contents, because slicing the ``Array`` is what makes the cube **range-readable**: it pulls
    only the chunks the slice touches, which is the whole point of the Zarr representation
    (link.md §5, §8).
    """
    node = root[name]
    if not isinstance(node, zarr.Array):
        raise LinkProductsError(f"{name!r} is not an array in the cube at {root.store_path}")
    return node


def read_cube_time_window(path: str | Path, window: EpochWindow) -> CubeSlice:
    """Range-read only ``window``'s ``[start, end)`` slice of the cube (loads just those chunks).

    ``window`` is a typed :class:`~astro_mine.core.units.EpochWindow` (RFC-0007; conventions.md §5);
    its bounds are unwrapped to bare TDB seconds for the searchsorted numeric kernel here."""
    root = zarr.open_group(store=str(path), mode="r")
    epochs = np.asarray(_cube_array(root, "epoch_tdb_s")[:])
    lo = int(np.searchsorted(epochs, window.start.tdb_seconds, side="left"))
    hi = int(np.searchsorted(epochs, window.end.tdb_seconds, side="left"))
    return CubeSlice(
        epoch_tdb_s=epochs[lo:hi],
        pairs=tuple(_cube_pairs(root)),
        rate_bps=np.asarray(_cube_array(root, "rate_bps")[lo:hi, :]),
        latency_s=np.asarray(_cube_array(root, "latency_s")[lo:hi, :]),
        reachable=np.asarray(_cube_array(root, "reachable")[lo:hi, :]),
    )


def read_cube_pair(path: str | Path, node_a: str, node_b: str) -> CubeSlice:
    """Range-read only one link's column across all epochs (the pair's rate/latency series)."""
    root = zarr.open_group(store=str(path), mode="r")
    pairs = _cube_pairs(root)
    key = _pair_key(node_a, node_b)
    if key not in pairs:
        raise LinkProductsError(f"pair {key!r} is not in the cube; known pairs: {pairs}")
    col = pairs.index(key)
    return CubeSlice(
        epoch_tdb_s=np.asarray(_cube_array(root, "epoch_tdb_s")[:]),
        pairs=(key,),
        rate_bps=np.asarray(_cube_array(root, "rate_bps")[:, col : col + 1]),
        latency_s=np.asarray(_cube_array(root, "latency_s")[:, col : col + 1]),
        reachable=np.asarray(_cube_array(root, "reachable")[:, col : col + 1]),
    )
