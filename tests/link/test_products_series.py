"""Latency/bandwidth time-series products for the planners (RM-P1-LINK-12).

Both representations from one plan: the Parquet contact-graph edge table (for combinatorial
Allocate/Mind) and the range-readable Zarr latency/bandwidth cube (for fidelity-sensitive
consumers and Sim). Verifies the round-trip, that a range read loads only the requested slice,
provenance is stamped, and the products are deterministic.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import zarr

from astro_mine.core.messages import ContactInterval, ContactNode, ContactPlan
from astro_mine.core.messages.enums import NodeRole
from astro_mine.core.units import Epoch, EpochWindow, TimeScale
from astro_mine.link.products import (
    LinkProductsError,
    contact_edge_table,
    emit_time_series,
    read_cube_pair,
    read_cube_time_window,
    write_contact_edge_table,
)
from astro_mine.link.products._series import _TIME_SCALE_KEY


def _window(start: float, end: float) -> EpochWindow:
    return EpochWindow(
        start=Epoch(tdb_seconds=start, scale=TimeScale.TDB),
        end=Epoch(tdb_seconds=end, scale=TimeScale.TDB),
    )


def _plan() -> ContactPlan:
    return ContactPlan(
        nodes=[
            ContactNode(id="S", role=NodeRole.SPACE),
            ContactNode(id="R", role=NodeRole.SPACE),
            ContactNode(id="G", role=NodeRole.GROUND),
        ],
        intervals=[
            ContactInterval(
                node_a="S",
                node_b="R",
                start_tdb_s=0.0,
                end_tdb_s=40.0,
                max_rate_bps=500.0,
                mean_latency_s=1.0,
            ),
            ContactInterval(
                node_a="R",
                node_b="G",
                start_tdb_s=20.0,
                end_tdb_s=60.0,
                max_rate_bps=800.0,
                mean_latency_s=2.0,
            ),
        ],
    )


def test_contact_edge_table_has_one_row_per_interval() -> None:
    table = contact_edge_table(_plan())
    assert table.num_rows == 2
    assert set(table.column_names) >= {
        "node_a",
        "node_b",
        "start_tdb_s",
        "max_rate_bps",
        "confidence",
    }


def test_write_edge_table_roundtrips_parquet(tmp_path: Path) -> None:
    path = write_contact_edge_table(_plan(), tmp_path / "edges.parquet")
    table = pq.read_table(path)
    assert table.num_rows == 2
    assert set(table.column("node_b").to_pylist()) == {"R", "G"}


def test_emit_time_series_writes_both_products(tmp_path: Path) -> None:
    manifest = emit_time_series(
        _plan(),
        _window(0.0, 100.0),
        10.0,
        tmp_path,
        provenance={"kernels": "meta.tm@abc", "dem": "lola@def"},
    )
    assert manifest.edge_table.exists() and manifest.cube.exists()
    assert manifest.n_time == 10
    assert manifest.pairs == ("G|R", "R|S")
    assert manifest.provenance["kernels"] == "meta.tm@abc"


def test_cube_pair_series_tracks_reachability_and_rate(tmp_path: Path) -> None:
    manifest = emit_time_series(_plan(), _window(0.0, 100.0), 10.0, tmp_path)
    sr = read_cube_pair(manifest.cube, "S", "R")
    assert sr.pairs == ("R|S",)
    reach = sr.reachable[:, 0]
    # S-R open [0,40) ⇒ reachable at t=0,10,20,30 (indices 0..3), closed afterwards.
    assert list(reach) == [True, True, True, True] + [False] * 6
    assert sr.rate_bps[0, 0] == 500.0 and np.isnan(sr.rate_bps[5, 0])


def test_range_read_loads_only_the_time_window(tmp_path: Path) -> None:
    manifest = emit_time_series(_plan(), _window(0.0, 100.0), 10.0, tmp_path)
    sl = read_cube_time_window(manifest.cube, _window(20.0, 50.0))
    assert list(sl.epoch_tdb_s) == [20.0, 30.0, 40.0]  # only the requested slice materialized
    assert sl.rate_bps.shape == (3, 2)
    # R-G (rate 800) is open across this slice; index of "G|R" is 0.
    assert sl.rate_bps[0, 0] == 800.0


def test_products_record_their_time_scale(tmp_path: Path) -> None:
    """Both serialized products record their TimeScale; a round-trip reads it back as TDB.

    RM-P1-LINK-14 / RFC-0007: nothing in ``latency_bandwidth.zarr`` recorded the scale before,
    and ``contact_edges.parquet`` carried its epoch columns purely by naming convention."""
    manifest = emit_time_series(_plan(), _window(0.0, 100.0), 10.0, tmp_path)
    assert manifest.scale is TimeScale.TDB

    # latency_bandwidth.zarr: the epoch axis is tagged with its scale alongside step_s.
    root = zarr.open_group(store=str(manifest.cube), mode="r")
    assert TimeScale(root.attrs[_TIME_SCALE_KEY]) is TimeScale.TDB

    # contact_edges.parquet: the scale lives in the Arrow schema metadata.
    metadata = pq.read_table(manifest.edge_table).schema.metadata
    assert TimeScale(metadata[_TIME_SCALE_KEY.encode()].decode()) is TimeScale.TDB


def test_edge_table_scale_prefers_typed_plan_window(tmp_path: Path) -> None:
    # A plan carrying a typed ``window`` is the authoritative scale source (RFC-0007).
    plan = _plan().model_copy(update={"window": _window(0.0, 100.0)})
    metadata = contact_edge_table(plan).schema.metadata
    assert TimeScale(metadata[_TIME_SCALE_KEY.encode()].decode()) is TimeScale.TDB


def test_emit_time_series_is_deterministic(tmp_path: Path) -> None:
    a = emit_time_series(_plan(), _window(0.0, 100.0), 10.0, tmp_path / "a")
    b = emit_time_series(_plan(), _window(0.0, 100.0), 10.0, tmp_path / "b")
    ra = read_cube_pair(a.cube, "R", "G")
    rb = read_cube_pair(b.cube, "R", "G")
    assert np.array_equal(ra.reachable, rb.reachable)
    assert np.array_equal(ra.rate_bps, rb.rate_bps, equal_nan=True)
    assert np.array_equal(ra.latency_s, rb.latency_s, equal_nan=True)


def test_non_positive_step_raises(tmp_path: Path) -> None:
    with pytest.raises(LinkProductsError, match="step_s must be positive"):
        emit_time_series(_plan(), _window(0.0, 100.0), 0.0, tmp_path)


def test_unknown_pair_read_raises(tmp_path: Path) -> None:
    manifest = emit_time_series(_plan(), _window(0.0, 100.0), 10.0, tmp_path)
    with pytest.raises(LinkProductsError, match="not in the cube"):
        read_cube_pair(manifest.cube, "S", "G")
