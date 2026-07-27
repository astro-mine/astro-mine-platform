"""Bench-facing safety metrics + Parquet aggregation (RM-P1-GUARD-06).

Proves the two first-class scored metrics — "violations per scenario" (detect-layer trips, fallbacks
keyed by constraint) and "performance cost of shielding" (intervention rate + action divergence +
latency) — are computed from the verdict stream, and that the per-tick table round-trips through
Parquet for Bench to consume (guard.md §6, §8).
"""

from __future__ import annotations

import math
from pathlib import Path

from astro_mine.guard.audit import (
    ShieldingCost,
    ViolationCounts,
    shielding_cost,
    verdicts_to_table,
    violations_per_scenario,
    write_metrics_parquet,
)
from tests.guard.conftest import make_verdict


def _mixed_run() -> list[object]:
    return [
        make_verdict(
            reason="certified", intervention="none", action_divergence=0.0, shield_latency_us=2.0
        ),
        make_verdict(
            reason="shield_corrected",
            intervention="modified",
            action_divergence=4.0,
            shield_latency_us=6.0,
        ),
        make_verdict(
            reason="scalar_violated",
            intervention="fallback",
            constraint_ids=["c_anchor_torque"],
            action_divergence=9.0,
            shield_latency_us=10.0,
        ),
        make_verdict(
            reason="monitor_fired",
            intervention="fallback",
            constraint_ids=["c_night_soc_survival"],
            action_divergence=1.0,
            shield_latency_us=4.0,
        ),
    ]


def test_violations_per_scenario() -> None:
    counts = violations_per_scenario(_mixed_run())  # type: ignore[arg-type]
    assert counts.total == 2  # scalar_violated + monitor_fired
    assert counts.fallbacks == 2
    assert counts.by_constraint == {"c_anchor_torque": 1, "c_night_soc_survival": 1}


def test_violations_empty() -> None:
    assert violations_per_scenario([]) == ViolationCounts()


def test_shielding_cost() -> None:
    cost = shielding_cost(_mixed_run())  # type: ignore[arg-type]
    assert cost.ticks == 4
    assert cost.intervention_rate == 0.75  # 3 of 4 intervened
    assert cost.mean_action_divergence == (0.0 + 4.0 + 9.0 + 1.0) / 4
    assert cost.max_action_divergence == 9.0
    assert cost.max_latency_us == 10.0
    assert cost.mean_latency_us == (2.0 + 6.0 + 10.0 + 4.0) / 4


def test_shielding_cost_empty() -> None:
    assert shielding_cost([]) == ShieldingCost()


def test_verdicts_to_table_columns() -> None:
    table = verdicts_to_table(_mixed_run())  # type: ignore[arg-type]
    assert table.num_rows == 4
    assert "constraint_ids" in table.column_names
    assert "shield_latency_us" in table.column_names


def test_empty_table_keeps_schema() -> None:
    table = verdicts_to_table([])
    assert table.num_rows == 0
    assert "action_divergence" in table.column_names


def test_parquet_roundtrip_with_infinity(tmp_path: Path) -> None:
    import pyarrow.parquet as pq

    verdicts = [
        make_verdict(min_barrier_margin=math.inf, backup_kind="brake_to_stop"),
        make_verdict(min_barrier_margin=2.5, backup_kind=None),
    ]
    path = tmp_path / "metrics.parquet"
    write_metrics_parquet(verdicts, path)  # type: ignore[arg-type]
    table = pq.read_table(path)
    assert table.num_rows == 2
    margins = table.column("min_barrier_margin").to_pylist()
    assert math.isinf(margins[0]) and margins[1] == 2.5
    backups = table.column("backup_kind").to_pylist()
    assert backups == ["brake_to_stop", None]
