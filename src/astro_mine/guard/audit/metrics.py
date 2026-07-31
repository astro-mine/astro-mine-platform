"""Safety-metric extraction + Parquet aggregation from the verdict stream (RM-P1-GUARD-06).

The Bench-facing scoring surface: Guard **emits** the per-tick verdicts; this module derives the
two first-class metrics Bench scores a shielded run on (guard.md §6, §8; conventions.md §8):

- **"violations per scenario"** — :func:`violations_per_scenario`: the count of detect-layer trips
  (``reason`` ∈ :data:`VIOLATION_REASONS`) and fallbacks, keyed by the invoked spec clause
  (``constraint_ids``);
- **"performance cost of shielding"** — :func:`shielding_cost`: the intervention rate, the
  ``‖certified - proposed‖`` action divergence, and the per-tick shield latency.

The **paired with/without-shield return delta** — "did shielding cost performance?" — is *Bench's*
to compute (it owns the two rollouts); this module only emits the per-tick data Bench differences.

:func:`write_metrics_parquet` lands the per-tick table in **Parquet** (``pyarrow``) so Bench and
analysis read the columnar metrics; Bench also consumes the raw verdicts through its MCAP decoder
(``bench/recording``). ``pyarrow`` is an **optional extra**
(``astro-mine-platform[guard-metrics]``),
imported lazily — importing this module never imports ``pyarrow``.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from astro_mine.guard.audit.model import SafetyVerdict

if TYPE_CHECKING:
    import pyarrow as pa

__all__ = [
    "FALLBACK_INTERVENTION",
    "NO_INTERVENTION",
    "VIOLATION_REASONS",
    "ShieldingCost",
    "ViolationCounts",
    "shielding_cost",
    "verdicts_to_table",
    "violations_per_scenario",
    "write_metrics_parquet",
]

#: The audit reasons that mark a *detect-layer trip* — a hard constraint that was violated (a
#: scalar bound) or a temporal monitor that fired. These are the "violations" Bench counts.
VIOLATION_REASONS: frozenset[str] = frozenset({"scalar_violated", "monitor_fired"})
#: The intervention value marking a fall-back to the verified backup controller.
FALLBACK_INTERVENTION = "fallback"
#: The intervention value marking a pass-through (the proposed action was certified as-is).
NO_INTERVENTION = "none"


@dataclass(frozen=True, slots=True)
class ViolationCounts:
    """The "violations per scenario" metric (guard.md §6).

    ``total`` counts detect-layer trips (``reason`` ∈ :data:`VIOLATION_REASONS`); ``fallbacks``
    counts fall-backs to the backup controller; ``by_constraint`` counts the invoked spec clause
    ids across the trip ticks — so a scenario's violations trace to the exact constraint(s)."""

    total: int = 0
    fallbacks: int = 0
    by_constraint: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ShieldingCost:
    """The "performance cost of shielding" metric (guard.md §6, §8; charter §9).

    The per-tick shield overhead Bench reports so the "without neutering performance" claim is
    reproducible: how often Guard intervened (:attr:`intervention_rate`), how far it moved the
    action (``‖certified - proposed‖`` divergence), and how long the shield took."""

    ticks: int = 0
    intervention_rate: float = 0.0
    mean_action_divergence: float = 0.0
    max_action_divergence: float = 0.0
    mean_latency_us: float = 0.0
    max_latency_us: float = 0.0


def violations_per_scenario(verdicts: Iterable[SafetyVerdict]) -> ViolationCounts:
    """Count detect-layer trips + fallbacks, keyed by the invoked spec clause (guard.md §6)."""
    total = 0
    fallbacks = 0
    by_constraint: Counter[str] = Counter()
    for v in verdicts:
        if v.reason in VIOLATION_REASONS:
            total += 1
            by_constraint.update(v.constraint_ids)
        if v.intervention == FALLBACK_INTERVENTION:
            fallbacks += 1
    return ViolationCounts(total=total, fallbacks=fallbacks, by_constraint=dict(by_constraint))


def shielding_cost(verdicts: Iterable[SafetyVerdict]) -> ShieldingCost:
    """Aggregate the intervention rate, action divergence, and latency (guard.md §6, §8)."""
    rows = list(verdicts)
    ticks = len(rows)
    if ticks == 0:
        return ShieldingCost()
    interventions = sum(1 for v in rows if v.intervention != NO_INTERVENTION)
    divergences = [v.action_divergence for v in rows]
    latencies = [v.shield_latency_us for v in rows]
    return ShieldingCost(
        ticks=ticks,
        intervention_rate=interventions / ticks,
        mean_action_divergence=sum(divergences) / ticks,
        max_action_divergence=max(divergences),
        mean_latency_us=sum(latencies) / ticks,
        max_latency_us=max(latencies),
    )


def _table_schema() -> pa.Schema:
    import pyarrow as pa  # lazily: the [metrics] extra supplies pyarrow

    return pa.schema(
        [
            ("agent_id", pa.string()),
            ("tick", pa.int64()),
            ("sim_time_s", pa.float64()),
            ("layer", pa.string()),
            ("intervention", pa.string()),
            ("reason", pa.string()),
            ("backup_kind", pa.string()),
            ("constraint_ids", pa.list_(pa.string())),
            ("min_barrier_margin", pa.float64()),
            ("action_divergence", pa.float64()),
            ("shield_latency_us", pa.float64()),
            ("spec_content_hash", pa.string()),
            ("compiled_content_hash", pa.string()),
        ]
    )


def verdicts_to_table(verdicts: Iterable[SafetyVerdict]) -> pa.Table:
    """Project the per-tick verdicts into a typed ``pyarrow`` table (the Parquet row shape).

    An explicit schema keeps the column types stable even for an empty run. Requires the
    ``[metrics]`` extra (``pyarrow``), imported lazily."""
    import pyarrow as pa  # lazily: the [metrics] extra supplies pyarrow

    rows = list(verdicts)
    columns = {
        "agent_id": [v.agent_id for v in rows],
        "tick": [v.tick for v in rows],
        "sim_time_s": [v.sim_time_s for v in rows],
        "layer": [v.layer for v in rows],
        "intervention": [v.intervention for v in rows],
        "reason": [v.reason for v in rows],
        "backup_kind": [v.backup_kind for v in rows],
        "constraint_ids": [list(v.constraint_ids) for v in rows],
        "min_barrier_margin": [v.min_barrier_margin for v in rows],
        "action_divergence": [v.action_divergence for v in rows],
        "shield_latency_us": [v.shield_latency_us for v in rows],
        "spec_content_hash": [v.spec_content_hash for v in rows],
        "compiled_content_hash": [v.compiled_content_hash for v in rows],
    }
    return pa.table(columns, schema=_table_schema())


def write_metrics_parquet(verdicts: Iterable[SafetyVerdict], path: str | Path) -> None:
    """Write the aggregate per-tick safety-metric table to a Parquet file at ``path``.

    The columnar artifact Bench and analysis read the shielding-cost / violation metrics from
    (guard.md §5). Requires the ``[metrics]`` extra (``pyarrow``), imported lazily."""
    import pyarrow.parquet as pq  # lazily: the [metrics] extra supplies pyarrow

    pq.write_table(verdicts_to_table(verdicts), str(path))
