# SPDX-License-Identifier: Apache-2.0
"""Error-budget reports as Parquet (RM-P1-SIM-03; sim.md §5).

The multi-fidelity scheduler's per-tier verdict — each surrogate/low-fidelity tier's declared or
tracked deviation vs. the high-fidelity reference, against the task tolerance — written as a
**Parquet** table [Bench](bench.md) ingests (sim.md §5 "error-budget reports"). One row per
:class:`~astro_mine.core.provenance.model.ErrorBudgetOutcome`; the columns are exactly its fields,
so the report round-trips back into the Core type for Bench.

pyarrow (the ``[report]`` extra) is imported lazily — the base wheel stays Parquet-free, and a run
that never emits a report needs neither the extra nor the dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_mine.core.provenance.model import ErrorBudgetOutcome

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

__all__ = ["ERROR_BUDGET_COLUMNS", "read_error_budget_report", "write_error_budget_report"]

#: The Parquet columns — exactly the :class:`ErrorBudgetOutcome` fields, in a fixed order so the
#: schema is stable for Bench (a new field is an additive, reviewed schema change).
ERROR_BUDGET_COLUMNS = ("name", "within_budget", "tier", "metric", "value", "tolerance")


def write_error_budget_report(outcomes: Sequence[ErrorBudgetOutcome], path: str | Path) -> None:
    """Write ``outcomes`` to a Parquet table at ``path`` (one row per outcome).

    Deterministic: rows are written in the given order and the column schema is fixed, so the same
    outcomes produce the same table. Requires the ``[report]`` extra (``pyarrow``)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Dict keys become the column names, in this fixed (ERROR_BUDGET_COLUMNS) order.
    columns = {
        "name": pa.array([o.name for o in outcomes], pa.string()),
        "within_budget": pa.array([o.within_budget for o in outcomes], pa.bool_()),
        "tier": pa.array([o.tier for o in outcomes], pa.string()),
        "metric": pa.array([o.metric for o in outcomes], pa.string()),
        "value": pa.array([o.value for o in outcomes], pa.float64()),
        "tolerance": pa.array([o.tolerance for o in outcomes], pa.float64()),
    }
    pq.write_table(pa.table(columns), str(path))


def read_error_budget_report(path: str | Path) -> list[ErrorBudgetOutcome]:
    """Read a Parquet error-budget report back into :class:`ErrorBudgetOutcome`s (for Bench)."""
    import pyarrow.parquet as pq

    table = pq.read_table(str(path))
    rows = table.to_pylist()
    return [ErrorBudgetOutcome(**row) for row in rows]
