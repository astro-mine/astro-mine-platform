"""Error-budget Parquet reports (RM-P1-SIM-03; sim.md §5).

The scheduler's per-tier error-budget verdicts are emitted as a Parquet table Bench ingests; the
report round-trips back into the Core ``ErrorBudgetOutcome`` type.
"""

from __future__ import annotations

from astro_mine.core.provenance.model import ErrorBudgetOutcome
from astro_mine.core.sadf.enums import DeterminismClass, FidelityTier
from astro_mine.core.sadf.model import FidelityProfile
from astro_mine.sim.recording.error_budget import (
    ERROR_BUDGET_COLUMNS,
    read_error_budget_report,
    write_error_budget_report,
)
from astro_mine.sim.scheduler import FidelityPolicy, select_fidelity


def _outcomes() -> list[ErrorBudgetOutcome]:
    ladder = (
        FidelityProfile(
            tier=FidelityTier.ARTICULATED, determinism_class=DeterminismClass.TOLERANCE
        ),
        FidelityProfile(tier=FidelityTier.SURROGATE, determinism_class=DeterminismClass.TOLERANCE),
    )
    selection = select_fidelity(
        "digger",
        ladder,
        FidelityPolicy(error_budget={"pos_x": 0.01, "vel_x": 0.5}),
        tier_budgets={FidelityTier.SURROGATE: {"pos_x": 0.002, "vel_x": 0.12}},
    )
    return selection.error_budget_outcomes()


def test_report_round_trips_through_parquet(tmp_path) -> None:
    outcomes = _outcomes()
    assert outcomes  # the budget selection produced per-channel rows
    path = tmp_path / "error_budget.parquet"
    write_error_budget_report(outcomes, path)
    assert path.exists()
    restored = read_error_budget_report(path)
    assert restored == outcomes


def test_report_columns_are_the_error_budget_outcome_fields(tmp_path) -> None:
    import pyarrow.parquet as pq

    path = tmp_path / "r.parquet"
    write_error_budget_report(_outcomes(), path)
    assert tuple(pq.read_table(str(path)).column_names) == ERROR_BUDGET_COLUMNS


def test_empty_report_is_valid(tmp_path) -> None:
    path = tmp_path / "empty.parquet"
    write_error_budget_report([], path)
    assert read_error_budget_report(path) == []
