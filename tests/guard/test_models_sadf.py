"""SadfBudgets — extracting Fleet SADF budgets as SI scalars (RM-P1-GUARD-04)."""

from __future__ import annotations

from astro_mine.guard.models import SadfBudgets
from tests.guard.models_fixtures import sadf_document


def test_extracts_every_budget() -> None:
    b = SadfBudgets.from_document(sadf_document())
    assert b.power_floor_w == 15.0
    assert b.battery_capacity_j == 1_200_000.0
    assert b.thermal_operating_max_k == 320.0
    assert b.thermal_operating_min_k == 120.0
    assert b.thermal_survival_max_k == 350.0
    assert b.thermal_survival_min_k == 100.0
    assert b.max_actuator_torque_nm == 42.0
    # The most conservative (smallest) contact slope limit.
    assert b.max_slope_deg == 22.0


def test_energy_floor_is_a_fraction_of_capacity() -> None:
    b = SadfBudgets.from_document(sadf_document(capacity_j=1_000_000.0))
    assert b.energy_floor_j(0.15) == 150_000.0


def test_missing_subsystems_are_none_safe() -> None:
    b = SadfBudgets.from_document(
        sadf_document(
            floor_w=None,
            capacity_j=None,
            operating_range_k=None,
            survival_range_k=None,
            torque_nm=None,
            max_slope_deg=None,
        )
    )
    assert b.power_floor_w is None
    assert b.battery_capacity_j is None
    assert b.thermal_operating_max_k is None
    assert b.thermal_survival_min_k is None
    assert b.max_actuator_torque_nm is None
    assert b.max_slope_deg is None
    assert b.energy_floor_j(0.15) is None  # no capacity ⇒ no floor


def test_operating_range_without_survival_range() -> None:
    b = SadfBudgets.from_document(sadf_document(survival_range_k=None))
    assert b.thermal_operating_max_k == 320.0
    assert b.thermal_survival_min_k is None
    assert b.thermal_survival_max_k is None


def test_battery_capacity_sums_multiple_storages() -> None:
    doc = sadf_document(capacity_j=500_000.0)
    # Append a second storage bank.
    from astro_mine.core.sadf.model import PowerStorage

    assert doc.asset.power is not None
    doc.asset.power.storage.append(PowerStorage(name="bat2", capacity_j=250_000.0))
    b = SadfBudgets.from_document(doc)
    assert b.battery_capacity_j == 750_000.0
