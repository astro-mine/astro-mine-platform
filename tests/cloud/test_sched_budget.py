"""Per-tenant budget caps -- a runaway sweep is halted before it overspends."""

from __future__ import annotations

import pytest

from astro_mine.cloud.sched.budget import BudgetExceeded, BudgetLedger, CostRates, estimate_cost


def test_estimate_cost_spot_discount_and_mig_fraction() -> None:
    rates = CostRates(cpu_hour=0.1, gpu_hour=2.0, spot_discount=0.25)
    on_demand = estimate_cost(hours=2, cpus=4, gpus=1, spot=False, rates=rates)
    assert on_demand == pytest.approx(2 * (4 * 0.1 + 1 * 2.0))
    spot = estimate_cost(hours=2, cpus=4, gpus=1, spot=True, rates=rates)
    assert spot == pytest.approx(on_demand * 0.25)
    # a 1g MIG slice of a 7-way card is priced as 1/7 of a GPU-hour
    frac = estimate_cost(hours=1, gpus=1 / 7, rates=rates)
    assert frac == pytest.approx((1 / 7) * 2.0)


def test_estimate_cost_rejects_negatives() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        estimate_cost(hours=-1)


def test_runaway_sweep_is_halted_before_the_cap() -> None:
    ledger = BudgetLedger({"acme": 10.0})
    charged = 0
    with pytest.raises(BudgetExceeded, match="exceeded"):
        for _ in range(100):  # a runaway sweep of many jobs at 3.0 each
            ledger.charge("acme", 3.0)
            charged += 1
    # halted at 3 jobs (9.0 spent); the 4th (would be 12.0) is refused
    assert charged == 3
    assert ledger.spent("acme") == pytest.approx(9.0)
    assert ledger.remaining("acme") == pytest.approx(1.0)


def test_would_exceed_is_side_effect_free() -> None:
    ledger = BudgetLedger({"t": 5.0})
    assert ledger.would_exceed("t", 6.0) is True
    assert ledger.spent("t") == 0.0
    assert ledger.charge("t", 5.0) == pytest.approx(5.0)


def test_negative_charge_and_unknown_tenant_rejected() -> None:
    ledger = BudgetLedger({"t": 5.0})
    with pytest.raises(ValueError, match="non-negative"):
        ledger.charge("t", -1.0)
    with pytest.raises(KeyError, match="no budget cap"):
        ledger.charge("ghost", 1.0)
