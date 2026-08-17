# SPDX-License-Identifier: Apache-2.0
"""Hard per-tenant budget caps -- cost as a safety rail.

A :class:`BudgetLedger` gives each tenant a spend cap and **halts a runaway sweep before it
exceeds it** (``cloud.md`` §5, §9 "cost as a safety rail"): compute that runs longer than it
should is treated as a defect, not a fact of life (``cloud.md`` §2 principle 5).
:func:`estimate_cost` is the node-hours x price model (spot vs on-demand, GPU-MIG fractions) --
it feeds both
the cap and the per-tenant cost dashboards (``cloud.md`` §10).

Backlog: RM-P1-CLOUD-03 -- astro-mine-cloud#14
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["BudgetExceeded", "BudgetLedger", "CostRates", "estimate_cost"]


class BudgetExceeded(RuntimeError):
    """Raised when a charge would push a tenant past its hard budget cap."""


@dataclass(frozen=True)
class CostRates:
    """Per-hour prices for the cost model (illustrative defaults; a deployment sets its own)."""

    cpu_hour: float = 0.05
    gpu_hour: float = 2.50
    #: Spot/preemptible discount multiplier applied to the on-demand rate (``cloud.md`` §8).
    spot_discount: float = 0.3


def estimate_cost(
    *,
    hours: float,
    cpus: float = 0.0,
    gpus: float = 0.0,
    spot: bool = False,
    rates: CostRates | None = None,
) -> float:
    """Estimate a job's cost: (cpu + gpu) node-hours x price, discounted on spot.

    A fractional ``gpus`` (e.g. ``1/7`` for a ``1g`` MIG slice of a 7-way card) prices a
    shared card fairly, so MIG sharing shows up in cost attribution (``cloud.md`` §10).
    """
    if hours < 0 or cpus < 0 or gpus < 0:
        raise ValueError("hours, cpus, and gpus must be non-negative")
    rate = rates if rates is not None else CostRates()
    raw = hours * (cpus * rate.cpu_hour + gpus * rate.gpu_hour)
    return raw * rate.spot_discount if spot else raw


class BudgetLedger:
    """Tracks per-tenant spend against hard caps and refuses to overspend."""

    def __init__(self, caps: Mapping[str, float]) -> None:
        self._caps = dict(caps)
        self._spent: dict[str, float] = dict.fromkeys(caps, 0.0)

    def _cap(self, tenant: str) -> float:
        try:
            return self._caps[tenant]
        except KeyError:
            raise KeyError(f"no budget cap for tenant {tenant!r}") from None

    def would_exceed(self, tenant: str, amount: float) -> bool:
        """Whether charging *amount* would push *tenant* past its cap (no state change)."""
        return self._spent.get(tenant, 0.0) + amount > self._cap(tenant)

    def charge(self, tenant: str, amount: float) -> float:
        """Charge *amount* to *tenant*; raise :class:`BudgetExceeded` if it would overspend.

        This is the enforcement rail: a sweep that charges per job halts on the first job
        that would breach the cap, so a runaway never exhausts shared spend.
        """
        if amount < 0:
            raise ValueError("charge amount must be non-negative")
        if self.would_exceed(tenant, amount):
            raise BudgetExceeded(
                f"tenant {tenant!r} budget {self._cap(tenant)} exceeded: "
                f"spent {self._spent[tenant]} + {amount}"
            )
        self._spent[tenant] += amount
        return self._spent[tenant]

    def spent(self, tenant: str) -> float:
        """Total charged to *tenant* so far."""
        return self._spent[tenant]

    def remaining(self, tenant: str) -> float:
        """Budget left for *tenant*."""
        return self._cap(tenant) - self._spent[tenant]
