"""Scheduling -- Kueue queues/quotas/fair-share and hard per-tenant budgets.

Two rails keep a shared cluster fair and safe (``cloud.md`` §3, §9):

- :mod:`.kueue` compiles Kueue ``ResourceFlavor`` / ``ClusterQueue`` / ``LocalQueue`` objects
  and models **queue admission** -- work waits when a tenant is at quota, so one tenant
  cannot starve another (``cloud.md`` §2 principle 9, §9);
- :mod:`.budget` enforces **hard per-tenant budget caps**, halting a runaway sweep before it
  exceeds spend -- a combined cost and denial-of-service control (``cloud.md`` §5 principle,
  §9 "cost as a safety rail").

Backlog: RM-P1-CLOUD-03 -- https://github.com/astro-mine/astro-mine-cloud/issues/14
"""

from __future__ import annotations

from astro_mine.cloud.sched.budget import BudgetExceeded, BudgetLedger, CostRates, estimate_cost
from astro_mine.cloud.sched.kueue import (
    QueueAdmission,
    cluster_queue,
    local_queue,
    resource_flavor,
)

__all__ = [
    "BudgetExceeded",
    "BudgetLedger",
    "CostRates",
    "QueueAdmission",
    "cluster_queue",
    "estimate_cost",
    "local_queue",
    "resource_flavor",
]
