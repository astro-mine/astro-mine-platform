# SPDX-License-Identifier: Apache-2.0
"""Warm-start seam for incremental online re-solve (RM-P1-ALLOC-05).

The delta-path stub for online replanning (allocate.md §2 principle 4; issue #5): keep the IR and
a previous incumbent warm so a re-solve after reality drifts starts from the last plan rather than
from scratch. :func:`hints_from` extracts a warm start — the assignment (``assign::{task}::{asset}``
= 1) and start-time (``start::{task}``) variable values, keyed by IR variable id — from a prior
:class:`~astro_mine.allocate.Allocation` (or its plan), which feeds the backend's existing
``hints`` seam. The exact solver **verifies and never trusts** a warm start (we never throw away a
hint the solver can verify, and never trust one it cannot; principle 4).

This is the documented stub: full incremental delta re-solve and Redis-warm online hardening for
[Ops](ops.md) are deferred to P1-late/P2.
"""

from __future__ import annotations

from astro_mine.allocate.api.model import Allocation, AssetSchedule
from astro_mine.allocate.model.ir.compile import assignment_var_id, start_var_id

__all__ = ["hints_from"]


def hints_from(source: Allocation | list[AssetSchedule]) -> dict[str, float]:
    """Extract a warm-start hint map (IR variable id → value) from a prior plan.

    Each scheduled task contributes its assignment variable (set to ``1``) and its start-time
    variable (its scheduled start), keyed by the RM-P1-ALLOC-01 IR id conventions — so the map
    drops straight into :meth:`AllocationPlanner.solve_anytime`'s ``hints`` seam. A missing/absent
    plan yields an empty map (nothing to warm-start from).
    """
    plan = source.plan if isinstance(source, Allocation) else source
    hints: dict[str, float] = {}
    for asset_schedule in plan or []:
        for st in asset_schedule.tasks:
            hints[assignment_var_id(st.task_id, asset_schedule.asset_id)] = 1.0
            hints[start_var_id(st.task_id)] = st.start_s
    return hints
