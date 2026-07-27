"""Backend-agnostic plan decoding shared by the solvers (RM-P1-ALLOC-02).

The small, **OR-Tools-free** helpers that map a solved ``(task → asset, task → start)`` assignment
back onto the Core-typed schedule and recompute the realized objective from the float IR terms.
Kept here so the CP-SAT driver and the no-dependency ``trivial-stub`` solver share one decoding —
and so the stub path imports nothing heavy (allocate.md §7: the local tier must always work).
"""

from __future__ import annotations

from collections.abc import Mapping

from astro_mine.allocate.api.model import AssetSchedule, ScheduledTask
from astro_mine.allocate.model.ir.model import AllocationIR
from astro_mine.core.messages.enums import TaskKind

__all__ = ["Pair", "build_plan", "realized_objective"]

#: A ``(task_id, asset_id)`` pair — the key for per-pair durations/costs.
Pair = tuple[str, str]


def realized_objective(ir: AllocationIR, assignment: Mapping[str, str]) -> float:
    """``sum(coefficient * variable_value)`` over the IR objective terms — the honest oracle value.

    Computed from the *float* IR terms exactly as :func:`~astro_mine.allocate.verify_feasible`
    re-derives it, so a backend's reported objective and the independent verifier never disagree.
    """
    var_by_id = {v.id: v for v in ir.variables}
    total = 0.0
    for term in ir.objective_terms:
        var = var_by_id[term.var_ref]
        placed = var.task_ref is not None and assignment.get(var.task_ref) == var.asset_ref
        total += term.coefficient * (1.0 if placed else 0.0)
    return total


def build_plan(
    assignment: Mapping[str, str],
    starts: Mapping[str, float],
    task_kinds: Mapping[str, TaskKind],
    durations: Mapping[Pair, float],
) -> list[AssetSchedule]:
    """Map a solved assignment + start times onto per-asset, time-ordered schedules.

    Each assigned task becomes a :class:`~astro_mine.allocate.ScheduledTask` on its asset's timeline
    with ``end_s = start_s + duration`` (a zero-length point when the duration is unknown); assets
    are emitted sorted and each asset's tasks are ordered by ``(start_s, task_id)`` so the plan is
    deterministic (allocate.md §8).
    """
    by_asset: dict[str, list[ScheduledTask]] = {}
    for task_id, asset_id in assignment.items():
        start = starts.get(task_id, 0.0)
        end = start + durations.get((task_id, asset_id), 0.0)
        by_asset.setdefault(asset_id, []).append(
            ScheduledTask(task_id=task_id, kind=task_kinds[task_id], start_s=start, end_s=end)
        )
    return [
        AssetSchedule(asset_id=asset_id, tasks=sorted(sts, key=lambda s: (s.start_s, s.task_id)))
        for asset_id, sts in sorted(by_asset.items())
    ]
