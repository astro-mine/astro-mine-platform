# SPDX-License-Identifier: Apache-2.0
"""``build_comms_constraints`` — Link contact windows → comms-window gating (RM-P1-ALLOC-03).

Gates a relay-dependent task's execution to fall inside a [Link](link.md) contact window: a task
that must relay (e.g. a haul reporting back to a plant/orbiter) can only run when a contact interval
long enough to carry it exists between the executing asset's node and a relay/ground node. Where no
such window exists for an asset the ``(task, asset)`` pair is forbidden; where none exists for *any*
eligible asset the task is infeasible, with the explicit certificate the acceptance criteria name
("no contact window long enough to relay the haul"). Feasible pairs contribute a ``TIME_WINDOW``
constraint narrowing the task's start variable to the window envelope.

The contact truth arrives only through the Core
:class:`~astro_mine.core.messages.model.ContactPlan` in the
:class:`~astro_mine.allocate.ConstraintContext` — never a ``astro_mine.link`` import. Episode-time
task windows and Link's SPICE-TDB contact intervals are reconciled through the config's typed
``epoch0`` :class:`~astro_mine.core.units.Epoch` anchor (conventions.md §5: epochs are SPICE
TDB/ET — RFC-0007).
"""

from __future__ import annotations

from astro_mine.allocate.api.model import AllocationRequest, ConstraintContext, Task, TimeWindow
from astro_mine.allocate.constraints.config import CommsPolicy, ConstraintConfig
from astro_mine.allocate.constraints.result import ConstraintFinding, Pair, WindowResult
from astro_mine.allocate.enums import ConstraintKind, ConstraintSense
from astro_mine.allocate.model.ir.compile import start_var_id, window_envelope
from astro_mine.allocate.model.ir.model import AllocationIR, Constraint, ConstraintTerm
from astro_mine.allocate.model.ir.utils import assignment_pairs
from astro_mine.core.messages.enums import NodeRole
from astro_mine.core.messages.model import ContactInterval, ContactNode, ContactPlan

__all__ = ["build_comms_constraints", "comms_window_constraint_ids"]


def comms_window_constraint_ids(task_id: str) -> tuple[str, str]:
    """The ``(lower, upper)`` ids of a task's comms-window start-bound constraints."""
    return f"comms_lo::{task_id}", f"comms_hi::{task_id}"


def _relay_target_nodes(plan: ContactPlan) -> set[str]:
    """Node ids a relay-gated task may relay *to* — ground stations and relay orbiters.

    A node qualifies when its role is ``GROUND`` or its free ``kind`` label marks it a relay. When a
    plan classifies none (all bare ``SPACE`` peers), every node is treated as a valid relay
    target so the gate degrades to "any contact window" rather than forbidding everything.
    """
    targets = {n.id for n in plan.nodes if _is_relay_target(n)}
    return targets if targets else {n.id for n in plan.nodes}


def _is_relay_target(node: ContactNode) -> bool:
    if node.role is NodeRole.GROUND:
        return True
    return node.kind is not None and "relay" in node.kind.lower()


def _intervals_for(plan: ContactPlan, node_id: str, targets: set[str]) -> list[ContactInterval]:
    """Contact intervals connecting ``node_id`` to any relay target, in deterministic order."""
    out = [
        iv
        for iv in plan.intervals
        if (iv.node_a == node_id and iv.node_b in targets)
        or (iv.node_b == node_id and iv.node_a in targets)
    ]
    out.sort(key=lambda iv: (iv.start_tdb_s, iv.end_tdb_s, iv.node_a, iv.node_b))
    return out


def _feasible_window(
    task: Task,
    intervals: list[ContactInterval],
    duration_s: float,
    policy: CommsPolicy,
) -> tuple[float, float] | None:
    """The earliest feasible ``[start_lo, start_hi]`` (episode s) a task of ``duration_s`` can start
    within, given its time windows and the contact intervals — or ``None`` if none fits.

    Episode-time task windows are shifted into TDB (``+epoch0``) to intersect Link's TDB intervals,
    then the feasible start range (interval start .. interval end - duration, clamped to the task
    window) is shifted back to episode time. The union bounding envelope over all fits is returned
    (the exact per-window disjunction is a solver encoding, RM-P1-ALLOC-02).
    """
    task_windows = task.time_windows or [
        TimeWindow(start_s=window_envelope(task)[0], end_s=window_envelope(task)[1])
    ]
    # Extract the raw TDB seconds once: the numeric kernel stays float-based (the contact
    # intervals are TDB seconds too), while the contract surface carries the typed Epoch.
    epoch0_tdb_s = policy.epoch0.tdb_seconds
    lo: float | None = None
    hi: float | None = None
    for tw in task_windows:
        tw_lo_tdb = tw.start_s + epoch0_tdb_s
        tw_hi_tdb = tw.end_s + epoch0_tdb_s
        for iv in intervals:
            start_lo_tdb = max(tw_lo_tdb, iv.start_tdb_s)
            start_hi_tdb = min(tw_hi_tdb, iv.end_tdb_s) - duration_s
            if start_hi_tdb < start_lo_tdb:
                continue  # window too short to fit the task's duration
            start_lo = start_lo_tdb - epoch0_tdb_s
            start_hi = start_hi_tdb - epoch0_tdb_s
            lo = start_lo if lo is None else min(lo, start_lo)
            hi = start_hi if hi is None else max(hi, start_hi)
    if lo is None or hi is None:
        return None
    return lo, hi


def build_comms_constraints(
    request: AllocationRequest,
    base_ir: AllocationIR,
    context: ConstraintContext,
    *,
    config: ConstraintConfig,
    durations: dict[Pair, float],
    forbidden: frozenset[Pair],
) -> WindowResult:
    """Derive comms-window gating constraints for the relay-dependent tasks of a request."""
    policy = config.comms
    if policy is None:
        # No comms policy declared ⇒ no relay gating (Allocate presumes nothing).
        return WindowResult()
    pairs = assignment_pairs(base_ir)
    tasks = {t.task_id: t for t in request.tasks}

    new_forbidden: set[Pair] = set()
    findings: list[ConstraintFinding] = []
    degraded: set[str] = set()
    pair_windows: dict[Pair, tuple[float, float]] = {}
    window_constraints: list[Constraint] = []

    for task_id in sorted(pairs):
        task = tasks[task_id]
        if not policy.is_relay_gated(task_id, task.kind):
            continue

        # A relay-gated task with no contact data cannot be shown to close its link — degrade loudly
        # and forbid, never silently assume connectivity (link.md: "degrade loudly, never assume").
        if context.contacts is None:
            degraded.add("comms.no_contact_data")
            for asset_id in pairs[task_id]:
                if (task_id, asset_id) not in forbidden:
                    new_forbidden.add((task_id, asset_id))
                    findings.append(
                        ConstraintFinding(
                            code="comms.no_contact_data",
                            detail=f"{task_id} is relay-gated but no ContactPlan was provided",
                            task_id=task_id,
                            asset_id=asset_id,
                        )
                    )
            continue

        targets = _relay_target_nodes(context.contacts)
        task_lo: float | None = None
        task_hi: float | None = None
        any_feasible = False

        for asset_id in pairs[task_id]:
            if (task_id, asset_id) in forbidden:
                continue  # already kept out upstream (terrain) — no window to compute
            intervals = _intervals_for(context.contacts, policy.node_id(asset_id), targets)
            fit = _feasible_window(task, intervals, durations.get((task_id, asset_id), 0.0), policy)
            if fit is None:
                new_forbidden.add((task_id, asset_id))
                findings.append(
                    ConstraintFinding(
                        code="comms.no_window",
                        detail=(
                            f"no contact window long enough to relay {task_id} on {asset_id} "
                            "within its time window"
                        ),
                        task_id=task_id,
                        asset_id=asset_id,
                    )
                )
                continue
            any_feasible = True
            pair_windows[(task_id, asset_id)] = fit
            task_lo = fit[0] if task_lo is None else min(task_lo, fit[0])
            task_hi = fit[1] if task_hi is None else max(task_hi, fit[1])

        if not any_feasible:
            findings.append(
                ConstraintFinding(
                    code="comms.task_infeasible",
                    detail=f"no eligible asset has a contact window long enough to relay {task_id}",
                    task_id=task_id,
                )
            )
            continue

        # Narrow the (per-task) start variable to the union envelope of the feasible pair windows.
        assert task_lo is not None and task_hi is not None
        lo_id, hi_id = comms_window_constraint_ids(task_id)
        svid = start_var_id(task_id)
        window_constraints.append(
            Constraint(
                id=lo_id,
                kind=ConstraintKind.TIME_WINDOW,
                terms=[ConstraintTerm(var_ref=svid, coefficient=1.0)],
                sense=ConstraintSense.GE,
                rhs=task_lo,
            )
        )
        window_constraints.append(
            Constraint(
                id=hi_id,
                kind=ConstraintKind.TIME_WINDOW,
                terms=[ConstraintTerm(var_ref=svid, coefficient=1.0)],
                sense=ConstraintSense.LE,
                rhs=task_hi,
            )
        )

    return WindowResult(
        constraints=tuple(window_constraints),
        forbidden=frozenset(new_forbidden),
        findings=tuple(findings),
        degraded=tuple(sorted(degraded)),
        pair_windows=pair_windows,
    )
