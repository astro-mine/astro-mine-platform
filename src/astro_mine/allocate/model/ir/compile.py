"""``compile_request`` — lower an ``AllocationRequest`` to the solver-neutral IR (RM-P1-ALLOC-01).

The constraint-model compiler (allocate.md §3): it lifts tasks, assets, time windows,
precedence, and value into the canonical :class:`~astro_mine.allocate.AllocationIR` that any
backend lowers. It emits the **structural skeleton**:

- one ``BINARY`` **assignment** variable per (task, *eligible* asset) pair — an asset is
  eligible when it declares every capability a task requires (allocate.md §6);
- one ``CONTINUOUS`` **start-time** variable per task, bounded by its time-window envelope;
- one ``BINARY`` **window-select** variable per window of a task with **plural** windows, plus the
  ``TIME_WINDOW`` constraints that make those windows an exact **disjunction** (RM-P1-ALLOC-02);
- an ``ASSIGNMENT_COVER`` constraint per task (exactly one asset does it);
- ``TIME_WINDOW`` constraints binding a task's start to its window(s);
- ``PRECEDENCE`` constraints (a task starts no earlier than each predecessor);
- a per-asset ``NO_OVERLAP`` constraint over the intervals of the tasks that asset may take
  (RM-P1-ALLOC-02) — one asset does one task at a time;
- one objective term per assignment variable, weighted by the task's value ``mean``.

The **physics** constraint builders — power/energy, comms-window, terrain traversability — live in
:mod:`astro_mine.allocate.constraints` (RM-P1-ALLOC-03); they consume the
:class:`~astro_mine.allocate.ConstraintContext` handles and add constraint families to *this same*
IR, no new solver paradigm. The constrained compile also **re-emits** the scheduling constraints
from the *resolved per-pair* durations (a tracked excavator and a wheeled rover cross the same
slope differently), superseding the skeleton's declared nominal ``Task.duration_s``.

**Time windows are a disjunction, not an envelope.** A task's plural ``time_windows`` are genuinely
disjoint alternatives, so collapsing them to ``[min(start), max(end)]`` would silently admit a start
inside an availability **gap**. The start variable's *bounds* stay the envelope (a bound is only
ever a relaxation), and the exactness is carried by the constraints: ``twin_pick`` selects exactly
one window and ``twin_lo``/``twin_hi`` tie the start to the *selected* window. Because exactly one
selector is ``1``, ``start - sum(lo_k * y_k) >= 0`` and ``start - sum(hi_k * y_k) <= 0`` are an
exact — and purely **linear** — encoding of the disjunction, so a MILP or auction backend lowers it
with no more machinery than CP-SAT does (allocate.md §3: model/solver separation).

Every collection is emitted in **stable sorted order** (by ``id``); the compiler never
relies on dict/list iteration order, so a fixed request compiles to a byte-identical IR
(the determinism prerequisite for the golden-plan gate, RM-P1-ALLOC-07).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from astro_mine.allocate.api.model import AllocationRequest, Task, TimeWindow
from astro_mine.allocate.enums import (
    ConstraintKind,
    ConstraintSense,
    VariableKind,
    VariableSemantic,
)
from astro_mine.allocate.model.ir.model import (
    IR_VERSION,
    AllocationIR,
    Constraint,
    ConstraintTerm,
    DecisionVariable,
    ObjectiveTerm,
)

__all__ = [
    "Pair",
    "assignment_var_id",
    "compile_request",
    "cumulative_constraint_id",
    "earliest_start_in_windows",
    "no_overlap_constraint_id",
    "no_overlap_constraints",
    "start_var_id",
    "task_windows",
    "window_envelope",
    "window_select_var_id",
]

#: Finite upper bound for an unwindowed task's start time (SI seconds). A concrete large
#: value keeps the IR fully bounded and byte-stable rather than carrying an open interval.
_UNBOUNDED_HORIZON_S = 1.0e12

#: Absolute floor for the compiler's float comparisons (matches the IR verifier's epsilon).
_EPS = 1.0e-9

#: A ``(task_id, asset_id)`` pair — the key of the per-pair interval sizes a scheduling constraint
#: is emitted from (mirrors :data:`astro_mine.allocate.solvers._common.Pair`).
Pair = tuple[str, str]


def assignment_var_id(task_id: str, asset_id: str) -> str:
    """The stable id of the "``asset_id`` does ``task_id``" binary assignment variable."""
    return f"assign::{task_id}::{asset_id}"


def start_var_id(task_id: str) -> str:
    """The stable id of ``task_id``'s continuous start-time variable."""
    return f"start::{task_id}"


def window_select_var_id(task_id: str, index: int) -> str:
    """The stable id of the "``task_id`` runs in its ``index``-th window" binary selector.

    The index is zero-padded so the IR's lexicographic ``id`` sort (allocate.md §8) coincides with
    the windows' chronological order — the order :func:`task_windows` normalizes them into.
    """
    return f"window::{task_id}::{index:03d}"


def no_overlap_constraint_id(resource_id: str) -> str:
    """The id of the ``NO_OVERLAP`` constraint over one single-capacity resource (an asset)."""
    return f"no_overlap::{resource_id}"


def cumulative_constraint_id(resource_id: str) -> str:
    """The id of the ``CUMULATIVE`` constraint over one capacity-bearing resource."""
    return f"cumulative::{resource_id}"


def task_windows(task: Task) -> list[TimeWindow]:
    """A task's availability windows in canonical (chronological) order.

    The disjunction the compiler encodes, normalized once so the emitted IR is byte-stable: sorted
    by ``(start_s, end_s)``. An unwindowed task returns ``[]`` — it is available across the whole
    horizon and carries no window constraint at all.
    """
    return sorted(task.time_windows, key=lambda w: (w.start_s, w.end_s))


def window_envelope(task: Task) -> tuple[float, float]:
    """The ``[earliest, latest]`` **bounding envelope** of a task's time windows.

    The bound of the task's start variable — ``min`` start to ``max`` end over every window, or
    ``[0, horizon]`` when the task is unwindowed. A variable bound is only ever a *relaxation*: for
    a task with plural disjoint windows this envelope spans the **gaps** between them, and the
    exactness is carried by the disjunction constraints :func:`compile_request` emits alongside it
    (``twin_pick`` + ``twin_lo``/``twin_hi`` over the ``WINDOW_SELECT`` variables). Use
    :func:`earliest_start_in_windows` — never this envelope — to *place* a start.
    """
    windows = task_windows(task)
    if not windows:
        return 0.0, _UNBOUNDED_HORIZON_S
    return min(w.start_s for w in windows), max(w.end_s for w in windows)


def earliest_start_in_windows(task: Task, not_before: float) -> float | None:
    """The earliest start at or after ``not_before`` inside one of ``task``'s windows.

    The placement primitive every greedy path uses (the planner's constraint-aware greedy and the
    ``trivial-stub`` backend): it *snaps a start forward out of an availability gap* into the next
    window that can still host it, and returns ``None`` when no window can (the task is infeasible
    at or after ``not_before``). An unwindowed task can start immediately.
    """
    windows = task_windows(task)
    if not windows:
        return not_before
    for window in windows:
        candidate = max(not_before, window.start_s)
        if candidate <= window.end_s + _EPS:
            return candidate
    return None


def no_overlap_constraints(
    pairs: Mapping[str, Sequence[str]],
    sizes: Mapping[Pair, float],
    *,
    forbidden: frozenset[Pair] = frozenset(),
) -> list[Constraint]:
    """One ``NO_OVERLAP`` constraint per asset that could be double-booked (RM-P1-ALLOC-02).

    ``pairs`` maps each task to the assets eligible for it (the assignment variables the compiler
    emitted) and ``sizes`` gives each ``(task, asset)`` pair's interval length in SI seconds; a
    ``forbidden`` pair (a terrain/comms keep-out) contributes no interval. Each emitted constraint
    names its resource in the id (``no_overlap::{asset}``) and carries one term per candidate task:
    the task's **start variable**, with its **interval size** as the coefficient. A backend channels
    each interval to the matching ``assign::{task}::{asset}`` variable, so an interval is present
    only when that asset actually takes that task
    (:func:`~astro_mine.allocate.model.compile.cpsat.lower_to_cpsat`), and
    :func:`~astro_mine.allocate.verify_feasible` re-checks it against the *same* sizes.

    A resource with fewer than two candidate tasks cannot be double-booked, and one whose candidate
    tasks are all zero-length points occupies no time — neither emits a constraint, so a request
    that declares no durations compiles to exactly the pre-scheduling IR.
    """
    tasks_by_asset: dict[str, list[str]] = {}
    for task_id in sorted(pairs):
        for asset_id in pairs[task_id]:
            if (task_id, asset_id) in forbidden:
                continue
            tasks_by_asset.setdefault(asset_id, []).append(task_id)

    constraints: list[Constraint] = []
    for asset_id in sorted(tasks_by_asset):
        candidates = sorted(tasks_by_asset[asset_id])
        if len(candidates) < 2:
            continue  # a single-task resource cannot be double-booked
        terms = [
            ConstraintTerm(
                var_ref=start_var_id(task_id), coefficient=sizes.get((task_id, asset_id), 0.0)
            )
            for task_id in candidates
        ]
        if all(term.coefficient <= _EPS for term in terms):
            continue  # zero-length points occupy no time — nothing to keep apart
        constraints.append(
            Constraint(
                id=no_overlap_constraint_id(asset_id),
                kind=ConstraintKind.NO_OVERLAP,
                terms=terms,
                # A scheduling family carries no linear right-hand side; the sense/rhs slots are
                # pinned to a canonical `<= 0` so the IR stays one structural shape.
                sense=ConstraintSense.LE,
                rhs=0.0,
            )
        )
    return constraints


def _eligible_asset_ids(task: Task, asset_tags: dict[str, set[str]]) -> list[str]:
    required = {str(t) for t in task.required_capabilities}
    return sorted(aid for aid, tags in asset_tags.items() if required <= tags)


def _window_constraints(
    task: Task, windows: list[TimeWindow]
) -> tuple[list[DecisionVariable], list[Constraint]]:
    """One task's window variables + constraints — a bound for a single window, a disjunction for
    several.

    A single window needs no selector: the start is simply bounded by it (``twin_lo``/``twin_hi``
    carrying the window's own bounds), which is exactly the RM-P1-ALLOC-01 encoding. Plural windows
    get one ``WINDOW_SELECT`` variable each, a ``twin_pick`` exactly-one over them, and the *same*
    ``twin_lo``/``twin_hi`` ids re-expressed against the selected window — so every consumer's
    id-prefix convention (binding-constraint sources, IIS task attribution) is unchanged.
    """
    if not windows:
        return [], []

    svid = start_var_id(task.task_id)
    if len(windows) == 1:
        window = windows[0]
        return [], [
            Constraint(
                id=f"twin_lo::{task.task_id}",
                kind=ConstraintKind.TIME_WINDOW,
                terms=[ConstraintTerm(var_ref=svid, coefficient=1.0)],
                sense=ConstraintSense.GE,
                rhs=window.start_s,
            ),
            Constraint(
                id=f"twin_hi::{task.task_id}",
                kind=ConstraintKind.TIME_WINDOW,
                terms=[ConstraintTerm(var_ref=svid, coefficient=1.0)],
                sense=ConstraintSense.LE,
                rhs=window.end_s,
            ),
        ]

    selectors = [
        DecisionVariable(
            id=window_select_var_id(task.task_id, index),
            kind=VariableKind.BINARY,
            lower=0.0,
            upper=1.0,
            semantic=VariableSemantic.WINDOW_SELECT,
            task_ref=task.task_id,
        )
        for index in range(len(windows))
    ]
    pairs = list(zip(selectors, windows, strict=True))
    constraints = [
        # Exactly one of the task's disjoint windows hosts it.
        Constraint(
            id=f"twin_pick::{task.task_id}",
            kind=ConstraintKind.TIME_WINDOW,
            terms=[ConstraintTerm(var_ref=v.id, coefficient=1.0) for v in selectors],
            sense=ConstraintSense.EQ,
            rhs=1.0,
        ),
        # start - sum(lo_k * y_k) >= 0 : no earlier than the *selected* window's start.
        Constraint(
            id=f"twin_lo::{task.task_id}",
            kind=ConstraintKind.TIME_WINDOW,
            terms=[
                ConstraintTerm(var_ref=svid, coefficient=1.0),
                *(ConstraintTerm(var_ref=v.id, coefficient=-w.start_s) for v, w in pairs),
            ],
            sense=ConstraintSense.GE,
            rhs=0.0,
        ),
        # start - sum(hi_k * y_k) <= 0 : and no later than the *selected* window's end.
        Constraint(
            id=f"twin_hi::{task.task_id}",
            kind=ConstraintKind.TIME_WINDOW,
            terms=[
                ConstraintTerm(var_ref=svid, coefficient=1.0),
                *(ConstraintTerm(var_ref=v.id, coefficient=-w.end_s) for v, w in pairs),
            ],
            sense=ConstraintSense.LE,
            rhs=0.0,
        ),
    ]
    return selectors, constraints


def compile_request(request: AllocationRequest) -> AllocationIR:
    """Compile an :class:`AllocationRequest` into a byte-stable :class:`AllocationIR`."""
    asset_tags = {a.asset_id: {str(t) for t in a.capability_tags} for a in request.assets}
    tasks = sorted(request.tasks, key=lambda t: t.task_id)

    variables: list[DecisionVariable] = []
    constraints: list[Constraint] = []
    objective_terms: list[ObjectiveTerm] = []
    eligible_by_task: dict[str, list[str]] = {}
    sizes: dict[Pair, float] = {}

    for task in tasks:
        eligible = _eligible_asset_ids(task, asset_tags)
        eligible_by_task[task.task_id] = eligible
        cover_terms: list[ConstraintTerm] = []
        for asset_id in eligible:
            vid = assignment_var_id(task.task_id, asset_id)
            sizes[(task.task_id, asset_id)] = task.duration_s
            variables.append(
                DecisionVariable(
                    id=vid,
                    kind=VariableKind.BINARY,
                    lower=0.0,
                    upper=1.0,
                    semantic=VariableSemantic.ASSIGNMENT,
                    task_ref=task.task_id,
                    asset_ref=asset_id,
                )
            )
            cover_terms.append(ConstraintTerm(var_ref=vid, coefficient=1.0))
            objective_terms.append(
                ObjectiveTerm(id=f"obj::{vid}", var_ref=vid, coefficient=task.value.mean)
            )

        # Exactly one eligible asset does the task. When no asset is eligible the cover has
        # no terms (0 != 1) and is unsatisfiable — the request is infeasible, which the
        # verifier and the solver both surface honestly.
        constraints.append(
            Constraint(
                id=f"cover::{task.task_id}",
                kind=ConstraintKind.ASSIGNMENT_COVER,
                terms=cover_terms,
                sense=ConstraintSense.EQ,
                rhs=1.0,
            )
        )

        earliest, latest = window_envelope(task)
        variables.append(
            DecisionVariable(
                id=start_var_id(task.task_id),
                kind=VariableKind.CONTINUOUS,
                lower=earliest,
                upper=latest,
                semantic=VariableSemantic.START_TIME,
                task_ref=task.task_id,
            )
        )
        selectors, window_constraints = _window_constraints(task, task_windows(task))
        variables.extend(selectors)
        constraints.extend(window_constraints)

    for task in tasks:
        for pred in sorted(task.precedence):
            # start[task] - start[pred] >= 0 : the task starts no earlier than its predecessor.
            constraints.append(
                Constraint(
                    id=f"prec::{pred}->{task.task_id}",
                    kind=ConstraintKind.PRECEDENCE,
                    terms=[
                        ConstraintTerm(var_ref=start_var_id(task.task_id), coefficient=1.0),
                        ConstraintTerm(var_ref=start_var_id(pred), coefficient=-1.0),
                    ],
                    sense=ConstraintSense.GE,
                    rhs=0.0,
                )
            )

    # One asset does one task at a time, over the tasks' declared nominal durations. The constrained
    # compile re-emits these from the resolved per-pair durations (constraints/compose.py).
    constraints.extend(no_overlap_constraints(eligible_by_task, sizes))

    variables.sort(key=lambda v: v.id)
    constraints.sort(key=lambda c: c.id)
    objective_terms.sort(key=lambda o: o.id)

    return AllocationIR(
        ir_version=IR_VERSION,
        variables=variables,
        constraints=constraints,
        objective_terms=objective_terms,
        objective_sense=request.objective.sense,
        metadata={"request_id": request.request_id},
    )
