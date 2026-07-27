"""Temporal feasibility: no double-booked asset, no start in a window gap (RM-P1-ALLOC-02).

The two ways the model used to admit a plan that cannot physically run, and the fix for each
(allocate.md §3/§4/§10 — "returned plans are always feasible against the model"; ``LUNAR-FR-004``):

1. **A double-booked asset.** Nothing emitted a per-asset resource constraint from a real request,
   so one rover could be assigned two long tasks whose intervals overlapped and the plan still
   verified "feasible". Now :func:`~astro_mine.allocate.compile_request` emits a ``NO_OVERLAP``
   constraint per asset that could be double-booked, every backend schedules against it, and
   :func:`~astro_mine.allocate.verify_feasible` re-checks it from the **IR's** interval sizes — so a
   backend cannot hide an overlap by under-reporting a task's ``end_s``.
2. **A start inside an availability gap.** A task's plural disjoint ``time_windows`` were collapsed
   to a single ``[min(start), max(end)]`` envelope, which spans the *gaps between* them. Now the
   compiler emits the windows as an exact disjunction (a ``WINDOW_SELECT`` variable per window plus
   ``twin_pick``/``twin_lo``/``twin_hi``), so a start in a gap satisfies no window selector and the
   verifier rejects the plan.

Every test here fails against the pre-fix behavior — that is the point of the file.
"""

from __future__ import annotations

import importlib.util
from typing import cast

import pytest
from pydantic import ValidationError

from astro_mine.allocate import (
    Allocation,
    AllocationIR,
    AllocationPlanner,
    AllocationProvenance,
    AllocationRequest,
    AllocationStatus,
    AssetRef,
    AssetSchedule,
    Constraint,
    ConstraintKind,
    ConstraintSense,
    ConstraintTerm,
    DecisionVariable,
    ObjectiveSense,
    ObjectiveTerm,
    ScheduledTask,
    Task,
    TimeWindow,
    ValueEstimate,
    VariableKind,
    VariableSemantic,
    compile_request,
    cumulative_slack,
    earliest_start_in_windows,
    verify_feasible,
    window_choices,
)
from astro_mine.allocate.model.ir.compile import cumulative_constraint_id, no_overlap_constraint_id
from astro_mine.core.messages.enums import TaskKind
from astro_mine.core.sadf import CapabilityTag
from tests.allocate import constraint_factories as F
from tests.allocate.factories import anchor_request

_HAS_ORTOOLS = importlib.util.find_spec("ortools") is not None
requires_ortools = pytest.mark.skipif(not _HAS_ORTOOLS, reason="OR-Tools not installed")


# --- fixtures ---------------------------------------------------------------------


def _contended_request(**overrides: object) -> AllocationRequest:
    """Two 600 s tasks and exactly **one** asset able to do either: the asset must serialize them.

    The instance the old model got wrong — both tasks fit the window individually, so nothing
    stopped a solver from starting both at t=0 on the same rover.
    """
    kwargs: dict[str, object] = dict(
        request_id="contended-001",
        tasks=[
            Task(
                task_id="survey-a",
                kind=TaskKind.PROSPECT,
                required_capabilities=[CapabilityTag.PROSPECTING_NEUTRON],
                time_windows=[TimeWindow(start_s=0.0, end_s=3600.0)],
                duration_s=600.0,
                value=ValueEstimate(mean=10.0),
            ),
            Task(
                task_id="survey-b",
                kind=TaskKind.PROSPECT,
                required_capabilities=[CapabilityTag.PROSPECTING_NEUTRON],
                time_windows=[TimeWindow(start_s=0.0, end_s=3600.0)],
                duration_s=600.0,
                value=ValueEstimate(mean=10.0),
            ),
        ],
        assets=[
            AssetRef(asset_id="rover-1", capability_tags=[CapabilityTag.PROSPECTING_NEUTRON]),
        ],
    )
    kwargs.update(overrides)
    return AllocationRequest(**kwargs)  # type: ignore[arg-type]


def _gapped_request(**overrides: object) -> AllocationRequest:
    """One task with two **disjoint** windows — ``[0, 100]`` and ``[500, 600]`` — and a 300 s gap.

    The envelope ``[0, 600]`` the compiler used to emit admits a start at 300, which is in the
    shadow between the two passes and cannot be run.
    """
    kwargs: dict[str, object] = dict(
        request_id="gapped-001",
        tasks=[
            Task(
                task_id="relay-window",
                kind=TaskKind.PROSPECT,
                time_windows=[
                    TimeWindow(start_s=0.0, end_s=100.0),
                    TimeWindow(start_s=500.0, end_s=600.0),
                ],
                value=ValueEstimate(mean=5.0),
            )
        ],
        assets=[AssetRef(asset_id="rover-1")],
    )
    kwargs.update(overrides)
    return AllocationRequest(**kwargs)  # type: ignore[arg-type]


def _plan_at(*placements: tuple[str, str, float, float]) -> list[AssetSchedule]:
    """A plan from ``(asset, task, start, end)`` placements, grouped and ordered per asset."""
    by_asset: dict[str, list[ScheduledTask]] = {}
    for asset_id, task_id, start_s, end_s in placements:
        by_asset.setdefault(asset_id, []).append(
            ScheduledTask(task_id=task_id, kind=TaskKind.PROSPECT, start_s=start_s, end_s=end_s)
        )
    return [
        AssetSchedule(asset_id=a, tasks=sorted(t, key=lambda s: s.start_s))
        for a, t in sorted(by_asset.items())
    ]


def _allocation(plan: list[AssetSchedule], objective: float) -> Allocation:
    return Allocation(
        status=AllocationStatus.FEASIBLE,
        plan=plan,
        realized_objective=objective,
        provenance=AllocationProvenance(ir_version="0.1.0", backend="test"),
    )


# --- the compiler emits scheduling constraints from a real request ------------------


def test_a_real_compiled_request_emits_a_per_asset_no_overlap_constraint() -> None:
    ir = compile_request(_contended_request())
    scheduling = [c for c in ir.constraints if c.kind is ConstraintKind.NO_OVERLAP]

    assert [c.id for c in scheduling] == [no_overlap_constraint_id("rover-1")]
    # One term per candidate task: its start variable, carrying the interval size as coefficient.
    assert {(t.var_ref, t.coefficient) for t in scheduling[0].terms} == {
        ("start::survey-a", 600.0),
        ("start::survey-b", 600.0),
    }


def test_the_constrained_compile_re_emits_no_overlap_over_resolved_per_pair_durations() -> None:
    # The anchor's prospector-rover-1 is eligible for both the prospect and the haul; the cost table
    # gives it a *different* duration for each. The scheduling constraint must carry those, not the
    # request's nominal ones.
    comp = (
        F.context(),
        F.cost_table(
            {
                ("prospect-crater-a", "prospector-rover-1"): (600.0, 1.0e6),
                ("haul-to-plant", "prospector-rover-1"): (900.0, 1.0e6),
            }
        ),
    )
    from astro_mine.allocate import compile_with_constraints

    ir = compile_with_constraints(anchor_request(), comp[0], costs=comp[1]).ir
    scheduling = [c for c in ir.constraints if c.kind is ConstraintKind.NO_OVERLAP]

    assert [c.id for c in scheduling] == [no_overlap_constraint_id("prospector-rover-1")]
    assert {(t.var_ref, t.coefficient) for t in scheduling[0].terms} == {
        ("start::prospect-crater-a", 600.0),
        ("start::haul-to-plant", 900.0),
    }


def test_a_request_with_no_durations_compiles_to_the_pre_scheduling_ir() -> None:
    # Zero-length point tasks occupy no time: no scheduling constraint is emitted at all, so the
    # existing golden plans (which declare no durations) are untouched.
    ir = compile_request(anchor_request())
    assert not [c for c in ir.constraints if c.kind is ConstraintKind.NO_OVERLAP]


# --- disjoint time windows compile to a disjunction, not an envelope ----------------


def test_disjoint_windows_compile_to_a_disjunction_not_a_min_max_envelope() -> None:
    ir = compile_request(_gapped_request())
    choices = window_choices(ir)

    # One selector per window, carrying that window's *own* bounds — not the [0, 600] envelope.
    assert [(w.start_s, w.end_s) for w in choices["relay-window"]] == [(0.0, 100.0), (500.0, 600.0)]
    # And an exactly-one over them, so precisely one window hosts the task.
    pick = next(c for c in ir.constraints if c.id == "twin_pick::relay-window")
    assert pick.rhs == 1.0 and len(pick.terms) == 2


def test_a_single_window_needs_no_selector() -> None:
    ir = compile_request(anchor_request())
    assert window_choices(ir) == {}


def test_earliest_start_in_windows_snaps_forward_out_of_a_gap() -> None:
    task = _gapped_request().tasks[0]
    assert earliest_start_in_windows(task, 0.0) == 0.0
    assert earliest_start_in_windows(task, 50.0) == 50.0
    assert earliest_start_in_windows(task, 300.0) == 500.0  # in the gap → snapped to window 2
    assert earliest_start_in_windows(task, 550.0) == 550.0
    assert earliest_start_in_windows(task, 700.0) is None  # past every window


# --- verify_feasible rejects both failure modes -------------------------------------


def test_verify_feasible_rejects_two_overlapping_tasks_on_one_asset() -> None:
    ir = compile_request(_contended_request())

    # The plan *reports* two zero-length points, so it is a structurally valid AssetSchedule — but
    # the IR reserves 600 s for each, and they are only 100 s apart. A backend cannot hide an
    # overlap behind an under-reported end_s.
    dishonest = _allocation(
        _plan_at(
            ("rover-1", "survey-a", 0.0, 0.0),
            ("rover-1", "survey-b", 100.0, 100.0),
        ),
        objective=20.0,
    )
    assert not verify_feasible(dishonest, ir)

    # Serialized on the same asset (600 s apart) it verifies.
    honest = _allocation(
        _plan_at(
            ("rover-1", "survey-a", 0.0, 600.0),
            ("rover-1", "survey-b", 600.0, 1200.0),
        ),
        objective=20.0,
    )
    assert verify_feasible(honest, ir)


def test_asset_schedule_refuses_to_represent_a_double_booked_asset() -> None:
    with pytest.raises(ValidationError, match="double-books the asset"):
        AssetSchedule(
            asset_id="rover-1",
            tasks=[
                ScheduledTask(task_id="survey-a", kind=TaskKind.PROSPECT, start_s=0.0, end_s=600.0),
                ScheduledTask(
                    task_id="survey-b", kind=TaskKind.PROSPECT, start_s=100.0, end_s=700.0
                ),
            ],
        )


def test_verify_feasible_rejects_a_task_scheduled_in_a_gap_between_disjoint_windows() -> None:
    ir = compile_request(_gapped_request())

    # 300 s is inside the [0, 600] *envelope* but inside neither window — the old encoding accepted
    # exactly this.
    in_the_gap = _allocation(_plan_at(("rover-1", "relay-window", 300.0, 300.0)), objective=5.0)
    assert not verify_feasible(in_the_gap, ir)

    # Both real windows verify.
    for start in (50.0, 550.0):
        placed = _allocation(_plan_at(("rover-1", "relay-window", start, start)), objective=5.0)
        assert verify_feasible(placed, ir), f"start {start} lies inside a declared window"


# --- every shipped backend honors both families -------------------------------------


@pytest.mark.parametrize(
    "backend", ["trivial-stub", pytest.param("cp-sat", marks=requires_ortools)]
)
def test_no_backend_double_books_an_asset(backend: str) -> None:
    request = _contended_request()
    ir = compile_request(request)
    allocation = AllocationPlanner(backend=backend).solve(request)

    assert allocation.status in (AllocationStatus.OPTIMAL, AllocationStatus.FEASIBLE)
    assert verify_feasible(allocation, ir)

    (schedule,) = allocation.plan or []
    first, second = schedule.tasks
    assert second.start_s >= first.start_s + 600.0, "the asset must serialize its two 600 s tasks"


@pytest.mark.parametrize(
    "backend", ["trivial-stub", pytest.param("cp-sat", marks=requires_ortools)]
)
def test_no_backend_starts_a_task_inside_a_window_gap(backend: str) -> None:
    # A predecessor that ends deep inside the gap forces the successor *out* of window 1: a solver
    # reading the [0, 600] envelope would happily start it at 300.
    request = _gapped_request(
        tasks=[
            Task(
                task_id="pre",
                kind=TaskKind.PROSPECT,
                time_windows=[TimeWindow(start_s=300.0, end_s=300.0)],
                value=ValueEstimate(mean=1.0),
            ),
            Task(
                task_id="relay-window",
                kind=TaskKind.PROSPECT,
                time_windows=[
                    TimeWindow(start_s=0.0, end_s=100.0),
                    TimeWindow(start_s=500.0, end_s=600.0),
                ],
                precedence=["pre"],
                value=ValueEstimate(mean=5.0),
            ),
        ]
    )
    ir = compile_request(request)
    allocation = AllocationPlanner(backend=backend).solve(request)

    assert allocation.status in (AllocationStatus.OPTIMAL, AllocationStatus.FEASIBLE)
    assert verify_feasible(allocation, ir)

    placed = {st.task_id: st.start_s for sched in (allocation.plan or []) for st in sched.tasks}
    assert placed["relay-window"] >= 500.0, (
        "the successor must land in the second window, not the gap"
    )


@requires_ortools
def test_cpsat_proves_an_over_subscribed_asset_infeasible_and_names_it() -> None:
    # Two 2000 s tasks on one asset, but both must *start* within [0, 1000]: serializing them puts
    # the second start at 2000, outside its window, so no conflict-free schedule exists. The IIS
    # must say so — and it can only name the resource because the scheduling global is reified into
    # the assumption model (it used to raise NotImplementedError instead).
    request = _contended_request(
        tasks=[
            Task(
                task_id="survey-a",
                kind=TaskKind.PROSPECT,
                required_capabilities=[CapabilityTag.PROSPECTING_NEUTRON],
                time_windows=[TimeWindow(start_s=0.0, end_s=1000.0)],
                duration_s=2000.0,
                value=ValueEstimate(mean=10.0),
            ),
            Task(
                task_id="survey-b",
                kind=TaskKind.PROSPECT,
                required_capabilities=[CapabilityTag.PROSPECTING_NEUTRON],
                time_windows=[TimeWindow(start_s=0.0, end_s=1000.0)],
                duration_s=2000.0,
                value=ValueEstimate(mean=10.0),
            ),
        ]
    )
    allocation = AllocationPlanner(backend="cp-sat").solve(request)

    assert allocation.status is AllocationStatus.INFEASIBLE
    certificate = allocation.infeasibility_certificate
    assert certificate is not None
    assert no_overlap_constraint_id("rover-1") in certificate.constraint_ids
    assert "no conflict-free schedule on: rover-1" in (certificate.explanation or "")


@requires_ortools
def test_a_saturated_asset_is_reported_as_a_binding_constraint() -> None:
    # Both tasks on one rover with exactly no idle time between them: the resource is what bound the
    # plan, and an operator must be told so (LUNAR-UX-004).
    request = _contended_request(
        tasks=[
            Task(
                task_id="survey-a",
                kind=TaskKind.PROSPECT,
                required_capabilities=[CapabilityTag.PROSPECTING_NEUTRON],
                time_windows=[TimeWindow(start_s=0.0, end_s=1200.0)],
                duration_s=600.0,
                value=ValueEstimate(mean=10.0),
            ),
            Task(
                task_id="survey-b",
                kind=TaskKind.PROSPECT,
                required_capabilities=[CapabilityTag.PROSPECTING_NEUTRON],
                time_windows=[TimeWindow(start_s=0.0, end_s=1200.0)],
                duration_s=600.0,
                value=ValueEstimate(mean=10.0),
            ),
        ]
    )
    allocation = AllocationPlanner(backend="cp-sat").solve(request)

    binding = {b.constraint_id: b for b in allocation.binding_constraints}
    resource = binding[no_overlap_constraint_id("rover-1")]
    assert resource.kind is ConstraintKind.NO_OVERLAP
    assert resource.source == "schedule:resource"


# --- the CUMULATIVE family (a shared, capacity-bearing resource) --------------------------------


def _cumulative_ir(capacity: float, size: float = 600.0) -> AllocationIR:
    """Three tasks on three rovers, all contending for one **shared** ``capacity``-bay dock.

    ``NO_OVERLAP`` is what the compiler emits per *asset* (a rover does one task at a time).
    ``CUMULATIVE`` is the IR's other scheduling family, for a resource several assets share at once
    — a charging dock with N bays, a plant hopper. Its resource is **not** an asset, so no
    ``assign::{task}::{dock}`` variable exists and every interval is *mandatory* (each task uses the
    dock whoever performs it) — the other half of the channeling rule in
    :func:`~astro_mine.allocate.model.ir.schedule.resource_intervals` and the CP-SAT lowering.
    Its semantics come from the same module the verifier and the lowering share, so it is checked
    exactly the way ``NO_OVERLAP`` is.
    """
    tasks = ("survey-a", "survey-b", "survey-c")
    rovers = {t: f"rover-{index}" for index, t in enumerate(tasks)}
    return AllocationIR(
        variables=[
            *(
                DecisionVariable(
                    id=f"assign::{t}::{rovers[t]}",
                    kind=VariableKind.BINARY,
                    lower=0.0,
                    upper=1.0,
                    semantic=VariableSemantic.ASSIGNMENT,
                    task_ref=t,
                    asset_ref=rovers[t],
                )
                for t in tasks
            ),
            *(
                DecisionVariable(
                    id=f"start::{t}",
                    kind=VariableKind.CONTINUOUS,
                    lower=0.0,
                    upper=3600.0,
                    semantic=VariableSemantic.START_TIME,
                    task_ref=t,
                )
                for t in tasks
            ),
        ],
        constraints=[
            *(
                Constraint(
                    id=f"cover::{t}",
                    kind=ConstraintKind.ASSIGNMENT_COVER,
                    terms=[ConstraintTerm(var_ref=f"assign::{t}::{rovers[t]}", coefficient=1.0)],
                    sense=ConstraintSense.EQ,
                    rhs=1.0,
                )
                for t in tasks
            ),
            Constraint(
                id=cumulative_constraint_id("dock-1"),
                kind=ConstraintKind.CUMULATIVE,
                terms=[ConstraintTerm(var_ref=f"start::{t}", coefficient=size) for t in tasks],
                sense=ConstraintSense.LE,
                rhs=capacity,
            ),
        ],
        objective_terms=[
            ObjectiveTerm(id=f"obj::{t}", var_ref=f"assign::{t}::{rovers[t]}", coefficient=1.0)
            for t in tasks
        ],
        objective_sense=ObjectiveSense.MAXIMIZE,
    )


def test_verify_feasible_rejects_a_resource_over_its_cumulative_capacity() -> None:
    ir = _cumulative_ir(capacity=2.0)

    # All three rovers occupy the 2-bay dock at once: peak concurrency 3 > capacity 2.
    over = _allocation(
        _plan_at(
            ("rover-0", "survey-a", 0.0, 600.0),
            ("rover-1", "survey-b", 0.0, 600.0),
            ("rover-2", "survey-c", 0.0, 600.0),
        ),
        objective=3.0,
    )
    assert not verify_feasible(over, ir)

    # Two share the dock and the third waits for a bay: peak concurrency 2 == capacity.
    within = _allocation(
        _plan_at(
            ("rover-0", "survey-a", 0.0, 600.0),
            ("rover-1", "survey-b", 0.0, 600.0),
            ("rover-2", "survey-c", 600.0, 1200.0),
        ),
        objective=3.0,
    )
    assert verify_feasible(within, ir)


def test_cumulative_slack_is_the_spare_capacity_at_the_busiest_instant() -> None:
    assert cumulative_slack([], capacity=2.0) is None  # vacuous: nothing present
    assert cumulative_slack([(0.0, 10.0)], capacity=2.0) == 1.0
    assert cumulative_slack([(0.0, 10.0), (0.0, 10.0)], capacity=2.0) == 0.0  # tight
    assert cumulative_slack([(0.0, 10.0), (5.0, 15.0), (6.0, 8.0)], capacity=2.0) == -1.0
    # Back-to-back intervals do not contend: the earlier ends exactly as the later starts.
    assert cumulative_slack([(0.0, 10.0), (10.0, 20.0)], capacity=1.0) == 0.0


@requires_ortools
def test_cpsat_honors_a_shared_cumulative_capacity() -> None:
    from ortools.sat.python import cp_model

    from astro_mine.allocate.model.compile.cpsat import lower_to_cpsat

    compiled = lower_to_cpsat(_cumulative_ir(capacity=2.0))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    # cast: OR-Tools types `Solve()` as the enum wrapper class, not its int-valued members — see
    # the note in tests/test_explain_iis.py::_is_infeasible_without.
    assert cast(int, solver.Solve(compiled.model)) in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    starts = sorted(
        solver.Value(compiled.variables[f"start::{t}"])
        for t in ("survey-a", "survey-b", "survey-c")
    )
    # At most two of the three 600 s tasks may hold a bay at any instant.
    assert starts[2] >= starts[0] + 600.0
