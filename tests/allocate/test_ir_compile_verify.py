"""IR compilation + the independent feasibility verifier (RM-P1-ALLOC-01).

``compile_request`` lifts a request into the solver-neutral structural skeleton (assignment
cover, time-window, precedence constraints; value objective terms); ``verify_feasible`` is the
Guard-recheckable oracle that a returned plan is structurally feasible against that IR
(acceptance: assignment covers required tasks, no time-window/precedence contradiction, and the
reported objective is correctly derived from the IR objective terms).
"""

from __future__ import annotations

from astro_mine.allocate import (
    Allocation,
    AllocationIR,
    AllocationPlanner,
    AllocationProvenance,
    AllocationRequest,
    AllocationStatus,
    AssetRef,
    AssetSchedule,
    ConstraintKind,
    ConstraintSense,
    ScheduledTask,
    Task,
    TimeWindow,
    ValueEstimate,
    VariableKind,
    VariableSemantic,
    compile_request,
    verify_feasible,
)
from astro_mine.allocate.model.ir.compile import window_envelope
from astro_mine.core.messages.enums import TaskKind
from tests.allocate.factories import anchor_request, solved, unwindowed_request

_PROV = AllocationProvenance(ir_version="0.1.0", backend="test")

# Anchor task kinds, for building plans by hand.
_KIND = {
    "prospect-crater-a": TaskKind.PROSPECT,
    "excavate-crater-a": TaskKind.EXCAVATE,
    "haul-to-plant": TaskKind.HAUL,
}


def _allocation(placements: dict[str, tuple[str, float]], *, realized: float | None) -> Allocation:
    """Build a feasible-shaped Allocation from ``task_id -> (asset_id, start_s)`` placements."""
    by_asset: dict[str, list[ScheduledTask]] = {}
    for task_id, (asset_id, start_s) in placements.items():
        by_asset.setdefault(asset_id, []).append(
            ScheduledTask(task_id=task_id, kind=_KIND[task_id], start_s=start_s, end_s=start_s)
        )
    plan = [
        AssetSchedule(asset_id=aid, tasks=sorted(sts, key=lambda s: (s.start_s, s.task_id)))
        for aid, sts in sorted(by_asset.items())
    ]
    return Allocation(
        status=AllocationStatus.FEASIBLE, plan=plan, realized_objective=realized, provenance=_PROV
    )


# --- compile: the structural skeleton --------------------------------------------


def test_compile_emits_assignment_start_cover_window_precedence() -> None:
    ir = compile_request(anchor_request())
    kinds = {v.semantic for v in ir.variables}
    assert kinds == {VariableSemantic.ASSIGNMENT, VariableSemantic.START_TIME}
    # one start var per task; assignment vars only for eligible (task, asset) pairs.
    start_vars = [v for v in ir.variables if v.semantic is VariableSemantic.START_TIME]
    assert len(start_vars) == 3
    assign_vars = [v for v in ir.variables if v.semantic is VariableSemantic.ASSIGNMENT]
    assert len(assign_vars) == 4  # prospect:1, excavate:1, haul:2 (rover + hauler)
    assert all(
        v.kind is VariableKind.BINARY and v.lower == 0.0 and v.upper == 1.0 for v in assign_vars
    )
    cover = [c for c in ir.constraints if c.kind is ConstraintKind.ASSIGNMENT_COVER]
    assert len(cover) == 3
    assert all(c.sense is ConstraintSense.EQ and c.rhs == 1.0 for c in cover)
    prec = [c for c in ir.constraints if c.kind is ConstraintKind.PRECEDENCE]
    assert len(prec) == 2
    assert all(c.sense is ConstraintSense.GE and c.rhs == 0.0 for c in prec)
    windows = [c for c in ir.constraints if c.kind is ConstraintKind.TIME_WINDOW]
    assert len(windows) == 6  # two (lo/hi) per windowed task
    # one objective term per assignment var, weighted by the task value mean.
    assert len(ir.objective_terms) == len(assign_vars)


def test_compile_is_sorted_and_referentially_intact() -> None:
    ir = compile_request(anchor_request())
    assert [v.id for v in ir.variables] == sorted(v.id for v in ir.variables)
    assert [c.id for c in ir.constraints] == sorted(c.id for c in ir.constraints)
    assert [o.id for o in ir.objective_terms] == sorted(o.id for o in ir.objective_terms)
    var_ids = {v.id for v in ir.variables}
    assert all(t.var_ref in var_ids for c in ir.constraints for t in c.terms)


def test_unwindowed_task_gets_open_envelope_and_no_window_constraints() -> None:
    request = unwindowed_request()
    assert window_envelope(request.tasks[0]) == (0.0, 1.0e12)
    ir = compile_request(request)
    assert not [c for c in ir.constraints if c.kind is ConstraintKind.TIME_WINDOW]
    # no required capabilities → every asset is eligible for every task.
    assign = [v for v in ir.variables if v.semantic is VariableSemantic.ASSIGNMENT]
    assert len(assign) == 4  # 2 tasks x 2 assets


# --- verify: the feasibility oracle ----------------------------------------------


def test_solved_anchor_verifies_against_its_ir() -> None:
    ir, allocation = solved(anchor_request())
    assert allocation.status is AllocationStatus.FEASIBLE
    assert allocation.realized_objective == 65.0  # 10 + 40 + 15
    assert verify_feasible(allocation, ir) is True


def test_an_alternative_valid_assignment_also_verifies() -> None:
    # Hauling on the rover (also MOBILITY_WHEELED) is a different but equally feasible cover.
    ir = compile_request(anchor_request())
    allocation = _allocation(
        {
            "prospect-crater-a": ("prospector-rover-1", 0.0),
            "excavate-crater-a": ("excavator-1", 1800.0),
            "haul-to-plant": ("prospector-rover-1", 3600.0),
        },
        realized=65.0,
    )
    assert verify_feasible(allocation, ir) is True


def test_verify_rejects_a_missing_plan() -> None:
    ir, _ = solved(anchor_request())
    infeasible = Allocation(status=AllocationStatus.INFEASIBLE, provenance=_PROV)
    assert verify_feasible(infeasible, ir) is False


def test_verify_rejects_an_uncovered_task() -> None:
    ir = compile_request(anchor_request())
    # haul-to-plant is never scheduled → its start variable is uncovered.
    allocation = _allocation(
        {
            "prospect-crater-a": ("prospector-rover-1", 0.0),
            "excavate-crater-a": ("excavator-1", 1800.0),
        },
        realized=50.0,
    )
    assert verify_feasible(allocation, ir) is False


def test_verify_rejects_a_task_scheduled_twice() -> None:
    ir = compile_request(anchor_request())
    dup = ScheduledTask(task_id="prospect-crater-a", kind=TaskKind.PROSPECT, start_s=0.0, end_s=0.0)
    allocation = Allocation(
        status=AllocationStatus.FEASIBLE,
        plan=[
            AssetSchedule(asset_id="prospector-rover-1", tasks=[dup]),
            AssetSchedule(asset_id="excavator-1", tasks=[dup]),
        ],
        realized_objective=65.0,
        provenance=_PROV,
    )
    assert verify_feasible(allocation, ir) is False


def test_verify_rejects_a_precedence_violation() -> None:
    ir = compile_request(anchor_request())
    # excavate starts *before* its prospect predecessor.
    allocation = _allocation(
        {
            "prospect-crater-a": ("prospector-rover-1", 3000.0),
            "excavate-crater-a": ("excavator-1", 1800.0),
            "haul-to-plant": ("hauler-1", 3600.0),
        },
        realized=65.0,
    )
    assert verify_feasible(allocation, ir) is False


def test_verify_rejects_a_time_window_violation() -> None:
    ir = compile_request(anchor_request())
    # prospect starts past its window's upper bound (3600 s).
    allocation = _allocation(
        {
            "prospect-crater-a": ("prospector-rover-1", 5000.0),
            "excavate-crater-a": ("excavator-1", 5000.0),
            "haul-to-plant": ("hauler-1", 5000.0),
        },
        realized=65.0,
    )
    assert verify_feasible(allocation, ir) is False


def test_verify_rejects_a_missing_realized_objective() -> None:
    ir, allocation = solved(anchor_request())
    tampered = allocation.model_copy(update={"realized_objective": None})
    assert verify_feasible(tampered, ir) is False


def test_verify_rejects_a_dishonest_objective() -> None:
    ir, allocation = solved(anchor_request())
    tampered = allocation.model_copy(update={"realized_objective": 999.0})
    assert verify_feasible(tampered, ir) is False


def test_solve_reports_a_window_precedence_conflict_as_infeasible() -> None:
    # B must follow A, but A cannot start until after B's window closes — no feasible start.
    request = AllocationRequest(
        request_id="conflict",
        tasks=[
            Task(
                task_id="a",
                kind=TaskKind.PROSPECT,
                time_windows=[TimeWindow(start_s=1000.0, end_s=2000.0)],
                value=ValueEstimate(mean=1.0),
            ),
            Task(
                task_id="b",
                kind=TaskKind.HAUL,
                precedence=["a"],
                time_windows=[TimeWindow(start_s=0.0, end_s=500.0)],
                value=ValueEstimate(mean=1.0),
            ),
        ],
        assets=[AssetRef(asset_id="rover-a")],
    )
    allocation = AllocationPlanner().solve(request)
    assert allocation.status is AllocationStatus.INFEASIBLE
    assert allocation.plan is None
    assert allocation.provenance.backend == "trivial-stub"


def test_verify_on_an_empty_ir_and_empty_plan_is_trivially_true() -> None:
    empty_ir = AllocationIR(objective_sense=compile_request(anchor_request()).objective_sense)
    allocation = Allocation(
        status=AllocationStatus.FEASIBLE,
        plan=[],
        realized_objective=0.0,
        provenance=_PROV,
    )
    assert verify_feasible(allocation, empty_ir) is True
