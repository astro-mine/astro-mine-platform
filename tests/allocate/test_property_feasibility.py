"""Property: a solved plan is always feasible against its IR (RM-P1-ALLOC-01).

The solver invariant Hypothesis asserts (allocate.md §10: "returned plans are always feasible
against the model; objective is correctly computed"): for a well-formed request, the compiled
IR + the plan the (stub) solver returns satisfy :func:`~astro_mine.allocate.verify_feasible`,
and the compile is byte-stable. Requests are generated eligible-by-construction (no capability
requirements) with wide windows and DAG precedence, so a feasible plan always exists — the
property is that the pipeline *finds and reports* one honestly.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from astro_mine.allocate import (
    AllocationRequest,
    AllocationStatus,
    AssetRef,
    Objective,
    ObjectiveSense,
    Task,
    TimeWindow,
    ValueEstimate,
    compile_request,
    ir_to_wire,
    verify_feasible,
)
from astro_mine.allocate.api.planner import AllocationPlanner
from astro_mine.core.messages.enums import TaskKind

_KINDS = list(TaskKind)


@st.composite
def well_formed_requests(draw: st.DrawFn) -> AllocationRequest:
    n_tasks = draw(st.integers(min_value=1, max_value=6))
    n_assets = draw(st.integers(min_value=1, max_value=4))
    senses = st.sampled_from([ObjectiveSense.MAXIMIZE, ObjectiveSense.MINIMIZE])

    tasks: list[Task] = []
    for i in range(n_tasks):
        # Precedence references only strictly-earlier tasks → acyclic by construction.
        preds = draw(
            st.lists(
                st.sampled_from([f"t{j}" for j in range(i)]) if i else st.nothing(),
                max_size=i,
                unique=True,
            )
        )
        # A wide window (optional) always admits any precedence ordering.
        windows = draw(
            st.lists(
                st.just(TimeWindow(start_s=0.0, end_s=1.0e9)),
                max_size=1,
            )
        )
        tasks.append(
            Task(
                task_id=f"t{i}",
                kind=draw(st.sampled_from(_KINDS)),
                precedence=preds,
                time_windows=windows,
                value=ValueEstimate(
                    mean=draw(st.floats(min_value=-100.0, max_value=100.0, allow_nan=False)),
                ),
            )
        )

    assets = [AssetRef(asset_id=f"a{k}") for k in range(n_assets)]
    return AllocationRequest(
        request_id=draw(st.text(min_size=1, max_size=8, alphabet="abcdef0123456789")),
        tasks=tasks,
        assets=assets,
        objective=Objective(sense=draw(senses)),
    )


@given(request=well_formed_requests())
def test_solved_plan_is_feasible_against_its_ir(request: AllocationRequest) -> None:
    ir = compile_request(request)
    allocation = AllocationPlanner().solve(request)
    assert allocation.status is AllocationStatus.FEASIBLE
    assert verify_feasible(allocation, ir) is True


@given(request=well_formed_requests())
def test_compile_is_byte_stable(request: AllocationRequest) -> None:
    assert ir_to_wire(compile_request(request)) == ir_to_wire(compile_request(request))
