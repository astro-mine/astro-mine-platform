"""Property: every CP-SAT plan is feasible against its IR and its objective is honest (ALLOC-02).

The solver invariant the acceptance criteria name (allocate.md §10; RM-P1-ALLOC-02): for a
well-formed request, the plan the **CP-SAT** backend returns satisfies
:func:`~astro_mine.allocate.verify_feasible` against the compiled IR, and the reported
``realized_objective`` equals the objective recomputed from the IR terms and the plan's
assignment. Requests are generated eligible-by-construction (no capability requirements, wide
integer-second windows, DAG precedence) so a feasible plan always exists — the property is that
CP-SAT *finds, verifies, and honestly scores* one.
"""

from __future__ import annotations

import importlib.util

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from astro_mine.allocate import (
    Allocation,
    AllocationIR,
    AllocationPlanner,
    AllocationRequest,
    AllocationStatus,
    AssetRef,
    Objective,
    ObjectiveSense,
    SolveBudget,
    Task,
    TimeWindow,
    ValueEstimate,
    compile_request,
    verify_feasible,
)
from astro_mine.core.messages.enums import TaskKind

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ortools") is None, reason="OR-Tools (cp-sat backend) not installed"
)

_KINDS = list(TaskKind)


@st.composite
def feasible_requests(draw: st.DrawFn) -> AllocationRequest:
    """A well-formed, eligible-by-construction request a CP-SAT solve can always satisfy."""
    n_tasks = draw(st.integers(min_value=1, max_value=5))
    n_assets = draw(st.integers(min_value=1, max_value=3))
    tasks: list[Task] = []
    for i in range(n_tasks):
        preds = draw(
            st.lists(
                st.sampled_from([f"t{j}" for j in range(i)]) if i else st.nothing(),
                max_size=i,
                unique=True,
            )
        )
        # A wide integer-second window keeps the model on the exact CP-SAT time grid.
        windows = draw(st.lists(st.just(TimeWindow(start_s=0.0, end_s=100000.0)), max_size=1))
        tasks.append(
            Task(
                task_id=f"t{i}",
                kind=draw(st.sampled_from(_KINDS)),
                precedence=preds,
                time_windows=windows,
                value=ValueEstimate(mean=draw(st.integers(min_value=-50, max_value=50).map(float))),
            )
        )
    assets = [AssetRef(asset_id=f"a{k}") for k in range(n_assets)]
    sense = draw(st.sampled_from([ObjectiveSense.MAXIMIZE, ObjectiveSense.MINIMIZE]))
    return AllocationRequest(
        request_id=draw(st.text(min_size=1, max_size=6, alphabet="abcdef0123456789")),
        tasks=tasks,
        assets=assets,
        objective=Objective(sense=sense),
        budget=SolveBudget(deterministic=True, seed=draw(st.integers(0, 2**31 - 1))),
    )


def _objective_from_plan(ir: AllocationIR, alloc: Allocation) -> float:
    assignment = {st.task_id: sched.asset_id for sched in (alloc.plan or []) for st in sched.tasks}
    var_by_id = {v.id: v for v in ir.variables}
    total = 0.0
    for term in ir.objective_terms:
        var = var_by_id[term.var_ref]
        if var.task_ref is not None and assignment.get(var.task_ref) == var.asset_ref:
            total += term.coefficient
    return total


@settings(max_examples=40, deadline=None)
@given(request=feasible_requests())
def test_cpsat_plan_is_feasible_and_honestly_scored(request: AllocationRequest) -> None:
    ir = compile_request(request)
    alloc = AllocationPlanner(backend="cp-sat").solve(request)

    assert alloc.status in (AllocationStatus.OPTIMAL, AllocationStatus.FEASIBLE)
    # The plan re-checks against the IR by the independent oracle (never trusting the solver).
    assert verify_feasible(alloc, ir) is True
    # The reported objective equals the IR-recomputed objective of the plan's assignment.
    assert alloc.realized_objective == pytest.approx(_objective_from_plan(ir, alloc))


@settings(max_examples=40, deadline=None)
@given(request=feasible_requests())
def test_cpsat_optimal_gap_is_zero(request: AllocationRequest) -> None:
    alloc = AllocationPlanner(backend="cp-sat").solve(request)
    if alloc.status is AllocationStatus.OPTIMAL:
        assert alloc.optimality_gap == pytest.approx(0.0)
