"""Objective decomposition sums to the realized objective (RM-P1-ALLOC-06).

The acceptance property (allocate.md §10; issue #6): a feasible plan's objective decomposition —
the per-family (roi / info_gain / value) breakdown — **sums to the realized objective**. Computed
over the solver-neutral IR from the same variable-value mapping the verifier uses, so the identity
holds by construction (the info-gain-vs-ROI split is the RM-P1-ALLOC-04 seam).
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from astro_mine.allocate import (
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
)
from astro_mine.allocate.constraints.config import CommsPolicy, ConstraintConfig
from astro_mine.allocate.explain import decompose_objective
from astro_mine.core.messages.enums import TaskKind
from astro_mine.core.units import J2000_EPOCH
from tests.allocate import constraint_factories as F
from tests.allocate.factories import anchor_request

_KINDS = list(TaskKind)


def test_skeleton_decomposition_is_the_value_family_and_sums_to_objective() -> None:
    request = anchor_request()
    ir = compile_request(request)
    alloc = AllocationPlanner(backend="cp-sat").solve(request)
    decomposition = decompose_objective(alloc.plan or [], ir)

    assert {c.family for c in decomposition.contributions} == {"value"}
    assert decomposition.total == pytest.approx(alloc.realized_objective)
    assert sum(c.value for c in decomposition.contributions) == pytest.approx(decomposition.total)


def test_infogain_vs_roi_split_is_reported_and_sums_to_objective() -> None:
    # An injected EVPI activates the info-gain family alongside the ROI family (RM-P1-ALLOC-04).
    ctx = F.context(info_values={"prospect-crater-a": 4.0})
    config = ConstraintConfig(comms=CommsPolicy(epoch0=J2000_EPOCH))
    request = anchor_request(
        objective=Objective(sense=ObjectiveSense.MAXIMIZE, weights={"roi": 1.0})
    )
    from astro_mine.allocate.constraints.compose import compile_with_constraints

    ir = compile_with_constraints(request, ctx, config=config).ir
    alloc = AllocationPlanner(backend="cp-sat").solve(request, context=ctx, config=config)
    assert alloc.status is AllocationStatus.OPTIMAL

    decomposition = alloc.objective_decomposition
    assert decomposition is not None
    families = {c.family for c in decomposition.contributions}
    assert "roi" in families and "info_gain" in families
    assert decomposition.total == pytest.approx(alloc.realized_objective)
    # The decomposition computed off the returned plan agrees with the one attached to the plan.
    assert decompose_objective(alloc.plan or [], ir).total == pytest.approx(decomposition.total)


def test_by_task_breakdown_sums_to_the_same_total() -> None:
    request = anchor_request()
    ir = compile_request(request)
    alloc = AllocationPlanner(backend="cp-sat").solve(request)

    per_family = decompose_objective(alloc.plan or [], ir)
    per_task = decompose_objective(alloc.plan or [], ir, by_task=True)
    assert per_task.total == pytest.approx(per_family.total)
    # Each per-task contribution carries a task id; the family-level one does not.
    assert all(c.task_id is not None for c in per_task.contributions)
    assert all(c.task_id is None for c in per_family.contributions)


@st.composite
def feasible_requests(draw: st.DrawFn) -> AllocationRequest:
    """A well-formed, eligible-by-construction request a CP-SAT solve can always satisfy."""
    n_tasks = draw(st.integers(min_value=1, max_value=4))
    n_assets = draw(st.integers(min_value=1, max_value=3))
    tasks: list[Task] = []
    for i in range(n_tasks):
        tasks.append(
            Task(
                task_id=f"t{i}",
                kind=draw(st.sampled_from(_KINDS)),
                time_windows=[TimeWindow(start_s=0.0, end_s=100000.0)],
                value=ValueEstimate(mean=draw(st.integers(min_value=-50, max_value=50).map(float))),
            )
        )
    return AllocationRequest(
        request_id="d",
        tasks=tasks,
        assets=[AssetRef(asset_id=f"a{k}") for k in range(n_assets)],
        objective=Objective(sense=draw(st.sampled_from(list(ObjectiveSense)))),
        budget=SolveBudget(deterministic=True, seed=draw(st.integers(0, 2**31 - 1))),
    )


@settings(max_examples=40, deadline=None)
@given(request=feasible_requests())
def test_decomposition_sums_to_realized_objective(request: AllocationRequest) -> None:
    ir = compile_request(request)
    alloc = AllocationPlanner(backend="cp-sat").solve(request)
    decomposition = decompose_objective(alloc.plan or [], ir)
    assert decomposition.total == pytest.approx(alloc.realized_objective)
    assert sum(c.value for c in decomposition.contributions) == pytest.approx(
        alloc.realized_objective
    )
