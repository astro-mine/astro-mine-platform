"""Property: repeated deterministic solves of a golden instance are byte-identical (RM-P1-ALLOC-07).

Determinism as reproducibility and integrity (principle 8; conventions.md §11): same model + same
seed + same pinned solver ⇒ same plan, so a [Bench](bench.md) score reproduces exactly and a
tampered plan is detectable. Where the golden-plan gate pins *one* anchor, this asserts the
invariant across many generated instances: two deterministic-mode solves of the same request yield
a byte-identical reproducible artifact and a stable objective.
"""

from __future__ import annotations

import importlib.util
import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from astro_mine.allocate import (
    Allocation,
    AllocationPlanner,
    AllocationRequest,
    AssetRef,
    Objective,
    ObjectiveSense,
    SolveBudget,
    Task,
    TimeWindow,
    ValueEstimate,
)
from astro_mine.core.messages.enums import TaskKind

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ortools") is None, reason="OR-Tools (cp-sat backend) not installed"
)


def _reproducible(alloc: Allocation) -> str:
    return json.dumps(
        {
            "status": alloc.status.value,
            "realized_objective": alloc.realized_objective,
            "optimality_gap": alloc.optimality_gap,
            "plan": [a.model_dump(mode="json") for a in (alloc.plan or [])],
            "binding_constraints": [b.model_dump(mode="json") for b in alloc.binding_constraints],
            "objective_decomposition": (
                alloc.objective_decomposition.model_dump(mode="json")
                if alloc.objective_decomposition is not None
                else None
            ),
        },
        sort_keys=True,
    )


@st.composite
def golden_instances(draw: st.DrawFn) -> AllocationRequest:
    """A well-formed, eligible-by-construction request with a fixed deterministic budget."""
    n_tasks = draw(st.integers(min_value=1, max_value=5))
    n_assets = draw(st.integers(min_value=1, max_value=3))
    tasks = [
        Task(
            task_id=f"t{i}",
            kind=draw(st.sampled_from(list(TaskKind))),
            time_windows=[TimeWindow(start_s=0.0, end_s=100000.0)],
            value=ValueEstimate(mean=draw(st.integers(min_value=-50, max_value=50).map(float))),
        )
        for i in range(n_tasks)
    ]
    return AllocationRequest(
        request_id=draw(st.text(min_size=1, max_size=6, alphabet="abcdef0123456789")),
        tasks=tasks,
        assets=[AssetRef(asset_id=f"a{k}") for k in range(n_assets)],
        objective=Objective(sense=draw(st.sampled_from(list(ObjectiveSense)))),
        budget=SolveBudget(deterministic=True, seed=draw(st.integers(0, 2**31 - 1))),
    )


@settings(max_examples=50, deadline=None)
@given(request=golden_instances())
def test_repeated_deterministic_solves_are_byte_identical(request: AllocationRequest) -> None:
    first = AllocationPlanner(backend="cp-sat").solve(request)
    second = AllocationPlanner(backend="cp-sat").solve(request)
    # Same model + seed + pinned solver ⇒ byte-identical reproducible artifact.
    assert _reproducible(first) == _reproducible(second)
    # The reported objective is stable across solves.
    assert first.realized_objective == second.realized_objective
