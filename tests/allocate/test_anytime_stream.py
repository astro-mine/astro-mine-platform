"""The anytime contract: monotone bounds, honest deadline status, warm start (RM-P1-ALLOC-05).

The acceptance criteria (issue #5; allocate.md §2 principle 1/2): incumbents stream with
**monotonically improving bounds** (a property test asserts it), stopping at a deadline yields the
best *feasible* plan with a correct explicit gap, and an unresolved solve returns an honest
``TIMEOUT``/``INFEASIBLE`` — never a plan, never a false certificate. A warm-start seam feeds an
incremental online re-solve, verified and never trusted by the exact layer.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable, Iterator, Mapping
from itertools import pairwise

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from astro_mine.allocate import (
    AllocationPlanner,
    AllocationRequest,
    AllocationStatus,
    AssetRef,
    BoundTracker,
    Incumbent,
    Objective,
    ObjectiveSense,
    SolveBudget,
    Task,
    TimeWindow,
    ValueEstimate,
    compile_request,
    finalize_status,
    hints_from,
    stream_incumbents,
    verify_feasible,
)
from astro_mine.allocate.model.ir.model import AllocationIR
from astro_mine.allocate.solvers.base import Solver
from tests.allocate.factories import anchor_request

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ortools") is None, reason="OR-Tools (cp-sat backend) not installed"
)


class _ScriptedSolver:
    """A :class:`Solver` that replays a fixed incumbent list — the seam under test, no CP-SAT."""

    def __init__(self, incumbents: list[Incumbent]) -> None:
        self._incumbents = incumbents
        self.seen_hints: Mapping[str, float] | None = None

    def solve(
        self, ir: AllocationIR, budget: SolveBudget, *, hints: Mapping[str, float] | None = None
    ) -> Iterator[Incumbent]:
        self.seen_hints = hints
        return iter(self._incumbents)


def _inc(objective: float, bound: float | None, status: AllocationStatus) -> Incumbent:
    gap = None if bound is None else abs(bound - objective) / max(abs(objective), 1.0)
    return Incumbent(objective=objective, bound=bound, gap=gap, plan=[], status=status)


def _bounds(incumbents: Iterable[Incumbent]) -> list[float]:
    """The exposed bounds of incumbents that are expected to carry one (a missing bound is a
    failure of the case under test, not a `None` to compare against)."""
    proven = []
    for incumbent in incumbents:
        assert incumbent.bound is not None
        proven.append(incumbent.bound)
    return proven


def test_scripted_solver_is_a_solver() -> None:
    assert isinstance(_ScriptedSolver([]), Solver)


# --- BoundTracker: the exposed bound is clamped monotone regardless of the raw bound ---------


def test_maximize_upper_bound_is_clamped_non_increasing() -> None:
    tracker = BoundTracker(ObjectiveSense.MAXIMIZE)
    # A raw upper bound that loosens (50 → 60) must be clamped down to the tightest proven (50).
    raw = [
        _inc(10.0, 100.0, AllocationStatus.FEASIBLE),
        _inc(20.0, 50.0, AllocationStatus.FEASIBLE),
        _inc(25.0, 60.0, AllocationStatus.OPTIMAL),
    ]
    exposed = [tracker.track(i) for i in raw]
    bounds = _bounds(exposed)
    assert bounds == [100.0, 50.0, 50.0]
    assert all(b <= a for a, b in pairwise(bounds))  # non-increasing
    # The gap is recomputed against the clamped bound, not the raw one.
    assert exposed[-1].gap == pytest.approx(abs(50.0 - 25.0) / max(abs(25.0), 1.0))


def test_minimize_lower_bound_is_clamped_non_decreasing() -> None:
    tracker = BoundTracker(ObjectiveSense.MINIMIZE)
    # A raw lower bound that loosens (40 → 30) must be clamped up to the tightest proven (40).
    raw = [
        _inc(100.0, 10.0, AllocationStatus.FEASIBLE),
        _inc(80.0, 40.0, AllocationStatus.FEASIBLE),
        _inc(70.0, 30.0, AllocationStatus.OPTIMAL),
    ]
    bounds = _bounds(tracker.track(i) for i in raw)
    assert bounds == [10.0, 40.0, 40.0]
    assert all(b >= a for a, b in pairwise(bounds))  # non-decreasing


def test_feasibility_only_incumbent_passes_through_without_a_bound() -> None:
    tracker = BoundTracker(ObjectiveSense.MAXIMIZE)
    tracked = tracker.track(_inc(3.0, None, AllocationStatus.FEASIBLE))
    assert tracked.bound is None and tracked.gap is None


def test_stream_incumbents_exposes_monotone_bounds_and_passes_hints() -> None:
    scripted = [
        _inc(10.0, 100.0, AllocationStatus.FEASIBLE),
        _inc(20.0, 60.0, AllocationStatus.FEASIBLE),
        _inc(25.0, 70.0, AllocationStatus.OPTIMAL),  # raw bound loosens; must clamp
    ]
    solver = _ScriptedSolver(scripted)
    ir = compile_request(anchor_request())
    out = list(
        stream_incumbents(
            solver, ir, SolveBudget(), ObjectiveSense.MAXIMIZE, hints={"assign::t::a": 1.0}
        )
    )
    bounds = [i.bound for i in out]
    assert bounds == [100.0, 60.0, 60.0]
    assert solver.seen_hints == {"assign::t::a": 1.0}


# --- finalize_status: honest anytime deadline outcomes ---------------------------------------


def test_finalize_status_maps_unknown_to_timeout() -> None:
    assert finalize_status(AllocationStatus.OPTIMAL) is AllocationStatus.OPTIMAL
    assert finalize_status(AllocationStatus.FEASIBLE) is AllocationStatus.FEASIBLE
    assert finalize_status(AllocationStatus.INFEASIBLE) is AllocationStatus.INFEASIBLE
    assert finalize_status(AllocationStatus.UNKNOWN) is AllocationStatus.TIMEOUT
    assert finalize_status(AllocationStatus.TIMEOUT) is AllocationStatus.TIMEOUT


# --- solve_anytime: the end-to-end anytime sub-interface -------------------------------------


def test_solve_anytime_yields_verified_feasible_plans_ending_in_the_terminal() -> None:
    request = anchor_request()
    ir = compile_request(request)
    stream = list(AllocationPlanner(backend="cp-sat").solve_anytime(request))
    assert stream  # at least the terminal incumbent
    for alloc in stream:
        assert alloc.status in (AllocationStatus.OPTIMAL, AllocationStatus.FEASIBLE)
        assert verify_feasible(alloc, ir)  # every incumbent is independently re-checked
    # The terminal plan carries the full explanation; solve() returns exactly it.
    terminal = stream[-1]
    assert terminal.status is AllocationStatus.OPTIMAL
    assert terminal.binding_constraints
    assert terminal.objective_decomposition is not None
    # solve() returns exactly the terminal item (bar the wall-clock telemetry, which is not part
    # of the reproducible artifact — test_cpsat_determinism.py pins that boundary).
    one_shot = AllocationPlanner(backend="cp-sat").solve(request)
    exclude = {"provenance": {"budget_consumed_s"}}
    assert one_shot.model_dump(exclude=exclude) == terminal.model_dump(exclude=exclude)


def test_intermediate_incumbent_defers_the_expensive_explanation() -> None:
    # A non-terminal incumbent is superseded before the deadline, so its explanation is deferred.
    planner = AllocationPlanner(backend="cp-sat")
    request = anchor_request()
    ir = compile_request(request)
    prov = planner._provenance(request, ir)
    # Rebuild a feasible incumbent from a real terminal plan and map it as a non-terminal one.
    terminal = list(planner._stream_backend(request, None, None, None, None))[-1]
    synthetic = Incumbent(
        objective=terminal.realized_objective or 0.0,
        bound=None,
        gap=terminal.optimality_gap,
        plan=terminal.plan or [],
        status=AllocationStatus.FEASIBLE,
    )
    intermediate = planner._incumbent_allocation(request, ir, None, prov, synthetic, terminal=False)
    assert intermediate.status is AllocationStatus.FEASIBLE
    assert intermediate.binding_constraints == []
    assert intermediate.objective_decomposition is None
    assert verify_feasible(intermediate, ir)


# --- warm start: the online re-solve seam ----------------------------------------------------


def test_warm_start_from_a_prior_plan_reproduces_a_feasible_plan() -> None:
    request = anchor_request()
    first = AllocationPlanner(backend="cp-sat").solve(request)
    hints = hints_from(first)
    # Hints key the IR assignment + start variables of the prior plan.
    assert hints["assign::prospect-crater-a::prospector-rover-1"] == 1.0
    assert "start::prospect-crater-a" in hints
    warm = list(AllocationPlanner(backend="cp-sat").solve_anytime(request, hints=hints))[-1]
    assert warm.status is AllocationStatus.OPTIMAL
    assert warm.realized_objective == pytest.approx(first.realized_objective)


def test_hints_from_accepts_a_bare_plan_and_an_empty_source() -> None:
    request = anchor_request()
    plan = AllocationPlanner(backend="cp-sat").solve(request).plan
    assert plan is not None
    assert hints_from(plan) == hints_from(AllocationPlanner(backend="cp-sat").solve(request))
    assert hints_from([]) == {}


# --- property: bound monotonicity across a real solve ----------------------------------------


@st.composite
def feasible_requests(draw: st.DrawFn) -> AllocationRequest:
    from astro_mine.core.messages.enums import TaskKind

    n_tasks = draw(st.integers(min_value=1, max_value=4))
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
        request_id="p",
        tasks=tasks,
        assets=[AssetRef(asset_id=f"a{k}") for k in range(draw(st.integers(1, 3)))],
        objective=Objective(sense=draw(st.sampled_from(list(ObjectiveSense)))),
        budget=SolveBudget(deterministic=True, seed=draw(st.integers(0, 2**31 - 1))),
    )


@settings(max_examples=40, deadline=None)
@given(request=feasible_requests())
def test_streamed_bounds_are_monotone(request: AllocationRequest) -> None:
    ir = compile_request(request)
    from astro_mine.allocate.solvers.registry import CPSAT_BACKEND, resolve_solver

    solver = resolve_solver(CPSAT_BACKEND, task_kinds={t.task_id: t.kind for t in request.tasks})
    streamed = stream_incumbents(solver, ir, request.budget, ir.objective_sense)
    # A feasibility-only incumbent carries no bound; monotonicity is a claim about the proven ones.
    proven = [i.bound for i in streamed if i.bound is not None]
    if request.objective.sense is ObjectiveSense.MAXIMIZE:
        assert all(b <= a for a, b in pairwise(proven))  # non-increasing
    else:
        assert all(b >= a for a, b in pairwise(proven))  # non-decreasing
