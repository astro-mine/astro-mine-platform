"""The per-pair cost objective — the term that makes one feasible plan better than another (#22).

Value is a per-*task* quantity and the assignment cover is exactly-one, so an objective built from
value alone scores **every** feasible plan identically: the solver has nothing to optimize and the
optimality gap it reports is zero by construction, not by quality. These tests pin the family that
fixes that — the SI cost of *this* asset doing *this* task, priced into value units and subtracted —
and, crucially, prove the gap gate it enables can actually **fail**.

The fixture is deliberately tiny and symmetric: two interchangeable rovers, two tasks, and a cost
table in which each rover is the cheap choice for exactly one of them. The optimal plan is therefore
obvious by inspection (each task to its cheap rover), which makes "a worse plan scores worse" a
statement about the objective rather than about the solver.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from astro_mine.allocate import (
    AllocationPlanner,
    AllocationRequest,
    AllocationStatus,
    AssetRef,
    AssetSchedule,
    ConstraintConfig,
    CostPolicy,
    Objective,
    ObjectiveSense,
    ScheduledTask,
    SolveBudget,
    Task,
    TrivialStubSolver,
    ValueEstimate,
    compile_request,
    compile_with_constraints,
    decompose_objective,
    verify_feasible,
)
from astro_mine.allocate.constraints.cost_objective import refine_cost_objective
from astro_mine.core.messages.enums import TaskKind
from astro_mine.core.sadf import CapabilityTag
from tests.allocate.constraint_factories import context, cost_table

#: Both tasks are worth the same, so *all* of the difference between two plans is what they cost.
_VALUE = 10.0
_DURATION_S = 600.0

#: Each rover is the cheap one for exactly one task. Optimal: dig-a → rover-a, dig-b → rover-b.
_CHEAP_J = 1.0e5
_DEAR_J = 3.0e5

#: Value units per joule. Sized so a task's cost (1-3 units) is a real bite out of its value (10).
_PRICE = 1.0e-5

_COSTS = cost_table(
    {
        ("dig-a", "rover-a"): (_DURATION_S, _CHEAP_J),
        ("dig-a", "rover-b"): (_DURATION_S, _DEAR_J),
        ("dig-b", "rover-a"): (_DURATION_S, _DEAR_J),
        ("dig-b", "rover-b"): (_DURATION_S, _CHEAP_J),
    }
)


def _request(
    *,
    sense: ObjectiveSense = ObjectiveSense.MAXIMIZE,
    weights: dict[str, float] | None = None,
) -> AllocationRequest:
    tasks = [
        Task(
            task_id=task_id,
            kind=TaskKind.EXCAVATE,
            required_capabilities=[CapabilityTag.MOBILITY_WHEELED],
            duration_s=_DURATION_S,
            value=ValueEstimate(mean=_VALUE, variance=1.0),
        )
        for task_id in ("dig-a", "dig-b")
    ]
    assets = [
        AssetRef(
            asset_id=asset_id,
            capability_tags=[CapabilityTag.MOBILITY_WHEELED],
            budgets={"energy_j": 1.0e7},
        )
        for asset_id in ("rover-a", "rover-b")
    ]
    return AllocationRequest(
        request_id="cost-objective-fixture",
        tasks=tasks,
        assets=assets,
        objective=Objective(sense=sense, weights=weights or {}),
        budget=SolveBudget(seed=7, deterministic=True),
    )


_POLICY = CostPolicy(energy_price_per_j=_PRICE)


def _config(cost: CostPolicy | None = _POLICY) -> ConstraintConfig:
    return ConstraintConfig(cost=cost)


def _solve(request: AllocationRequest, config: ConstraintConfig):  # type: ignore[no-untyped-def]
    return AllocationPlanner(backend="cp-sat").solve(
        request, context=context(), config=config, costs=_COSTS
    )


def _assignment(plan: list[AssetSchedule]) -> dict[str, str]:
    return {st.task_id: schedule.asset_id for schedule in plan for st in schedule.tasks}


def _plan(assignment: dict[str, str]) -> list[AssetSchedule]:
    """A hand-built plan placing every task at t=0 on its assigned asset (windows are open here)."""
    by_asset: dict[str, list[ScheduledTask]] = {}
    for task_id, asset_id in sorted(assignment.items()):
        by_asset.setdefault(asset_id, []).append(
            ScheduledTask(task_id=task_id, kind=TaskKind.EXCAVATE, start_s=0.0, end_s=_DURATION_S)
        )
    return [
        AssetSchedule(asset_id=asset_id, tasks=tasks)
        for asset_id, tasks in sorted(by_asset.items())
    ]


def _score(plan: list[AssetSchedule], ir) -> float:  # type: ignore[no-untyped-def]
    """What a plan is worth under an IR — the identity ``verify_feasible`` re-derives."""
    return decompose_objective(plan, ir).total


# --- the family exists and is priced from the cached cost table -------------------------------


def test_each_pair_is_charged_the_energy_that_pair_actually_costs() -> None:
    comp = compile_with_constraints(_request(), context(), config=_config(), costs=_COSTS)

    assert comp.report.pair_costs == {
        ("dig-a", "rover-a"): _PRICE * _CHEAP_J,
        ("dig-a", "rover-b"): _PRICE * _DEAR_J,
        ("dig-b", "rover-a"): _PRICE * _DEAR_J,
        ("dig-b", "rover-b"): _PRICE * _CHEAP_J,
    }
    # A cost is *subtracted* from a MAXIMIZE objective: the coefficient is negative.
    costs = {t.id: t.coefficient for t in comp.ir.objective_terms if t.id.startswith("cost::")}
    assert costs["cost::assign::dig-a::rover-a"] == pytest.approx(-_PRICE * _CHEAP_J)
    assert costs["cost::assign::dig-a::rover-b"] == pytest.approx(-_PRICE * _DEAR_J)


def test_time_can_be_priced_instead_of_energy() -> None:
    """An asset's *time* is a cost too: the opportunity cost of the hours it spends."""
    comp = compile_with_constraints(
        _request(),
        context(),
        config=_config(CostPolicy(time_price_per_s=0.001)),
        costs=_COSTS,
    )
    # Every pair takes the same 600 s here, so time alone prices them all identically...
    assert len(comp.report.pair_costs) == 4
    assert all(cost == pytest.approx(0.6) for cost in comp.report.pair_costs.values())


def test_a_cost_policy_that_prices_nothing_is_rejected() -> None:
    # ...and a policy that prices *neither* would add a uniformly-zero family: a silent no-op that
    # looks like the cost objective is on when it is not. Say so instead.
    with pytest.raises(ValidationError, match="prices neither energy nor time"):
        CostPolicy()


def test_without_a_cost_policy_the_objective_is_untouched() -> None:
    """The family is opt-in: no policy ⇒ not one term, not one metadata key, nothing."""
    comp = compile_with_constraints(_request(), context(), config=_config(None), costs=_COSTS)

    assert not [t for t in comp.ir.objective_terms if t.id.startswith("cost::")]
    assert not [k for k in comp.ir.metadata if k.startswith("pair_cost::")]
    families = {v for k, v in comp.ir.metadata.items() if k.startswith("objective_family")}
    assert "cost" not in families
    assert comp.report.pair_costs == {}
    # And the objective is exactly the skeleton's per-task value — the degenerate one that scores
    # every feasible plan identically (which is *why* the cost family exists).
    assert [t.coefficient for t in comp.ir.objective_terms] == [_VALUE] * 4


# --- acceptance: the objective distinguishes between feasible plans ----------------------------


def test_the_objective_distinguishes_between_feasible_plans() -> None:
    """A deliberately worse assignment of the *same* tasks scores measurably worse (issue #22 AC1).

    Both plans are feasible, cover every task exactly once, and finish at the same time. They
    differ only in *who does what* — which, before the per-pair family, the objective could not
    see at all.
    """
    comp = compile_with_constraints(_request(), context(), config=_config(), costs=_COSTS)
    ir = comp.ir

    good = _plan({"dig-a": "rover-a", "dig-b": "rover-b"})  # each task to its cheap rover
    swapped = _plan({"dig-a": "rover-b", "dig-b": "rover-a"})  # each task to its dear one

    # Both are genuinely feasible — the comparison is about quality, not validity.
    for plan in (good, swapped):
        allocation = _solve(_request(), _config()).model_copy(
            update={"plan": plan, "realized_objective": _score(plan, ir)}
        )
        assert verify_feasible(allocation, ir)

    assert _score(good, ir) == pytest.approx(2 * _VALUE - 2 * _PRICE * _CHEAP_J)
    assert _score(swapped, ir) == pytest.approx(2 * _VALUE - 2 * _PRICE * _DEAR_J)
    assert _score(swapped, ir) < _score(good, ir)

    # The whole difference is the cost family — the two plans earn identical value (the cover is
    # exactly-one), which is exactly why value alone could never tell them apart.
    good_families = {c.family: c.value for c in decompose_objective(good, ir).contributions}
    swapped_families = {c.family: c.value for c in decompose_objective(swapped, ir).contributions}
    assert good_families["roi"] == pytest.approx(swapped_families["roi"])
    assert swapped_families["cost"] < good_families["cost"] < 0.0


def test_the_solver_takes_the_cheaper_assignment() -> None:
    allocation = _solve(_request(), _config())

    assert allocation.status is AllocationStatus.OPTIMAL
    assert allocation.plan is not None
    assert _assignment(allocation.plan) == {"dig-a": "rover-a", "dig-b": "rover-b"}
    assert allocation.realized_objective == pytest.approx(2 * _VALUE - 2 * _PRICE * _CHEAP_J)


def test_the_cost_family_is_reported_in_the_objective_decomposition() -> None:
    allocation = _solve(_request(), _config())
    assert allocation.objective_decomposition is not None
    families = {c.family: c.value for c in allocation.objective_decomposition.contributions}

    assert families["roi"] == pytest.approx(2 * _VALUE)
    assert families["cost"] == pytest.approx(-2 * _PRICE * _CHEAP_J)
    # `total` is the plan's realized objective by construction (RM-P1-ALLOC-06).
    assert allocation.objective_decomposition.total == pytest.approx(allocation.realized_objective)


# --- acceptance: the optimality-gap gate now has teeth -----------------------------------------


def test_the_gap_gate_fails_a_knowingly_suboptimal_solver() -> None:
    """The gap assertion the scale benchmark makes **fails** on a cost-blind solver (issue #22 AC2).

    ``TrivialStubSolver`` is a real, shipped backend and a knowingly-suboptimal one: it takes the
    first eligible asset for each task and never looks at what that asset costs. Here that lands
    both tasks on ``rover-a`` — cheap for one, dear for the other — a perfectly *feasible* plan
    that is simply worse. Scored against CP-SAT's proven optimum with the same formula the anytime
    tracker uses, it blows straight through a 5% gap bound. Before the per-pair objective, its gap
    would have been **zero**: it would have sailed through the gate untouched.
    """
    max_gap = 0.05  # tests.test_scale_benchmark.MAX_GAP — the bound the benchmark enforces
    request = _request()
    comp = compile_with_constraints(request, context(), config=_config(), costs=_COSTS)
    ir = comp.ir

    optimum = _solve(request, _config())
    assert optimum.status is AllocationStatus.OPTIMAL
    assert optimum.realized_objective is not None

    greedy = list(
        TrivialStubSolver(
            task_kinds={t.task_id: t.kind for t in request.tasks},
            durations=dict(comp.report.durations),
        ).solve(ir, request.budget)
    )[-1]
    assert greedy.status is AllocationStatus.FEASIBLE
    assert _assignment(greedy.plan) == {"dig-a": "rover-a", "dig-b": "rover-a"}

    # A proven optimum *is* the dual bound, so this is the gap the gate would compute for the
    # greedy's plan (BoundTracker: |bound - objective| / max(|objective|, 1)).
    greedy_objective = _score(greedy.plan, ir)
    gap = abs(optimum.realized_objective - greedy_objective) / max(abs(greedy_objective), 1.0)

    assert greedy_objective < optimum.realized_objective
    assert gap > max_gap, (
        f"the cost-blind greedy scored {greedy_objective} against the optimum "
        f"{optimum.realized_objective} — a gap of {gap:.2%}, which a {max_gap:.0%} gate would "
        "accept. The gap gate has no teeth."
    )


# --- edges ------------------------------------------------------------------------------------


def test_a_minimize_objective_is_still_penalized_by_cost() -> None:
    """A cost is a cost whichever way the objective is optimized — it never becomes a *reward*."""
    request = _request(sense=ObjectiveSense.MINIMIZE)
    comp = compile_with_constraints(request, context(), config=_config(), costs=_COSTS)

    costs = {t.id: t.coefficient for t in comp.ir.objective_terms if t.id.startswith("cost::")}
    # Minimizing, so the cheap pair must *add* less than the dear one (a negative coefficient
    # here would make the solver seek out the most expensive assignment it could find).
    assert costs["cost::assign::dig-a::rover-a"] == pytest.approx(+_PRICE * _CHEAP_J)
    assert costs["cost::assign::dig-a::rover-b"] == pytest.approx(+_PRICE * _DEAR_J)


def test_the_cost_weight_scales_the_family() -> None:
    weighted = compile_with_constraints(
        _request(weights={"cost": 2.0}), context(), config=_config(), costs=_COSTS
    )
    costs = {t.id: t.coefficient for t in weighted.ir.objective_terms if t.id.startswith("cost::")}
    assert costs["cost::assign::dig-a::rover-a"] == pytest.approx(-2.0 * _PRICE * _CHEAP_J)


def test_a_kept_out_pair_is_never_charged() -> None:
    """A forbidden pair's variable is pinned to 0, so pricing it would only add a dead term."""
    request = _request()
    base_ir = compile_request(request)
    result = refine_cost_objective(
        base_ir,
        (),
        config=_config(),
        durations={("dig-a", "rover-a"): _DURATION_S},
        energy_costs={("dig-a", "rover-a"): _CHEAP_J, ("dig-a", "rover-b"): _DEAR_J},
        weights={},
        forbidden=frozenset({("dig-a", "rover-b")}),
    )

    assert ("dig-a", "rover-b") not in result.pair_costs
    assert not [t for t in result.objective_terms if t.var_ref == "assign::dig-a::rover-b"]
