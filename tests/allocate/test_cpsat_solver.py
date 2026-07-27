"""Driver-level CP-SAT behaviour: hints, budget mapping, and the incumbent stream (ALLOC-02).

Exercises the :class:`~astro_mine.allocate.solvers.cpsat.CpSatSolver` seams beyond the end-to-end
planner path: the warm-start hint hook (accepted and verified, never trusted), the non-deterministic
budget → worker/deadline mapping, a pure-feasibility (no-objective) model, and the streamed
incumbent trajectory.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping

import pytest

from astro_mine.allocate import (
    AllocationIR,
    AllocationStatus,
    Constraint,
    ConstraintTerm,
    DecisionVariable,
    Incumbent,
    ObjectiveSense,
    ObjectiveTerm,
    SolveBudget,
    compile_request,
    verify_feasible,
)
from astro_mine.allocate.enums import (
    ConstraintKind,
    ConstraintSense,
    VariableKind,
    VariableSemantic,
)
from astro_mine.core.messages.enums import TaskKind
from tests.allocate.factories import anchor_request

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ortools") is None, reason="OR-Tools (cp-sat backend) not installed"
)

from astro_mine.allocate.solvers.cpsat import CpSatSolver  # noqa: E402 - after importorskip guard


def _anchor_kinds() -> dict[str, TaskKind]:
    return {t.task_id: t.kind for t in anchor_request().tasks}


def _last(
    solver: CpSatSolver,
    ir: AllocationIR,
    budget: SolveBudget,
    *,
    hints: Mapping[str, float] | None = None,
) -> Incumbent:
    return list(solver.solve(ir, budget, hints=hints))[-1]


def test_warm_start_hints_are_accepted_and_verified() -> None:
    request = anchor_request()
    ir = compile_request(request)
    solver = CpSatSolver(task_kinds=_anchor_kinds())
    # Hints for an assignment var (binary), a start var (continuous, scaled), and an unknown var.
    hints = {
        "assign::prospect-crater-a::prospector-rover-1": 1.0,
        "start::prospect-crater-a": 0.0,
        "assign::ghost::nobody": 1.0,  # a variable this IR does not carry — silently ignored
    }
    terminal = _last(solver, ir, request.budget, hints=hints)
    assert terminal.is_feasible
    # A verified warm start still yields a feasible, honestly-scored plan.
    from astro_mine.allocate import Allocation, AllocationProvenance

    alloc = Allocation(
        status=terminal.status,
        plan=terminal.plan,
        realized_objective=terminal.objective,
        provenance=AllocationProvenance(ir_version="0.1.0", backend="cp-sat"),
    )
    assert verify_feasible(alloc, ir)


def test_non_deterministic_budget_sets_workers_and_wall_clock() -> None:
    request = anchor_request()
    ir = compile_request(request)
    budget = SolveBudget(deterministic=False, workers=2, wall_clock_deadline_s=5.0, seed=7)
    terminal = _last(CpSatSolver(task_kinds=_anchor_kinds()), ir, budget)
    assert terminal.status in (AllocationStatus.OPTIMAL, AllocationStatus.FEASIBLE)
    assert terminal.is_feasible


def test_non_deterministic_without_workers_or_deadline() -> None:
    request = anchor_request()
    ir = compile_request(request)
    budget = SolveBudget(deterministic=False)  # no workers, no deadline — CP-SAT defaults
    terminal = _last(CpSatSolver(task_kinds=_anchor_kinds()), ir, budget)
    assert terminal.is_feasible


def test_feasibility_only_model_has_no_bound_or_gap() -> None:
    # A no-objective IR is a pure feasibility model: an incumbent carries no optimality bound.
    variables = [
        DecisionVariable(
            id="assign::t::a",
            kind=VariableKind.BINARY,
            lower=0.0,
            upper=1.0,
            semantic=VariableSemantic.ASSIGNMENT,
            task_ref="t",
            asset_ref="a",
        ),
        DecisionVariable(
            id="start::t",
            kind=VariableKind.CONTINUOUS,
            lower=0.0,
            upper=100.0,
            semantic=VariableSemantic.START_TIME,
            task_ref="t",
        ),
    ]
    cover = Constraint(
        id="cover::t",
        kind=ConstraintKind.ASSIGNMENT_COVER,
        terms=[ConstraintTerm(var_ref="assign::t::a", coefficient=1.0)],
        sense=ConstraintSense.EQ,
        rhs=1.0,
    )
    ir = AllocationIR(
        variables=variables, constraints=[cover], objective_sense=ObjectiveSense.MAXIMIZE
    )
    terminal = _last(CpSatSolver(task_kinds={"t": TaskKind.PROSPECT}), ir, SolveBudget())
    assert terminal.is_feasible
    assert terminal.bound is None and terminal.gap is None


def test_incumbent_stream_is_nonempty_and_terminal_is_last() -> None:
    # A choose-the-best objective fires the solution callback, so the stream carries incumbents.
    variables = [
        DecisionVariable(
            id=f"assign::t::{a}",
            kind=VariableKind.BINARY,
            lower=0.0,
            upper=1.0,
            semantic=VariableSemantic.ASSIGNMENT,
            task_ref="t",
            asset_ref=a,
        )
        for a in ("a", "b")
    ]
    cover = Constraint(
        id="cover::t",
        kind=ConstraintKind.ASSIGNMENT_COVER,
        terms=[ConstraintTerm(var_ref=f"assign::t::{a}", coefficient=1.0) for a in ("a", "b")],
        sense=ConstraintSense.EQ,
        rhs=1.0,
    )
    obj = [
        ObjectiveTerm(id="o::a", var_ref="assign::t::a", coefficient=1.0),
        ObjectiveTerm(id="o::b", var_ref="assign::t::b", coefficient=5.0),
    ]
    ir = AllocationIR(
        variables=variables,
        constraints=[cover],
        objective_terms=obj,
        objective_sense=ObjectiveSense.MAXIMIZE,
    )
    stream = list(CpSatSolver(task_kinds={"t": TaskKind.PROSPECT}).solve(ir, SolveBudget(seed=7)))
    assert stream  # at least one incumbent
    assert stream[-1].status is AllocationStatus.OPTIMAL
    assert stream[-1].objective == pytest.approx(5.0)  # picks the higher-value asset
