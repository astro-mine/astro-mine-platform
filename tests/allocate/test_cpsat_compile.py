"""Unit tests for the pure IR → CP-SAT lowering (RM-P1-ALLOC-02, ``model/compile/cpsat.py``).

Exercises the encoding directly — variable domains, the ``AddExactlyOne`` cover, generic linear
constraints, the ``NO_OVERLAP`` / ``CUMULATIVE`` interval encodings with assignment↔interval
channeling, and the objective direction — by lowering hand-built IRs and solving the resulting
CP-SAT model, decoupled from the search driver (:mod:`astro_mine.allocate.solvers.cpsat`).
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable
from typing import Any

import pytest

from astro_mine.allocate.enums import (
    ConstraintKind,
    ConstraintSense,
    ObjectiveSense,
    VariableKind,
    VariableSemantic,
)
from astro_mine.allocate.model.ir.model import (
    AllocationIR,
    Constraint,
    ConstraintTerm,
    DecisionVariable,
    ObjectiveTerm,
)

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ortools") is None, reason="OR-Tools (cp-sat backend) not installed"
)

cp_model = pytest.importorskip("ortools.sat.python.cp_model")

from astro_mine.allocate.model.compile.cpsat import (  # noqa: E402 - after importorskip
    CpSatModel,
    _time_var_bounds,
    cumulative_constraint_id,
    lower_to_cpsat,
    no_overlap_constraint_id,
)


def _assign(task: str, asset: str) -> DecisionVariable:
    return DecisionVariable(
        id=f"assign::{task}::{asset}",
        kind=VariableKind.BINARY,
        lower=0.0,
        upper=1.0,
        semantic=VariableSemantic.ASSIGNMENT,
        task_ref=task,
        asset_ref=asset,
    )


def _start(task: str, lo: float = 0.0, hi: float = 100.0) -> DecisionVariable:
    return DecisionVariable(
        id=f"start::{task}",
        kind=VariableKind.CONTINUOUS,
        lower=lo,
        upper=hi,
        semantic=VariableSemantic.START_TIME,
        task_ref=task,
    )


def _cover(task: str, assets: list[str]) -> Constraint:
    return Constraint(
        id=f"cover::{task}",
        kind=ConstraintKind.ASSIGNMENT_COVER,
        terms=[ConstraintTerm(var_ref=f"assign::{task}::{a}", coefficient=1.0) for a in assets],
        sense=ConstraintSense.EQ,
        rhs=1.0,
    )


def _ir(
    variables: Iterable[DecisionVariable],
    constraints: Iterable[Constraint],
    objective_terms: Iterable[ObjectiveTerm] = (),
    sense: ObjectiveSense = ObjectiveSense.MAXIMIZE,
) -> AllocationIR:
    return AllocationIR(
        variables=list(variables),
        constraints=list(constraints),
        objective_terms=list(objective_terms),
        objective_sense=sense,
    )


def _solve(compiled: CpSatModel) -> tuple[Any, Any]:
    """Solve a lowered model. ``cp_model`` arrives via ``importorskip``, so the solver and its
    status are untyped here; the status is an ``int`` enum value at runtime."""
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 7
    status = solver.Solve(compiled.model)
    return status, solver


# --- variables ---------------------------------------------------------------------


def test_continuous_bounds_are_conservatively_rounded() -> None:
    lo_hi = _time_var_bounds(_start("t", lo=10.5, hi=20.5), 1)
    assert lo_hi == (11, 20)  # lower ceils, upper floors — a solved start always satisfies the IR


def test_open_horizon_is_clamped_to_a_finite_grid() -> None:
    lo, hi = _time_var_bounds(_start("t", lo=0.0, hi=1.0e12), 1)
    assert (lo, hi) == (0, 1_000_000_000)


def test_empty_subgrid_interval_degenerates_to_a_point() -> None:
    lo, hi = _time_var_bounds(_start("t", lo=10.4, hi=10.6), 1)  # ceil 11 > floor 10
    assert lo == hi == 11


def test_binary_and_continuous_variables_are_created() -> None:
    compiled = lower_to_cpsat(_ir([_assign("t", "a"), _start("t")], [_cover("t", ["a"])]))
    assert set(compiled.variables) == {"assign::t::a", "start::t"}
    assert compiled.has_objective is False


def test_integer_variables_use_their_bounds_or_a_default_domain() -> None:
    bounded = DecisionVariable(
        id="n",
        kind=VariableKind.INTEGER,
        lower=2.0,
        upper=5.0,
        semantic=VariableSemantic.START_TIME,
        task_ref="n",
    )
    unbounded = DecisionVariable(
        id="m", kind=VariableKind.INTEGER, semantic=VariableSemantic.START_TIME, task_ref="m"
    )
    compiled = lower_to_cpsat(_ir([bounded, unbounded], []))
    assert set(compiled.variables) == {"n", "m"}
    status, solver = _solve(compiled)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    assert 2 <= solver.Value(compiled.variables["n"]) <= 5


# --- linear families ---------------------------------------------------------------


def test_assignment_cover_is_exactly_one() -> None:
    ir = _ir([_assign("t", "a"), _assign("t", "b"), _start("t")], [_cover("t", ["a", "b"])])
    compiled = lower_to_cpsat(ir)
    status, solver = _solve(compiled)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    chosen = solver.Value(compiled.variables["assign::t::a"]) + solver.Value(
        compiled.variables["assign::t::b"]
    )
    assert chosen == 1


def test_time_window_and_precedence_are_enforced() -> None:
    variables = [_assign("a", "r"), _assign("b", "r"), _start("a"), _start("b")]
    constraints = [
        _cover("a", ["r"]),
        _cover("b", ["r"]),
        Constraint(
            id="twin_lo::a",
            kind=ConstraintKind.TIME_WINDOW,
            terms=[ConstraintTerm(var_ref="start::a", coefficient=1.0)],
            sense=ConstraintSense.GE,
            rhs=30.0,
        ),
        Constraint(
            id="prec::a->b",
            kind=ConstraintKind.PRECEDENCE,
            terms=[
                ConstraintTerm(var_ref="start::b", coefficient=1.0),
                ConstraintTerm(var_ref="start::a", coefficient=-1.0),
            ],
            sense=ConstraintSense.GE,
            rhs=0.0,
        ),
    ]
    compiled = lower_to_cpsat(_ir(variables, constraints))
    status, solver = _solve(compiled)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    a = solver.Value(compiled.variables["start::a"])
    b = solver.Value(compiled.variables["start::b"])
    assert a >= 30
    assert b >= a


def test_linear_le_budget_is_enforced() -> None:
    # A single-asset energy budget: each of the two tasks costs 8, capacity 10 admits only one.
    variables = [_assign("a", "r"), _assign("b", "r"), _start("a"), _start("b")]
    constraints = [
        Constraint(
            id="power::r",
            kind=ConstraintKind.LINEAR,
            terms=[
                ConstraintTerm(var_ref="assign::a::r", coefficient=8.0),
                ConstraintTerm(var_ref="assign::b::r", coefficient=8.0),
            ],
            sense=ConstraintSense.LE,
            rhs=10.0,
        ),
    ]
    # Maximize assignments to push against the budget.
    obj = [
        ObjectiveTerm(id="o::a", var_ref="assign::a::r", coefficient=1.0),
        ObjectiveTerm(id="o::b", var_ref="assign::b::r", coefficient=1.0),
    ]
    compiled = lower_to_cpsat(_ir(variables, constraints, obj))
    status, solver = _solve(compiled)
    assert status == cp_model.OPTIMAL
    total = solver.Value(compiled.variables["assign::a::r"]) + solver.Value(
        compiled.variables["assign::b::r"]
    )
    assert total == 1  # the 10-unit budget admits only one of the two 8-unit tasks


# --- scheduling globals (no-overlap / cumulative / channeling) ---------------------


def _no_overlap_ir(resource: str, size: float) -> AllocationIR:
    variables = [_assign("a", resource), _assign("b", resource), _start("a"), _start("b")]
    constraints = [
        _cover("a", [resource]),
        _cover("b", [resource]),
        Constraint(
            id=no_overlap_constraint_id(resource),
            kind=ConstraintKind.NO_OVERLAP,
            terms=[
                ConstraintTerm(var_ref="start::a", coefficient=size),
                ConstraintTerm(var_ref="start::b", coefficient=size),
            ],
            sense=ConstraintSense.LE,
            rhs=0.0,
        ),
    ]
    return _ir(variables, constraints)


def test_no_overlap_serializes_channelled_intervals() -> None:
    compiled = lower_to_cpsat(_no_overlap_ir("r", size=10.0))
    status, solver = _solve(compiled)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    a = solver.Value(compiled.variables["start::a"])
    b = solver.Value(compiled.variables["start::b"])
    # Both assigned to r ⇒ their size-10 intervals must not overlap.
    assert not (a < b + 10 and b < a + 10)


def test_a_fractional_interval_reserves_a_whole_grid_second() -> None:
    """A float duration rounds **up** onto the integer grid — never down (issue #22).

    CP-SAT schedules on integer seconds while the IR's durations are floats, so a 10.2 s task must
    reserve 11 grid seconds. Truncating it to 10 would let CP-SAT butt the next task against an end
    the *float* model has not reached yet: a schedule that looks conflict-free on the grid and
    double-books the asset by 0.2 s against :func:`~astro_mine.allocate.verify_feasible`, which
    re-checks it in floats. Nothing rewarded tight packing until the per-pair cost objective landed,
    which is why this went unnoticed — the very first cost-driven solve tripped it.
    """
    compiled = lower_to_cpsat(_no_overlap_ir("r", size=10.2))
    status, solver = _solve(compiled)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)

    a = solver.Value(compiled.variables["start::a"])
    b = solver.Value(compiled.variables["start::b"])
    # The gap between the two starts must cover the *float* duration, not its truncation.
    assert abs(a - b) >= 10.2


def test_no_overlap_is_mandatory_when_no_assignment_channels_it() -> None:
    # No assign::*::ghost variables exist, so the intervals are mandatory (always present).
    variables = [_start("a"), _start("b")]
    constraints = [
        Constraint(
            id=no_overlap_constraint_id("ghost"),
            kind=ConstraintKind.NO_OVERLAP,
            terms=[
                ConstraintTerm(var_ref="start::a", coefficient=10.0),
                ConstraintTerm(var_ref="start::b", coefficient=10.0),
            ],
            sense=ConstraintSense.LE,
            rhs=0.0,
        ),
    ]
    compiled = lower_to_cpsat(_ir(variables, constraints))
    status, solver = _solve(compiled)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    a = solver.Value(compiled.variables["start::a"])
    b = solver.Value(compiled.variables["start::b"])
    assert not (a < b + 10 and b < a + 10)


def test_cumulative_capacity_one_serializes() -> None:
    variables = [_assign("a", "r"), _assign("b", "r"), _start("a"), _start("b")]
    constraints = [
        _cover("a", ["r"]),
        _cover("b", ["r"]),
        Constraint(
            id=cumulative_constraint_id("r"),
            kind=ConstraintKind.CUMULATIVE,
            terms=[
                ConstraintTerm(var_ref="start::a", coefficient=10.0),
                ConstraintTerm(var_ref="start::b", coefficient=10.0),
            ],
            sense=ConstraintSense.LE,
            rhs=1.0,  # unit capacity ⇒ at most one interval at a time (like no-overlap)
        ),
    ]
    compiled = lower_to_cpsat(_ir(variables, constraints))
    status, solver = _solve(compiled)
    assert status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    a = solver.Value(compiled.variables["start::a"])
    b = solver.Value(compiled.variables["start::b"])
    assert not (a < b + 10 and b < a + 10)


# --- objective ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sense", "expected_asset"),
    [(ObjectiveSense.MAXIMIZE, "hi"), (ObjectiveSense.MINIMIZE, "lo")],
)
def test_objective_direction(sense: ObjectiveSense, expected_asset: str) -> None:
    # One task, choose exactly one of two assets; the hi asset scores 10, the lo asset 1.
    variables = [_assign("t", "hi"), _assign("t", "lo"), _start("t")]
    constraints = [_cover("t", ["hi", "lo"])]
    obj = [
        ObjectiveTerm(id="o::hi", var_ref="assign::t::hi", coefficient=10.0),
        ObjectiveTerm(id="o::lo", var_ref="assign::t::lo", coefficient=1.0),
    ]
    compiled = lower_to_cpsat(_ir(variables, constraints, obj, sense))
    assert compiled.has_objective is True
    status, solver = _solve(compiled)
    assert status == cp_model.OPTIMAL
    assert solver.Value(compiled.variables[f"assign::t::{expected_asset}"]) == 1


def test_interval_variable_kind_is_unsupported() -> None:
    bogus = DecisionVariable(
        id="iv::t", kind=VariableKind.INTERVAL, semantic=VariableSemantic.START_TIME, task_ref="t"
    )
    with pytest.raises(NotImplementedError):
        lower_to_cpsat(_ir([bogus], []))
