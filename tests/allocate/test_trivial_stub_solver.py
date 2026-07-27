"""The no-dependency ``trivial-stub`` solver and the solver registry (RM-P1-ALLOC-02).

The stub proves the ``Solver`` strategy has a backend that always works on the local tier — the
fast-test / no-OR-Tools path — decoding a plan from the IR identically to CP-SAT and re-checkable by
:func:`~astro_mine.allocate.verify_feasible` (it is not trusted either). The registry is the
Allocate-internal name → factory map :class:`~astro_mine.allocate.AllocationPlanner` resolves a
backend id through.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping

import pytest

from astro_mine.allocate import (
    Allocation,
    AllocationIR,
    AllocationProvenance,
    AllocationRequest,
    AllocationStatus,
    Constraint,
    ConstraintConfig,
    ConstraintTerm,
    DecisionVariable,
    ObjectiveSense,
    SolveBudget,
    TrivialStubSolver,
    available_backends,
    compile_request,
    compile_with_constraints,
    known_backends,
    resolve_solver,
    verify_feasible,
)
from astro_mine.allocate.constraints.config import CommsPolicy
from astro_mine.allocate.enums import (
    ConstraintKind,
    ConstraintSense,
    VariableKind,
    VariableSemantic,
)
from astro_mine.allocate.solvers import CPSAT_BACKEND, TRIVIAL_STUB_BACKEND, Incumbent, Solver
from astro_mine.core.messages.enums import TaskKind
from astro_mine.core.units import J2000_EPOCH
from tests.allocate import constraint_factories as F
from tests.allocate.factories import anchor_request, infeasible_request

_HAS_ORTOOLS = importlib.util.find_spec("ortools") is not None
_B = SolveBudget()  # the stub ignores the budget; a default suffices for the custom-IR cases


def _task_kinds(request: AllocationRequest) -> dict[str, TaskKind]:
    return {t.task_id: t.kind for t in request.tasks}


def _stub_incumbent(
    ir: AllocationIR,
    request: AllocationRequest,
    durations: Mapping[tuple[str, str], float] | None = None,
) -> Incumbent:
    solver = TrivialStubSolver(task_kinds=_task_kinds(request), durations=durations or {})
    return next(reversed(list(solver.solve(ir, request.budget))))


def _as_allocation(inc: Incumbent) -> Allocation:
    return Allocation(
        status=inc.status,
        plan=inc.plan,
        realized_objective=inc.objective,
        provenance=AllocationProvenance(ir_version="0.1.0", backend="trivial-stub"),
    )


# --- the stub solver ---------------------------------------------------------------


def test_stub_conforms_to_the_solver_protocol() -> None:
    assert isinstance(TrivialStubSolver(task_kinds={}), Solver)


def test_stub_solves_the_skeleton_and_verifies() -> None:
    request = anchor_request()
    ir = compile_request(request)
    inc = _stub_incumbent(ir, request)
    assert inc.status is AllocationStatus.FEASIBLE
    assert inc.is_feasible
    assert inc.bound is None and inc.gap is None  # feasibility only — no optimality bound
    assert verify_feasible(_as_allocation(inc), ir)


def test_stub_respects_precedence_ordering() -> None:
    request = anchor_request()
    inc = _stub_incumbent(compile_request(request), request)
    starts = {st.task_id: st.start_s for a in inc.plan for st in a.tasks}
    # excavate follows prospect; haul follows excavate (the anchor precedence chain).
    assert starts["excavate-crater-a"] >= starts["prospect-crater-a"]
    assert starts["haul-to-plant"] >= starts["excavate-crater-a"]


def test_stub_honors_constrained_ir_windows_and_budgets() -> None:
    request = anchor_request()
    cfg = ConstraintConfig(
        comms=CommsPolicy(relay_required_task_ids=frozenset({"haul-to-plant"}), epoch0=J2000_EPOCH)
    )
    costs = F.cost_table(
        {
            ("prospect-crater-a", "prospector-rover-1"): (600.0, 1.0e6),
            ("excavate-crater-a", "excavator-1"): (900.0, 3.0e6),
            ("haul-to-plant", "hauler-1"): (600.0, 1.0e6),
            ("haul-to-plant", "prospector-rover-1"): (600.0, 1.0e6),
        }
    )
    ctx = F.context(
        world=F.FakeWorld(slope_deg=4.0),
        contacts=F.contact_plan(
            {"prospector-rover-1": (5000.0, 9000.0), "hauler-1": (5000.0, 9000.0)}
        ),
    )
    comp = compile_with_constraints(request, ctx, config=cfg, costs=costs)
    inc = _stub_incumbent(comp.ir, request, durations=dict(comp.report.durations))
    assert inc.status is AllocationStatus.FEASIBLE
    assert verify_feasible(_as_allocation(inc), comp.ir)
    haul = next(st for a in inc.plan for st in a.tasks if st.task_id == "haul-to-plant")
    assert haul.start_s >= 5000.0  # inside the comms contact window


def test_stub_surfaces_infeasibility_when_a_task_has_no_eligible_asset() -> None:
    request = infeasible_request()
    inc = _stub_incumbent(compile_request(request), request)
    assert inc.status is AllocationStatus.INFEASIBLE
    assert inc.is_feasible is False
    assert inc.plan == []


def test_stub_surfaces_infeasibility_on_an_energy_shortfall() -> None:
    request = anchor_request()
    costs = F.cost_table(
        {("excavate-crater-a", "excavator-1"): (900.0, 2.0e7)}
    )  # over 1.2e7 budget
    comp = compile_with_constraints(request, F.context(), costs=costs)
    inc = _stub_incumbent(comp.ir, request, durations=dict(comp.report.durations))
    assert inc.status is AllocationStatus.INFEASIBLE


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


def _start(task: str, lo: float, hi: float) -> DecisionVariable:
    return DecisionVariable(
        id=f"start::{task}",
        kind=VariableKind.CONTINUOUS,
        lower=lo,
        upper=hi,
        semantic=VariableSemantic.START_TIME,
        task_ref=task,
    )


def test_stub_routes_around_a_kept_out_asset() -> None:
    # Task t is eligible for a and b, but a keep-out pins assign(t, a) = 0 → the stub picks b.
    ir = AllocationIR(
        variables=[_assign("t", "a"), _assign("t", "b"), _start("t", 0.0, 100.0)],
        constraints=[
            Constraint(
                id="keepout::t::a",
                kind=ConstraintKind.LINEAR,
                terms=[ConstraintTerm(var_ref="assign::t::a", coefficient=1.0)],
                sense=ConstraintSense.EQ,
                rhs=0.0,
            )
        ],
        objective_sense=ObjectiveSense.MAXIMIZE,
    )
    inc = next(reversed(list(TrivialStubSolver(task_kinds={"t": TaskKind.PROSPECT}).solve(ir, _B))))
    assert inc.status is AllocationStatus.FEASIBLE
    assert {(s.task_id, a.asset_id) for a in inc.plan for s in a.tasks} == {("t", "b")}


def test_stub_reports_infeasible_when_the_window_lower_exceeds_the_upper() -> None:
    # A comms/time lower bound (>= 200) past the start variable's upper bound (100) → infeasible.
    ir = AllocationIR(
        variables=[_assign("t", "a"), _start("t", 0.0, 100.0)],
        constraints=[
            Constraint(
                id="cover::t",
                kind=ConstraintKind.ASSIGNMENT_COVER,
                terms=[ConstraintTerm(var_ref="assign::t::a", coefficient=1.0)],
                sense=ConstraintSense.EQ,
                rhs=1.0,
            ),
            Constraint(
                id="comms_lo::t",
                kind=ConstraintKind.TIME_WINDOW,
                terms=[ConstraintTerm(var_ref="start::t", coefficient=1.0)],
                sense=ConstraintSense.GE,
                rhs=200.0,
            ),
        ],
        objective_sense=ObjectiveSense.MAXIMIZE,
    )
    inc = next(reversed(list(TrivialStubSolver(task_kinds={"t": TaskKind.PROSPECT}).solve(ir, _B))))
    assert inc.status is AllocationStatus.INFEASIBLE
    assert inc.is_feasible is False


# --- the registry ------------------------------------------------------------------


def test_registry_lists_known_and_available_backends() -> None:
    # A superset, not an equality: the backend set is open — any installed distribution may
    # advertise `astro_mine.allocate.solvers`, so pinning the exact set would assert "no solver
    # plugin is installed", which is not the property under test.
    assert set(known_backends()) >= {CPSAT_BACKEND, TRIVIAL_STUB_BACKEND}
    assert TRIVIAL_STUB_BACKEND in available_backends()  # the stub has no third-party dependency


def test_resolve_trivial_stub_backend() -> None:
    solver = resolve_solver(
        TRIVIAL_STUB_BACKEND, task_kinds={"t": next(iter(_task_kinds(anchor_request()).values()))}
    )
    assert isinstance(solver, TrivialStubSolver)


def test_resolve_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown solver backend"):
        resolve_solver("gurobi", task_kinds={})


@pytest.mark.skipif(not _HAS_ORTOOLS, reason="OR-Tools not installed")
def test_resolve_cpsat_backend_when_available() -> None:
    solver = resolve_solver(CPSAT_BACKEND, task_kinds={})
    assert isinstance(solver, Solver)
    assert CPSAT_BACKEND in available_backends()
