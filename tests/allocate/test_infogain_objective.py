"""The info-gain-vs-ROI objective (RM-P1-ALLOC-04).

Traces the acceptance criteria (allocate.md §11; scenario §7; charter §8): the builder folds a
Prospect **EVPI** info-gain term alongside the extraction-ROI term weighted by the Core
``Objective.weights``; sweeping the trade weight moves the plan **monotonically** between
extraction-heavy and prospect-heavy; the realized objective equals the info-gain + ROI split; the
distributional value is honest (a variance proxy is flagged degraded); and feasibility and
determinism are preserved.

The builder tests need no solver; the trade / property / determinism tests drive CP-SAT.
"""

from __future__ import annotations

import importlib.util
import json

import pytest

from astro_mine.allocate import (
    Allocation,
    AllocationIR,
    AllocationPlanner,
    AllocationRequest,
    AllocationStatus,
    AssetRef,
    Constraint,
    ConstraintContext,
    ConstraintTerm,
    DecisionVariable,
    Objective,
    ObjectiveSense,
    ObjectiveTerm,
    Task,
    ValueEstimate,
    compile_request,
    compile_with_constraints,
    verify_feasible,
)
from astro_mine.allocate.constraints.config import ConstraintConfig
from astro_mine.allocate.constraints.infogain import refine_infogain_objective
from astro_mine.allocate.enums import (
    ConstraintKind,
    ConstraintSense,
    VariableKind,
    VariableSemantic,
)
from astro_mine.core.messages.enums import TaskKind
from tests.allocate import constraint_factories as F
from tests.allocate.factories import anchor_request

_HAS_ORTOOLS = importlib.util.find_spec("ortools") is not None
requires_ortools = pytest.mark.skipif(not _HAS_ORTOOLS, reason="OR-Tools not installed")


# --- builder (no solver) -----------------------------------------------------------


def _base() -> tuple[AllocationRequest, AllocationIR, tuple[ObjectiveTerm, ...]]:
    request = anchor_request()
    base_ir = compile_request(request)
    return request, base_ir, tuple(base_ir.objective_terms)


def test_weights_scale_roi_and_add_injected_info_terms() -> None:
    request, base_ir, roi = _base()
    ctx = ConstraintContext(info_values={"prospect-crater-a": 100.0})
    result = refine_infogain_objective(
        request,
        base_ir,
        roi,
        ctx,
        config=ConstraintConfig(),
        weights={"roi": 2.0, "info_gain": 3.0},
    )
    by_id = {t.id: t for t in result.objective_terms}
    # Every ROI term is scaled by w_roi and tagged "roi".
    for term in roi:
        assert by_id[term.id].coefficient == pytest.approx(term.coefficient * 2.0)
        assert result.metadata[f"objective_family::{term.id}"] == "roi"
    # An info-gain term rides each prospect assignment var at w_info * EVPI, tagged info_gain.
    info_terms = [t for t in result.objective_terms if t.id.startswith("infogain::")]
    assert info_terms and all(t.coefficient == pytest.approx(3.0 * 100.0) for t in info_terms)
    assert all(result.metadata[f"objective_family::{t.id}"] == "info_gain" for t in info_terms)
    assert result.metadata["info_value::prospect-crater-a"] == repr(100.0)
    assert any(f.code == "infogain.ingested" for f in result.findings)
    assert result.degraded == ()  # injected EVPI is distributional — not degraded


def test_infogain_is_inactive_without_a_weight_or_injection() -> None:
    request, base_ir, roi = _base()
    result = refine_infogain_objective(
        request, base_ir, roi, ConstraintContext(), config=ConstraintConfig(), weights={}
    )
    # ROI terms unchanged (x1.0), no info-gain terms, no findings.
    assert {t.id for t in result.objective_terms} == {t.id for t in roi}
    assert result.findings == () and result.degraded == ()


def test_variance_proxy_is_flagged_degraded() -> None:
    request, base_ir, roi = _base()
    ctx = ConstraintContext(resource=F.FakeField(mean=0.4, variance=0.05))
    result = refine_infogain_objective(
        request, base_ir, roi, ctx, config=ConstraintConfig(), weights={"info_gain": 1.0}
    )
    info_terms = [t for t in result.objective_terms if t.id.startswith("infogain::")]
    # prospect-crater-a carries a location + resource target → a variance proxy stands in for EVPI.
    assert info_terms and all(t.coefficient == pytest.approx(0.05) for t in info_terms)
    assert "infogain.deterministic" in result.degraded
    assert any(f.code == "infogain.variance_proxy" for f in result.findings)


def test_compose_lands_infogain_in_the_augmented_ir() -> None:
    request = anchor_request(
        objective=Objective(sense=ObjectiveSense.MAXIMIZE, weights={"roi": 1.0, "info_gain": 2.0})
    )
    ctx = F.context(info_values={"prospect-crater-a": 100.0})
    comp = compile_with_constraints(request, ctx)
    families = {
        v.split("::", 1)[1]: fam
        for v, fam in comp.ir.metadata.items()
        if v.startswith("objective_family::")
    }
    assert "info_gain" in families.values()
    assert "roi" in families.values()


# --- trade / property / determinism / feasibility (CP-SAT) -------------------------


def _assign_var(task: str) -> DecisionVariable:
    return DecisionVariable(
        id=f"assign::{task}::R",
        kind=VariableKind.BINARY,
        lower=0.0,
        upper=1.0,
        semantic=VariableSemantic.ASSIGNMENT,
        task_ref=task,
        asset_ref="R",
    )


def _selection_ir(w_info: float) -> tuple[AllocationRequest, AllocationIR]:
    """A prospect(P)-vs-extract(E) selection: one shared slot, low-ROI/high-info P vs high-ROI E."""
    request = AllocationRequest(
        request_id="trade",
        tasks=[
            Task(task_id="P", kind=TaskKind.PROSPECT, value=ValueEstimate(mean=1.0)),
            Task(task_id="E", kind=TaskKind.EXCAVATE, value=ValueEstimate(mean=50.0)),
        ],
        assets=[AssetRef(asset_id="R")],
    )
    variables = [_assign_var("P"), _assign_var("E")]
    roi = (
        ObjectiveTerm(id="obj::assign::P::R", var_ref="assign::P::R", coefficient=1.0),
        ObjectiveTerm(id="obj::assign::E::R", var_ref="assign::E::R", coefficient=50.0),
    )
    base_ir = AllocationIR(
        variables=variables, objective_terms=list(roi), objective_sense=ObjectiveSense.MAXIMIZE
    )
    result = refine_infogain_objective(
        request,
        base_ir,
        roi,
        ConstraintContext(info_values={"P": 100.0}),
        config=ConstraintConfig(),
        weights={"roi": 1.0, "info_gain": w_info},
    )
    mutex = Constraint(
        id="mutex::R",
        kind=ConstraintKind.LINEAR,
        terms=[
            ConstraintTerm(var_ref="assign::P::R", coefficient=1.0),
            ConstraintTerm(var_ref="assign::E::R", coefficient=1.0),
        ],
        sense=ConstraintSense.LE,
        rhs=1.0,
    )
    ir = AllocationIR(
        variables=variables,
        constraints=[mutex],
        objective_terms=list(result.objective_terms),
        objective_sense=ObjectiveSense.MAXIMIZE,
    )
    return request, ir


@requires_ortools
def test_trade_weight_moves_the_plan_monotonically() -> None:
    from astro_mine.allocate.solvers.cpsat import CpSatSolver

    task_kinds = {"P": TaskKind.PROSPECT, "E": TaskKind.EXCAVATE}
    # "prospect-ness": 1 when the plan prospects (P), 0 when it extracts (E).
    prospectness = []
    for w_info in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        request, ir = _selection_ir(w_info)
        inc = list(CpSatSolver(task_kinds=task_kinds).solve(ir, request.budget))[-1]
        chosen = {st.task_id for a in inc.plan for st in a.tasks}
        prospectness.append(1 if "P" in chosen else 0)
    assert prospectness[0] == 0 and prospectness[-1] == 1  # extraction-heavy → prospect-heavy
    assert prospectness == sorted(prospectness)  # monotone, single transition


@requires_ortools
def test_realized_objective_is_the_info_gain_plus_roi_split() -> None:
    request = anchor_request(
        objective=Objective(sense=ObjectiveSense.MAXIMIZE, weights={"roi": 1.0, "info_gain": 2.0})
    )
    ctx = F.context(info_values={"prospect-crater-a": 100.0})
    comp = compile_with_constraints(request, ctx)
    alloc = AllocationPlanner(backend="cp-sat").solve(request, context=ctx)
    assert alloc.status in (AllocationStatus.OPTIMAL, AllocationStatus.FEASIBLE)
    assert alloc.plan is not None

    assignment = {st.task_id: sched.asset_id for sched in alloc.plan for st in sched.tasks}
    var_by_id = {v.id: v for v in comp.ir.variables}
    roi_sum = info_sum = 0.0
    for term in comp.ir.objective_terms:
        var = var_by_id[term.var_ref]
        if var.task_ref is None or assignment.get(var.task_ref) != var.asset_ref:
            continue
        family = comp.ir.metadata[f"objective_family::{term.id}"]
        if family == "roi":
            roi_sum += term.coefficient
        else:
            info_sum += term.coefficient
    assert info_sum > 0.0  # the info-gain family actually contributes
    assert alloc.realized_objective == pytest.approx(roi_sum + info_sum)
    assert verify_feasible(alloc, comp.ir)  # feasibility preserved with the objective added


@requires_ortools
def test_infogain_solve_is_deterministic() -> None:
    request = anchor_request(
        objective=Objective(sense=ObjectiveSense.MAXIMIZE, weights={"roi": 1.0, "info_gain": 5.0})
    )
    ctx = F.context(info_values={"prospect-crater-a": 100.0})
    first = AllocationPlanner(backend="cp-sat").solve(request, context=ctx)
    second = AllocationPlanner(backend="cp-sat").solve(request, context=ctx)

    def dump(alloc: Allocation) -> str:
        assert alloc.plan is not None
        return json.dumps([s.model_dump(mode="json") for s in alloc.plan], sort_keys=True)

    assert dump(first) == dump(second)
