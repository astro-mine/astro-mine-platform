"""IIS extraction names the minimal conflicting constraints (RM-P1-ALLOC-06/07).

The acceptance criteria (issue #6): an over-constrained instance returns an **irreducible
infeasible set** naming the minimal conflicting constraints — not a bare "infeasible" — and the
same infeasible model + seed + pinned solver yields the **same** IIS (the golden gate applies to
certificates too). Extracted through CP-SAT's assumption machinery over the solver-neutral IR.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import cast

import pytest

from astro_mine.allocate import (
    AllocationIR,
    AllocationPlanner,
    AllocationRequest,
    AllocationStatus,
    AssetRef,
    InfeasibilityCertificate,
    Task,
    TimeWindow,
    ValueEstimate,
    compile_request,
)
from astro_mine.allocate.constraints.compose import compile_with_constraints
from astro_mine.allocate.constraints.config import CommsPolicy, ConstraintConfig
from astro_mine.allocate.explain import extract_iis
from astro_mine.core.messages.enums import TaskKind
from astro_mine.core.sadf import CapabilityTag
from astro_mine.core.units import J2000_EPOCH
from tests.allocate import constraint_factories as F
from tests.allocate.factories import anchor_request, infeasible_request

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ortools") is None, reason="OR-Tools (cp-sat backend) not installed"
)

_GOLDEN = Path(__file__).resolve().parent / "golden" / "cpsat_iis_energy_shortfall.json"

_ENERGY_SHORTFALL_COSTS = F.cost_table(
    {("excavate-crater-a", "excavator-1"): (900.0, 2.0e7)}  # 2.0e7 J > excavator's 1.2e7 J budget
)


def _certificate_dict(cert: InfeasibilityCertificate) -> dict[str, object]:
    return {
        "constraint_ids": cert.constraint_ids,
        "task_ids": cert.task_ids,
        "explanation": cert.explanation,
    }


def test_energy_shortfall_iis_names_the_minimal_conflict() -> None:
    alloc = AllocationPlanner(backend="cp-sat").solve(
        anchor_request(), context=F.context(), costs=_ENERGY_SHORTFALL_COSTS
    )
    assert alloc.status is AllocationStatus.INFEASIBLE
    cert = alloc.infeasibility_certificate
    assert cert is not None
    # The excavate cover cannot be met without breaching the excavator's power budget — the
    # irreducible conflict is exactly those two constraints (not the whole model).
    assert cert.constraint_ids == ["cover::excavate-crater-a", "power::excavator-1"]
    assert cert.task_ids == ["excavate-crater-a"]
    assert cert.explanation is not None
    assert "CP-SAT proved the composed model infeasible" in cert.explanation


def test_iis_certificate_matches_the_committed_golden() -> None:
    alloc = AllocationPlanner(backend="cp-sat").solve(
        anchor_request(), context=F.context(), costs=_ENERGY_SHORTFALL_COSTS
    )
    golden = json.loads(_GOLDEN.read_text())
    cert = alloc.infeasibility_certificate
    assert cert is not None  # an infeasible solve must carry a certificate at all
    assert _certificate_dict(cert) == golden, (
        "the IIS certificate drifted from the golden — a non-reproducibility (or an intended "
        "change; regenerate tests/golden/cpsat_iis_energy_shortfall.json)"
    )


def test_same_infeasible_model_and_seed_yields_the_same_iis() -> None:
    ir = compile_with_constraints(anchor_request(), F.context(), costs=_ENERGY_SHORTFALL_COSTS).ir
    first = extract_iis(ir, seed=7)
    second = extract_iis(ir, seed=7)
    assert _certificate_dict(first) == _certificate_dict(second)


def test_comms_starved_haul_iis_names_the_haul_cover_and_keepouts() -> None:
    # Every eligible haul asset sees only a 300 s window but the haul needs 600 s → its pairs are
    # kept out and its cover becomes unsatisfiable.
    config = ConstraintConfig(
        comms=CommsPolicy(relay_required_task_ids=frozenset({"haul-to-plant"}), epoch0=J2000_EPOCH)
    )
    costs = F.cost_table(
        {
            ("excavate-crater-a", "excavator-1"): (900.0, 3.0e6),
            ("prospect-crater-a", "prospector-rover-1"): (600.0, 1.0e6),
            ("haul-to-plant", "hauler-1"): (600.0, 1.0e6),
            ("haul-to-plant", "prospector-rover-1"): (600.0, 1.0e6),
        }
    )
    ctx = F.context(
        contacts=F.contact_plan(
            {"prospector-rover-1": (5000.0, 5300.0), "hauler-1": (5000.0, 5300.0)}
        )
    )
    alloc = AllocationPlanner(backend="cp-sat").solve(
        anchor_request(), context=ctx, config=config, costs=costs
    )
    assert alloc.status is AllocationStatus.INFEASIBLE
    cert = alloc.infeasibility_certificate
    assert cert is not None
    assert "cover::haul-to-plant" in cert.constraint_ids
    assert all(c.startswith(("cover::haul", "keepout::haul")) for c in cert.constraint_ids)
    assert cert.task_ids == ["haul-to-plant"]
    assert cert.explanation is not None
    assert "no contact window long enough to relay" in cert.explanation


def test_skeleton_infeasibility_iis_names_the_uncoverable_task() -> None:
    # A task requiring a capability no asset declares → an empty cover, infeasible with no report.
    ir = compile_request(infeasible_request())
    cert = extract_iis(ir, seed=7)
    assert cert.constraint_ids == ["cover::drill-1"]
    assert cert.task_ids == ["drill-1"]
    assert cert.explanation is not None
    assert "no eligible asset for: drill-1" in cert.explanation
    assert "CP-SAT proved the model infeasible" in cert.explanation


def test_tasks_named_directly_in_a_constraint_id() -> None:
    # The id-parsing fallback that names tasks even when a constraint carries no terms.
    from astro_mine.allocate.explain.iis import _tasks_from_id

    assert _tasks_from_id("cover::drill-1") == {"drill-1"}
    assert _tasks_from_id("keepout::excavate-crater-a::excavator-1") == {"excavate-crater-a"}
    assert _tasks_from_id("prec::prospect->excavate") == {"prospect", "excavate"}
    assert _tasks_from_id("power::excavator-1") == set()  # names an asset, not a task
    assert _tasks_from_id("bare") == set()  # no "::" separator


# --- the deletion-filter refinement (RM-P1-ALLOC-06; the noted "does it scale" follow-on) ----


def test_the_deletion_filter_shrinks_a_non_minimal_candidate_to_an_irreducible_core() -> None:
    # Two *independent* conflicts in one model: an uncoverable task (no eligible asset) and a task
    # whose window contradicts its own precedence edge. CP-SAT's sufficient-assumption set is free
    # to hand back either — or a superset spanning both. Whatever it returns, the deletion filter
    # must reduce it to a set in which every member is load-bearing.
    ir = compile_request(_two_independent_conflicts())

    unrefined = extract_iis(ir, seed=7, refine=False)
    refined = extract_iis(ir, seed=7)

    assert set(refined.constraint_ids) <= set(unrefined.constraint_ids)
    # Irreducible: dropping any single member of the core admits a feasible relaxation.
    for constraint_id in refined.constraint_ids:
        remainder = [c for c in refined.constraint_ids if c != constraint_id]
        assert not _is_infeasible_without(ir, remainder, dropped=constraint_id)


def test_the_refined_certificate_is_deterministic() -> None:
    ir = compile_request(_two_independent_conflicts())
    first = extract_iis(ir, seed=7)
    second = extract_iis(ir, seed=7)
    assert first.model_dump(mode="json") == second.model_dump(mode="json")


def test_a_singleton_conflict_is_already_irreducible_and_skips_the_filter() -> None:
    # The early-out: one constraint cannot be reduced further, so no trial solve is spent on it.
    ir = compile_request(infeasible_request())
    assert extract_iis(ir, seed=7).constraint_ids == ["cover::drill-1"]


def _two_independent_conflicts() -> AllocationRequest:
    """A request carrying two unrelated infeasibilities, so a *sufficient* core may over-name."""
    return AllocationRequest(
        request_id="two-conflicts-001",
        tasks=[
            # (1) no asset declares the capability this task needs.
            Task(
                task_id="drill-1",
                kind=TaskKind.SAMPLE,
                required_capabilities=[CapabilityTag.SAMPLE_COLLECTION_DRILL],
                value=ValueEstimate(mean=1.0),
            ),
            # (2) `late` must precede `early`, but `early`'s window closes before `late`'s opens.
            Task(
                task_id="late",
                kind=TaskKind.PROSPECT,
                time_windows=[TimeWindow(start_s=1000.0, end_s=2000.0)],
                value=ValueEstimate(mean=1.0),
            ),
            Task(
                task_id="early",
                kind=TaskKind.PROSPECT,
                time_windows=[TimeWindow(start_s=0.0, end_s=500.0)],
                precedence=["late"],
                value=ValueEstimate(mean=1.0),
            ),
        ],
        assets=[AssetRef(asset_id="rover-a", capability_tags=[CapabilityTag.MOBILITY_WHEELED])],
    )


def _is_infeasible_without(ir: AllocationIR, keep: list[str], *, dropped: str) -> bool:
    """Whether the core minus ``dropped`` is still infeasible (it must not be, if the core is
    minimal)."""
    from ortools.sat.python import cp_model

    from astro_mine.allocate.model.compile.cpsat import lower_for_iis

    lowered = lower_for_iis(ir, keep=set(keep))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    # OR-Tools annotates `Solve()` as returning `CpSolverStatus` — the enum *wrapper class* —
    # while its status constants are `CpSolverStatus.ValueType`, a `NewType` over `int`. mypy
    # therefore reads the comparison as non-overlapping (always false). Both sides are plain
    # ints at runtime, so cast to the value's real type rather than suppress the check: a status
    # comparison that can never be true is exactly the failure this test must not have.
    return cast(int, solver.Solve(lowered.model)) == cp_model.INFEASIBLE
