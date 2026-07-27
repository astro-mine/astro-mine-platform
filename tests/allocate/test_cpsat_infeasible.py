"""CP-SAT infeasibility surfaces honestly to the certificate path (RM-P1-ALLOC-02).

The feasibility acceptance criterion (allocate.md §10): the backend returns a feasible plan **or**
surfaces infeasibility to the certificate slot — never a silent empty plan. A skeleton solve names
the uncoverable tasks; a constrained solve joins the builder findings (the terrain keep-out / comms
window / energy shortfall that bound the result); an anytime UNKNOWN carries neither plan nor a
false certificate.
"""

from __future__ import annotations

import importlib.util

import pytest

from astro_mine.allocate import (
    AllocationPlanner,
    AllocationStatus,
    ConstraintConfig,
    compile_request,
)
from astro_mine.allocate.constraints.config import CommsPolicy
from astro_mine.core.units import J2000_EPOCH
from tests.allocate import constraint_factories as F
from tests.allocate.factories import anchor_request, infeasible_request

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ortools") is None, reason="OR-Tools (cp-sat backend) not installed"
)

_HAUL_GATED = ConstraintConfig(
    comms=CommsPolicy(relay_required_task_ids=frozenset({"haul-to-plant"}), epoch0=J2000_EPOCH)
)
_ANCHOR_COSTS = F.cost_table(
    {
        ("prospect-crater-a", "prospector-rover-1"): (600.0, 1.0e6),
        ("excavate-crater-a", "excavator-1"): (900.0, 3.0e6),
        ("haul-to-plant", "hauler-1"): (600.0, 1.0e6),
        ("haul-to-plant", "prospector-rover-1"): (600.0, 1.0e6),
    }
)


def test_skeleton_infeasibility_names_the_uncoverable_task() -> None:
    alloc = AllocationPlanner(backend="cp-sat").solve(infeasible_request())
    assert alloc.status is AllocationStatus.INFEASIBLE
    assert alloc.plan is None
    cert = alloc.infeasibility_certificate
    assert cert is not None
    assert "drill-1" in cert.task_ids
    assert cert.explanation is not None
    assert "no eligible asset" in cert.explanation


def test_slope_keepout_is_an_explicit_certificate() -> None:
    ctx = F.context(world=F.FakeWorld(slope_deg=40.0))  # over the 25° default limit
    alloc = AllocationPlanner(backend="cp-sat").solve(anchor_request(), context=ctx)
    assert alloc.status is AllocationStatus.INFEASIBLE
    cert = alloc.infeasibility_certificate
    assert cert is not None
    assert cert.explanation is not None
    assert "CP-SAT proved the composed model infeasible" in cert.explanation


def test_no_contact_window_is_an_explicit_certificate() -> None:
    # Every eligible haul asset sees only a 300 s window but the haul needs 600 s → infeasible.
    ctx = F.context(
        contacts=F.contact_plan(
            {"prospector-rover-1": (5000.0, 5300.0), "hauler-1": (5000.0, 5300.0)}
        )
    )
    alloc = AllocationPlanner(backend="cp-sat").solve(
        anchor_request(), context=ctx, config=_HAUL_GATED, costs=_ANCHOR_COSTS
    )
    assert alloc.status is AllocationStatus.INFEASIBLE
    cert = alloc.infeasibility_certificate
    assert cert is not None
    assert cert.explanation is not None
    assert "no contact window long enough to relay" in cert.explanation
    assert "haul-to-plant" in cert.task_ids


def test_energy_budget_shortfall_is_infeasible() -> None:
    # The excavate energy cost (2.0e7 J) exceeds the excavator's 1.2e7 J budget → no feasible asset.
    # CP-SAT proves the composed model infeasible; per-constraint attribution (which task/budget
    # bound the result) is the RM-P1-ALLOC-06 IIS, not the MVP certificate.
    costs = F.cost_table({("excavate-crater-a", "excavator-1"): (900.0, 2.0e7)})
    alloc = AllocationPlanner(backend="cp-sat").solve(
        anchor_request(), context=F.context(), costs=costs
    )
    assert alloc.status is AllocationStatus.INFEASIBLE
    assert alloc.infeasibility_certificate is not None
    assert "CP-SAT proved the composed model infeasible" in (
        alloc.infeasibility_certificate.explanation or ""
    )


def test_unknown_terminal_status_maps_to_unknown() -> None:
    from ortools.sat.python import cp_model

    from astro_mine.allocate.solvers.cpsat import _map_terminal_status

    assert _map_terminal_status(cp_model.INFEASIBLE) is AllocationStatus.INFEASIBLE
    assert _map_terminal_status(cp_model.UNKNOWN) is AllocationStatus.UNKNOWN
    assert _map_terminal_status(cp_model.MODEL_INVALID) is AllocationStatus.UNKNOWN


def test_timeout_result_carries_no_plan_and_no_false_certificate() -> None:
    # A terminal solve that searched but found no incumbent and no infeasibility proof is an honest
    # anytime TIMEOUT, not a silent UNKNOWN (RM-P1-ALLOC-05). Exercised directly — CP-SAT will not
    # reliably time out on a small instance.
    planner = AllocationPlanner(backend="cp-sat")
    request = anchor_request()
    ir = compile_request(request)
    prov = planner._provenance(request, ir)
    alloc = planner._cpsat_unsolved(request, ir, None, prov, AllocationStatus.UNKNOWN)
    assert alloc.status is AllocationStatus.TIMEOUT
    assert alloc.plan is None
    assert alloc.infeasibility_certificate is None
