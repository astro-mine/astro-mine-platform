"""CP-SAT determinism: seeded solves are byte-reproducible (RM-P1-ALLOC-02/07 golden gate).

The determinism acceptance criterion (allocate.md §8; conventions.md §11): the **same model +
same seed + pinned OR-Tools** yields an identical plan, and a seeded **golden plan** re-solved is
byte-identical — CI fails on non-reproducibility. Determinism is fixed-worker deterministic-mode
(a single search worker + a deterministic, not wall-clock, time bound), so the plan does not depend
on machine speed. The wall-clock ``budget_consumed_s`` is runtime telemetry and deliberately
excluded from the reproducible artifact; the plan, objective, gap, and status are the golden.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest

from astro_mine.allocate import Allocation, AllocationPlanner, AllocationStatus, compile_request
from astro_mine.allocate.anytime import stream_incumbents
from astro_mine.allocate.solvers.registry import CPSAT_BACKEND, resolve_solver
from tests.allocate import constraint_factories as F
from tests.allocate.factories import anchor_request

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("ortools") is None, reason="OR-Tools (cp-sat backend) not installed"
)

_GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
_GOLDEN = _GOLDEN_DIR / "cpsat_anchor_seed7.json"
_CONSTRAINED_GOLDEN = _GOLDEN_DIR / "cpsat_anchor_constrained_seed7.json"
_TRAJECTORY_GOLDEN = _GOLDEN_DIR / "cpsat_anchor_trajectory_seed7.json"

#: A fixed, deterministic constrained scenario: the anchor with per-pair durations + energies so
#: the CP-SAT constrained path (augmented IR, real end times, energy budgets) is byte-reproducible.
_CONSTRAINED_COSTS = F.cost_table(
    {
        ("prospect-crater-a", "prospector-rover-1"): (600.0, 1.0e6),
        ("excavate-crater-a", "excavator-1"): (900.0, 3.0e6),
        ("haul-to-plant", "hauler-1"): (600.0, 1.0e6),
        ("haul-to-plant", "prospector-rover-1"): (600.0, 1.0e6),
    }
)


def _reproducible(alloc: Allocation) -> dict[str, Any]:
    """The reproducible slice of an allocation — everything but the wall-clock telemetry."""
    return {
        "status": alloc.status.value,
        "realized_objective": alloc.realized_objective,
        "optimality_gap": alloc.optimality_gap,
        "plan": [a.model_dump(mode="json") for a in (alloc.plan or [])],
    }


def _reproducible_explained(alloc: Allocation) -> dict[str, Any]:
    """The reproducible slice including the RM-P1-ALLOC-06 explanation (binding + decomposition)."""
    return {
        **_reproducible(alloc),
        "binding_constraints": [b.model_dump(mode="json") for b in alloc.binding_constraints],
        "objective_decomposition": (
            alloc.objective_decomposition.model_dump(mode="json")
            if alloc.objective_decomposition is not None
            else None
        ),
    }


def _trajectory(alloc_request: Any) -> list[dict[str, Any]]:
    """The incumbent trajectory (objective / monotone bound / gap) of a seeded solve."""
    ir = compile_request(alloc_request)
    solver = resolve_solver(
        CPSAT_BACKEND, task_kinds={t.task_id: t.kind for t in alloc_request.tasks}
    )
    return [
        {
            "objective": round(i.objective, 6),
            "bound": round(i.bound, 6) if i.bound is not None else None,
            "gap": round(i.gap, 6) if i.gap is not None else None,
            "status": i.status.value,
        }
        for i in stream_incumbents(solver, ir, alloc_request.budget, ir.objective_sense)
    ]


def test_same_request_and_seed_resolves_byte_identically() -> None:
    request = anchor_request()  # budget carries seed=7, deterministic=True
    first = AllocationPlanner(backend="cp-sat").solve(request)
    second = AllocationPlanner(backend="cp-sat").solve(request)
    assert json.dumps(_reproducible(first), sort_keys=True) == json.dumps(
        _reproducible(second), sort_keys=True
    )


def test_seeded_solve_matches_the_committed_golden_plan() -> None:
    alloc = AllocationPlanner(backend="cp-sat").solve(anchor_request())
    assert alloc.status is AllocationStatus.OPTIMAL
    golden = json.loads(_GOLDEN.read_text())
    assert _reproducible(alloc) == golden, (
        "the seeded CP-SAT plan drifted from the golden — a non-reproducibility (or an intended "
        "change; regenerate tests/golden/cpsat_anchor_seed7.json)"
    )


def test_seeded_constrained_solve_matches_the_committed_golden() -> None:
    # The constrained path (augmented IR + explanation) is reproducible too — the golden pins the
    # plan, the binding constraints, and the objective decomposition.
    first = AllocationPlanner(backend="cp-sat").solve(
        anchor_request(), context=F.context(), costs=_CONSTRAINED_COSTS
    )
    second = AllocationPlanner(backend="cp-sat").solve(
        anchor_request(), context=F.context(), costs=_CONSTRAINED_COSTS
    )
    assert _reproducible_explained(first) == _reproducible_explained(second)
    golden = json.loads(_CONSTRAINED_GOLDEN.read_text())
    assert _reproducible_explained(first) == golden, (
        "the seeded constrained CP-SAT plan/explanation drifted from the golden — a "
        "non-reproducibility (or an intended change; regenerate "
        "tests/golden/cpsat_anchor_constrained_seed7.json)"
    )


def test_incumbent_trajectory_matches_the_committed_golden() -> None:
    # The anytime incumbent trajectory (objective / monotone bound / gap sequence) is a golden too.
    first = _trajectory(anchor_request())
    second = _trajectory(anchor_request())
    assert first == second
    golden = json.loads(_TRAJECTORY_GOLDEN.read_text())
    assert first == golden, (
        "the seeded incumbent trajectory drifted from the golden — a non-reproducibility (or an "
        "intended change; regenerate tests/golden/cpsat_anchor_trajectory_seed7.json)"
    )


def test_provenance_pins_the_ortools_backend_version() -> None:
    alloc = AllocationPlanner(backend="cp-sat").solve(anchor_request())
    assert alloc.provenance.backend == "cp-sat"
    assert alloc.provenance.backend_version is not None
    assert alloc.provenance.backend_version.startswith("ortools ")
    assert alloc.provenance.seed == 7
