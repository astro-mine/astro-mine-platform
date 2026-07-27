"""The RM-P1 **scale benchmark** — tens of robots / hundreds of tasks (allocate.md §8).

The Phase-1 exit criterion Allocate has never actually been measured against: "near-optimal (a few
% gap at most) plans for **tens of robots / hundreds of tasks** over a multi-day horizon within
seconds to a few minutes". Every other fixture in this suite is a 3-task/3-asset toy.
:mod:`tests.scale_factories` builds the instance the criterion names — 25 assets, 252 tasks, a
3-day horizon, with power, comms, terrain, scheduling, **and** disjoint-window constraints all live
— and this module measures against it and records the number.

**Opt-in.** These tests are marked ``scale`` and deselected by default (``pyproject.toml``
``addopts``): a wall-clock assertion on a noisy shared CI runner is a flake, not a signal. Run them
deliberately::

    uv run pytest -m scale

The scheduled ``scale-bench`` workflow does exactly that and uploads the recorded artifact, so the
trend is tracked over time — "every performance claim is a reproducible Bench number with pinned
solver versions" (allocate.md §8).

**What the gates mean.** The first assertion is the *deadline*: can CP-SAT find **and prove** a
conflict-free assignment + schedule for 252 tasks across 25 assets — honoring 25 ``NO_OVERLAP``
resources, ~400 terrain/comms keep-outs, 25 energy budgets and 14 two-window disjunctions — inside
a wall-clock budget?

The second is the **optimality gap**, and it now means something. It used not to: the objective
coefficients were per-*task* (a task was worth its value whoever did it) and the assignment cover is
exactly-one, so every feasible plan realized the identical objective and the gap was zero *by
construction* — a gauge reading zero because it was unplugged. The instance now carries a per-*pair*
cost family (:class:`~astro_mine.allocate.CostPolicy`: the energy **this** asset spends on **this**
task, priced into the task's value units and subtracted from it), so a plan that sends the wrong
robot to the wrong crater scores measurably worse, and :data:`MAX_GAP` is a real solver-quality
bound. :func:`test_the_gap_gate_has_teeth` is the proof: it runs a knowingly-suboptimal
(cost-blind) solver on the same instance and shows the very same gap assertion **fails** on it.

The instance also asks CP-SAT to *prove* its optimum (no ``target_gap`` ⇒ no
``relative_gap_limit``), because CP-SAT reports ``OPTIMAL`` both when it has proven optimality and
when it stopped at a gap limit — and only the first is what ``status is OPTIMAL`` claims here.
"""

from __future__ import annotations

import importlib.util
import json
import os
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import pytest

from astro_mine.allocate import (
    AllocationPlanner,
    AllocationStatus,
    ConstraintKind,
    compile_with_constraints,
    decompose_objective,
    extract_iis,
    verify_feasible,
)
from astro_mine.allocate.model.ir.schedule import scheduling_slack
from tests.allocate.scale_factories import CONFLICT_SITE, infeasible_scale_instance, scale_instance

pytestmark = [
    pytest.mark.scale,
    pytest.mark.skipif(
        importlib.util.find_spec("ortools") is None,
        reason="OR-Tools (cp-sat backend) not installed",
    ),
]

#: The stated wall-clock deadline for the design-mode solve (allocate.md §8: "within seconds to a
#: few minutes"). The instance's ``SolveBudget`` carries the same value, so CP-SAT itself stops at
#: it; the assertion is that it *finished* — proved an optimum — rather than being cut off.
DEADLINE_S = 60.0

#: The stated optimality-gap bound (a few % at most) — a live solver-quality gate against the
#: per-pair objective, not a tautology (see the module docstring;
#: :func:`test_the_gap_gate_has_teeth` proves a worse solver trips it).
MAX_GAP = 0.05

#: An irreducible infeasible set for a **localized** conflict must stay small even when the
#: surrounding instance is large — an operator asking "why can this haul not run?" must not be
#: handed a hundred constraints.
MAX_IIS_CONSTRAINTS = 8

#: And it must be extracted promptly: an infeasibility certificate a controller waits a minute for
#: is not an operational answer (allocate.md §2, principle 1).
MAX_IIS_S = 45.0

#: Where each run's numbers are recorded (the tracked-over-time artifact). Overridable so CI can
#: point it at an uploaded directory.
_BENCH_DIR = Path(
    os.environ.get(
        "ASTRO_MINE_ALLOCATE_BENCH_DIR", Path(__file__).resolve().parents[2] / "benchmarks"
    )
)


def _record(name: str, payload: dict[str, Any]) -> None:
    """Append one benchmark run to the tracked record (one JSON document per run).

    The solver version is pinned into the record alongside the numbers, because a benchmark number
    without the solver that produced it is not reproducible (allocate.md §5/§8).
    """
    _BENCH_DIR.mkdir(parents=True, exist_ok=True)
    document = {
        "benchmark": name,
        "recorded_at": datetime.now(UTC).isoformat(),
        "ortools": version("ortools"),
        "allocate": version("astro-mine-allocate"),
        **payload,
    }
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S")
    (_BENCH_DIR / f"{name}-{stamp}.json").write_text(json.dumps(document, indent=2) + "\n")
    print(f"\n[scale-bench] {json.dumps(document)}")


def test_scale_solve_meets_the_deadline_and_the_gap_bound() -> None:
    instance = scale_instance()
    assert instance.assets >= 25, "the criterion says *tens* of robots"
    assert instance.tasks >= 250, "the criterion says *hundreds* of tasks"

    compile_start = time.perf_counter()
    compiled = compile_with_constraints(
        instance.request, instance.context, config=instance.config, costs=instance.costs
    )
    compile_s = time.perf_counter() - compile_start

    solve_start = time.perf_counter()
    allocation = AllocationPlanner(backend="cp-sat").solve(
        instance.request,
        context=instance.context,
        config=instance.config,
        costs=instance.costs,
    )
    solve_s = time.perf_counter() - solve_start

    decomposition = allocation.objective_decomposition
    assert decomposition is not None
    families = {c.family: c.value for c in decomposition.contributions}

    _record(
        "scale-solve",
        {
            "assets": instance.assets,
            "tasks": instance.tasks,
            "ir_variables": len(compiled.ir.variables),
            "ir_constraints": len(compiled.ir.constraints),
            "forbidden_pairs": len(compiled.report.forbidden),
            "compile_s": round(compile_s, 3),
            "solve_s": round(solve_s, 3),
            "deadline_s": DEADLINE_S,
            "status": allocation.status.value,
            "optimality_gap": allocation.optimality_gap,
            "max_gap": MAX_GAP,
            "realized_objective": allocation.realized_objective,
            "objective_families": families,
        },
    )

    # The deadline.
    assert solve_s <= DEADLINE_S, (
        f"the {instance.assets}-asset / {instance.tasks}-task solve took {solve_s:.1f}s, "
        f"over the {DEADLINE_S:.0f}s design-mode deadline (allocate.md §8)"
    )
    # Solved, and *proven* solved — not cut off at the deadline with an unproven incumbent, and not
    # stopped early at a gap limit (the instance sets none, so OPTIMAL means CP-SAT closed the gap).
    assert allocation.status is AllocationStatus.OPTIMAL, (
        f"expected a proven optimum within the deadline, got {allocation.status.value}"
    )
    # The stated gap bound — a live gate now that the objective discriminates between plans
    # (test_the_gap_gate_has_teeth proves a worse solver trips this very assertion).
    assert allocation.optimality_gap is not None
    assert allocation.optimality_gap <= MAX_GAP

    # And the objective really is "value minus what it cost": the cost family is present and is a
    # genuine charge against the plan, not a rounding artifact.
    assert families["cost"] < 0.0, f"the cost family contributed nothing: {families}"
    assert abs(families["cost"]) > 0.05 * families["roi"], (
        "the cost family is too small to move the objective — the gap gate would be measuring "
        f"almost nothing again: {families}"
    )

    # And the plan is feasible against the model — checked independently, not taken on trust.
    assert verify_feasible(allocation, compiled.ir)
    assert allocation.plan is not None
    scheduled = {st.task_id for schedule in allocation.plan for st in schedule.tasks}
    assert len(scheduled) == instance.tasks, "every task must be covered exactly once"


def test_the_gap_gate_has_teeth() -> None:
    """The ``optimality_gap <= MAX_GAP`` assertion above **fails** on a knowingly-worse solver.

    A gate that cannot fail is not a gate (issue #22). So: run a solver that is deliberately bad in
    exactly the way the objective is supposed to punish — it hunts for the *most expensive*
    feasible assignment, sending tracked excavators to the craters the light rovers should take —
    and holds its plan up to the same bar the benchmark holds CP-SAT to.

    The bad plan is feasible: it satisfies every keep-out, budget, window and no-overlap constraint,
    and ``verify_feasible`` confirms it. It fails purely on *quality* — which is precisely what an
    optimality gap is supposed to detect and, before the per-pair objective landed, could not.
    """
    instance = scale_instance()
    compiled = compile_with_constraints(
        instance.request, instance.context, config=instance.config, costs=instance.costs
    )
    planner = AllocationPlanner(backend="cp-sat")

    good = planner.solve(
        instance.request,
        context=instance.context,
        config=instance.config,
        costs=instance.costs,
    )
    assert good.status is AllocationStatus.OPTIMAL
    assert good.realized_objective is not None

    # The saboteur: the same request with the cost weight *inverted*, so its solver is rewarded
    # for burning energy instead of saving it. Same constraints, same feasible region — it just
    # optimizes for the wrong thing, which is what a broken (or regressed) solver looks like.
    wasteful = instance.request.model_copy(
        update={
            "objective": instance.request.objective.model_copy(update={"weights": {"cost": -1.0}})
        }
    )
    bad = planner.solve(
        wasteful, context=instance.context, config=instance.config, costs=instance.costs
    )
    assert bad.plan is not None

    # Score the bad plan against the *real* objective, and re-check it really is feasible — the gate
    # must be failing it for being wasteful, not for being invalid.
    bad_objective = decompose_objective(bad.plan, compiled.ir).total
    assert verify_feasible(
        bad.model_copy(update={"realized_objective": bad_objective}), compiled.ir
    ), "the deliberately-wasteful plan must still be feasible — otherwise this proves nothing"

    # The gap the benchmark's gate would see for this plan: the proven optimum is its dual bound.
    bound = good.realized_objective
    bad_gap = abs(bound - bad_objective) / max(abs(bad_objective), 1.0)

    _record(
        "scale-gap-teeth",
        {
            "assets": instance.assets,
            "tasks": instance.tasks,
            "optimal_objective": bound,
            "wasteful_objective": bad_objective,
            "wasteful_gap": bad_gap,
            "max_gap": MAX_GAP,
        },
    )

    assert bad_gap > MAX_GAP, (
        f"a knowingly-wasteful plan scored {bad_objective:.2f} against the proven optimum "
        f"{bound:.2f} — a gap of only {bad_gap:.2%}, which the {MAX_GAP:.0%} gate would happily "
        "accept. The objective is not discriminating between plans strongly enough for the "
        "optimality gap to mean anything (issue #22)."
    )


def test_the_scale_plan_double_books_no_asset_and_lands_in_no_window_gap() -> None:
    # The RM-P1-ALLOC-02 correctness fix, asserted at scale rather than on a 2-task toy: with 25
    # NO_OVERLAP resources and 14 two-window disjunctions live, every one of them must hold.
    instance = scale_instance()
    compiled = compile_with_constraints(
        instance.request, instance.context, config=instance.config, costs=instance.costs
    )
    allocation = AllocationPlanner(backend="cp-sat").solve(
        instance.request, context=instance.context, config=instance.config, costs=instance.costs
    )
    assert allocation.plan is not None

    resources = [c for c in compiled.ir.constraints if c.kind is ConstraintKind.NO_OVERLAP]
    assert len(resources) == instance.assets, "every asset is a single-capacity resource"

    from astro_mine.allocate import plan_variable_values

    values = plan_variable_values(allocation.plan, compiled.ir)
    for constraint in resources:
        slack = scheduling_slack(constraint, compiled.ir, values)
        assert slack is None or slack >= -1.0e-9, f"{constraint.id} is double-booked"

    # verify_feasible re-derives the window disjunction too: a start in a gap satisfies no selector.
    assert verify_feasible(allocation, compiled.ir)


def test_the_iis_stays_small_and_prompt_on_a_large_infeasible_instance() -> None:
    """A *localized* conflict on the full-size instance must yield a small, irreducible set, fast.

    This is the check ``explain/iis.py`` used to only promise ("already minimal for the Phase-1
    instance sizes ... a deletion-filter refinement toward guaranteed irreducibility on larger
    conflicts is a noted follow-on"). The refinement now ships, and this measures it where it
    matters: one broken precedence chain buried in 252 tasks.
    """
    instance = infeasible_scale_instance()
    compiled = compile_with_constraints(
        instance.request, instance.context, config=instance.config, costs=instance.costs
    )

    start = time.perf_counter()
    certificate = extract_iis(compiled.ir, seed=19, report=compiled.report)
    iis_s = time.perf_counter() - start

    unrefined = extract_iis(compiled.ir, seed=19, report=compiled.report, refine=False)

    _record(
        "scale-iis",
        {
            "assets": instance.assets,
            "tasks": instance.tasks,
            "ir_variables": len(compiled.ir.variables),
            "ir_constraints": len(compiled.ir.constraints),
            "iis_s": round(iis_s, 3),
            "max_iis_s": MAX_IIS_S,
            "core_size": len(certificate.constraint_ids),
            "core_size_unrefined": len(unrefined.constraint_ids),
            "max_core_size": MAX_IIS_CONSTRAINTS,
            "constraint_ids": certificate.constraint_ids,
        },
    )

    assert iis_s <= MAX_IIS_S, (
        f"IIS extraction took {iis_s:.1f}s on a {instance.tasks}-task instance"
    )
    assert len(certificate.constraint_ids) <= MAX_IIS_CONSTRAINTS

    # The refinement can only ever shrink a correct conflict set, never grow one.
    assert set(certificate.constraint_ids) <= set(unrefined.constraint_ids)

    # And it names the *actual* conflict — the one broken chain, not 252 tasks' worth of noise.
    assert set(certificate.task_ids) == {
        f"excavate-{CONFLICT_SITE:03d}",
        f"haul-{CONFLICT_SITE:03d}",
    }
    assert f"prec::excavate-{CONFLICT_SITE:03d}->haul-{CONFLICT_SITE:03d}" in (
        certificate.constraint_ids
    )


def test_the_end_to_end_solve_of_the_infeasible_instance_returns_that_certificate() -> None:
    instance = infeasible_scale_instance()
    allocation = AllocationPlanner(backend="cp-sat").solve(
        instance.request, context=instance.context, config=instance.config, costs=instance.costs
    )

    assert allocation.status is AllocationStatus.INFEASIBLE
    assert allocation.plan is None
    certificate = allocation.infeasibility_certificate
    assert certificate is not None
    assert 0 < len(certificate.constraint_ids) <= MAX_IIS_CONSTRAINTS
    assert f"excavate-{CONFLICT_SITE:03d}" in certificate.task_ids
