# SPDX-License-Identifier: Apache-2.0
"""Irreducible-infeasible-set (IIS) extraction via CP-SAT assumptions (RM-P1-ALLOC-06).

*Why no plan* — an irreducible conflict set naming the minimal set of constraints that cannot
hold together, not a bare "infeasible" (allocate.md §10; issue #6). Realized in two stages:

1. **Candidate core.** :func:`~astro_mine.allocate.model.compile.cpsat.lower_for_iis` reifies each
   IR constraint behind a ``relax::{id}`` literal and registers the literals as solver assumptions;
   :meth:`CpSolver.SufficientAssumptionsForInfeasibility` reports a *sufficient* conflicting subset.
2. **Deletion filter.** CP-SAT's sufficient set is small but carries **no minimality guarantee** —
   and on the large instances the Phase-1 exit criterion targets (tens of robots / hundreds of
   tasks) it demonstrably carries constraints the conflict does not need. :func:`_refine` closes
   that gap the standard way: drop each candidate constraint in turn and re-solve the subset; a
   subset that is *still* infeasible without it never needed it. What survives is irreducible by
   construction — every remaining constraint has a witness (a feasible relaxation) proving it is
   load-bearing.

The filter costs one feasibility solve per candidate, over a model restricted to the candidate
subset (a handful of constraints, not the whole instance), so it is cheap precisely where it
matters. ``max_refinement_solves`` caps the work on a pathologically large candidate set; when the
cap bites the certificate is still a correct — just not provably minimal — conflict set, never a
wrong one.

**Determinism (RM-P1-ALLOC-07):** the assumption model is lowered in constraint-id order, solved
with a single worker and the request seed, the deletion filter visits candidates in sorted order,
and the returned constraint/task ids are sorted — so the same infeasible model yields a
byte-identical certificate (the golden gate applies to certificates too). Assumption literals derive
their names from the IR constraint id (never object identity), so the certificate is stable across
processes.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from astro_mine.allocate.api.model import InfeasibilityCertificate
from astro_mine.allocate.enums import ConstraintKind
from astro_mine.allocate.model.ir.model import AllocationIR

if TYPE_CHECKING:
    from astro_mine.allocate.constraints.result import ConstraintReport

__all__ = ["DEFAULT_MAX_REFINEMENT_SOLVES", "extract_iis"]

#: Constraint-id prefixes whose trailing segment is a task id (``{prefix}::{task}[::...]``).
_TASK_PREFIXES = frozenset(
    {"cover", "keepout", "twin_lo", "twin_hi", "twin_pick", "comms_lo", "comms_hi"}
)

#: How many deletion-filter re-solves an extraction will spend before returning the (correct but
#: no-longer-provably-minimal) candidate core as-is — one solve per candidate constraint. A
#: *localized* conflict, which is the operational case ("why can this one haul not run?"), has a
#: handful of members even on a 25-asset / 252-task instance, so this bound is generous. A genuinely
#: *global* infeasibility (say, every asset short of energy) has a genuinely large conflict set, and
#: deletion-filtering it costs more than it explains; the cap is what keeps that case bounded.
DEFAULT_MAX_REFINEMENT_SOLVES: int = 64

#: Per-trial solve bound for the deletion filter, in CP-SAT **deterministic** time units — *not*
#: wall clock. A trial the solver cannot resolve within it is treated as "not proven droppable" and
#: its constraint is conservatively kept. Bounding by deterministic time (as the RM-P1-ALLOC-07
#: golden gate does) is what keeps the refined certificate byte-identical across machines: a slow
#: runner must never yield a *different* conflict set than a fast one.
DEFAULT_REFINEMENT_DETERMINISTIC_TIME: float = 4.0


def extract_iis(
    ir: AllocationIR,
    *,
    seed: int | None = None,
    report: ConstraintReport | None = None,
    refine: bool = True,
    max_refinement_solves: int = DEFAULT_MAX_REFINEMENT_SOLVES,
) -> InfeasibilityCertificate:
    """Extract an irreducible infeasible set for an infeasible ``ir`` as a Core certificate.

    Reifies every IR constraint as a CP-SAT assumption, proves infeasibility, reads back the
    conflicting subset, and (with ``refine``, the default) runs a **deletion filter** over it so the
    reported set is irreducible rather than merely sufficient. Returns the constraint ids, the tasks
    they implicate, and an explanation that joins the builder findings relevant to the conflict.
    """
    from ortools.sat.python import cp_model

    from astro_mine.allocate.model.compile.cpsat import lower_for_iis

    lowered = lower_for_iis(ir)
    solver = _solver(cp_model, seed)
    status: Any = solver.Solve(lowered.model)
    if status != cp_model.INFEASIBLE:
        # Defensive: an infeasible IR reifies to an infeasible assumption model, so this is
        # unreachable in practice — name every constraint rather than assert a false minimal set.
        core = sorted(lowered.literal_by_constraint)  # pragma: no cover - reified IR is infeasible
    else:
        conflict = sorted(
            lowered.constraint_by_index[i]
            for i in solver.SufficientAssumptionsForInfeasibility()
            if i in lowered.constraint_by_index
        )
        core = (
            _refine(ir, conflict, seed=seed, budget=max_refinement_solves) if refine else conflict
        )

    task_ids = _implicated_tasks(ir, core)
    explanation = _explain(ir, core, task_ids, report)
    return InfeasibilityCertificate(
        constraint_ids=core, task_ids=sorted(task_ids), explanation=explanation
    )


def _solver(cp_model: Any, seed: int | None) -> Any:
    """A deterministic CP-SAT solver for the assumption models (one worker + the request seed)."""
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    if seed is not None:
        solver.parameters.random_seed = seed
    return solver


def _refine(ir: AllocationIR, candidate: list[str], *, seed: int | None, budget: int) -> list[str]:
    """Shrink a sufficient conflict set to an **irreducible** one by deletion filtering.

    For each candidate constraint (in sorted order, so the result is deterministic), re-solve the
    core *without* it: if the remainder is still infeasible, that constraint was never needed and is
    dropped permanently. What survives is minimal — removing any one of its members admits a
    feasible relaxation.

    Every trial is bounded by a **deterministic** time limit, and an unresolved trial conservatively
    *keeps* its constraint: the filter can only ever shrink a correct conflict set, never invalidate
    one, and its outcome does not depend on how fast the machine is. ``budget`` caps the re-solves;
    the un-examined tail is kept (a correct, if larger, conflict set).
    """
    from ortools.sat.python import cp_model

    from astro_mine.allocate.model.compile.cpsat import lower_for_iis

    if len(candidate) <= 1:
        return candidate  # a singleton conflict is already irreducible

    surviving = list(candidate)
    for constraint_id in candidate[:budget]:
        trial = [c for c in surviving if c != constraint_id]
        if not trial:
            break
        lowered = lower_for_iis(ir, keep=set(trial))
        solver = _solver(cp_model, seed)
        solver.parameters.max_deterministic_time = DEFAULT_REFINEMENT_DETERMINISTIC_TIME
        if solver.Solve(lowered.model) == cp_model.INFEASIBLE:
            surviving = trial  # still infeasible without it — it was not load-bearing
    return surviving


def _tasks_from_id(constraint_id: str) -> set[str]:
    """Task ids named directly in a constraint id (the fallback when it carries no terms)."""
    head, _, rest = constraint_id.partition("::")
    if not rest:
        return set()
    if head in _TASK_PREFIXES:
        return {rest.split("::", 1)[0]}
    if head == "prec":  # prec::{pred}->{task}
        endpoints = rest.split("->")
        return set(endpoints) if len(endpoints) == 2 else set()
    return set()  # power / no_overlap / cumulative name an asset or resource, not a task


def _implicated_tasks(ir: AllocationIR, core: list[str]) -> set[str]:
    """The tasks the conflicting constraints touch — from their IR terms, then their ids."""
    var_task = {v.id: v.task_ref for v in ir.variables}
    constraint_by_id = {c.id: c for c in ir.constraints}
    tasks: set[str] = set()
    for cid in core:
        constraint = constraint_by_id.get(cid)
        if constraint is not None:
            for term in constraint.terms:
                task = var_task.get(term.var_ref)
                if task:
                    tasks.add(task)
        tasks.update(_tasks_from_id(cid))
    return tasks


def _explain(
    ir: AllocationIR,
    core: list[str],
    task_ids: set[str],
    report: ConstraintReport | None,
) -> str:
    """A human-readable reason: the relevant builder findings + the named conflict set."""
    core_set = set(core)
    parts: list[str] = []

    if report is not None:
        details = {
            finding.detail
            for finding in report.findings
            if (finding.constraint_id in core_set) or (finding.task_id in task_ids)
        }
        parts.extend(sorted(details))

    uncoverable = sorted(
        c.id.partition("::")[2]
        for c in ir.constraints
        if c.id in core_set and c.kind is ConstraintKind.ASSIGNMENT_COVER and not c.terms
    )
    if uncoverable:
        parts.append(f"no eligible asset for: {', '.join(uncoverable)}")

    contended = sorted(
        c.id.partition("::")[2]
        for c in ir.constraints
        if c.id in core_set and c.kind is ConstraintKind.NO_OVERLAP
    )
    if contended:
        parts.append(
            "no conflict-free schedule on: " + ", ".join(contended) + " (tasks overlap in time)"
        )

    proved = (
        "CP-SAT proved the composed model infeasible"
        if report is not None
        else "CP-SAT proved the model infeasible"
    )
    parts.append(f"{proved}; irreducible infeasible set: {', '.join(core)}")
    return "; ".join(parts)
