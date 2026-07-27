"""``AllocationPlanner`` — the Core allocation sub-interface implementation (RM-P1-ALLOC-01).

Allocate's realization of the Core :class:`~astro_mine.core.policy.protocol.Allocator`
sub-interface: a :class:`~astro_mine.core.policy.protocol.Policy` whose ``decide`` maps
observations + a :class:`~astro_mine.core.policy.model.DecisionContext` to an
:class:`~astro_mine.core.messages.model.ActionBatch` of scheduled ``TASK`` actions (core.md
``policy/protocol.py``; allocate.md §6). The allocation problem and its upstream constraint
truth (a :class:`~astro_mine.allocate.ConstraintContext`) ride in the Core-blessed
``DecisionContext.extras`` channel — never a sibling import.

Two pieces are fully implemented and tested here: the ``decide`` seam and the
**Allocation ↔ ActionBatch adapter** (:meth:`_to_action_batch`). :meth:`solve` is a
documented **trivial-feasible stub** for RM-P1-ALLOC-01 — a deterministic greedy assignment
that always returns a feasible plan (or an honest ``INFEASIBLE``) so the whole seam is
exercisable end to end; the real CP-SAT backend behind the same ``solve`` signature is
RM-P1-ALLOC-02, and the power/comms/terrain constraint builders it reads from the
``ConstraintContext`` are RM-P1-ALLOC-03.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

from astro_mine.allocate.anytime import finalize_status, stream_incumbents
from astro_mine.allocate.api._core import CORE_INTERFACES
from astro_mine.allocate.api.model import (
    Allocation,
    AllocationProvenance,
    AllocationRequest,
    AssetSchedule,
    ConstraintContext,
    InfeasibilityCertificate,
    ScheduledTask,
)
from astro_mine.allocate.constraints.compose import compile_with_constraints
from astro_mine.allocate.constraints.config import ConstraintConfig
from astro_mine.allocate.constraints.costs import CostTable
from astro_mine.allocate.constraints.result import ConstraintReport
from astro_mine.allocate.enums import AllocationStatus
from astro_mine.allocate.explain import (
    binding_constraints,
    decompose_objective,
    extract_iis,
)
from astro_mine.allocate.model.ir.compile import (
    Pair,
    compile_request,
    earliest_start_in_windows,
    window_envelope,
)
from astro_mine.allocate.model.ir.model import AllocationIR
from astro_mine.allocate.model.ir.utils import assignment_pairs
from astro_mine.allocate.model.ir.verify import verify_feasible
from astro_mine.allocate.solvers.base import Incumbent
from astro_mine.allocate.solvers.registry import (
    CPSAT_BACKEND,
    TRIVIAL_STUB_BACKEND,
    backend_provider,
    known_backends,
    resolve_solver,
)
from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.enums import ActionKind, TaskKind
from astro_mine.core.messages.model import Action, ActionBatch, Observation, TaskDirective
from astro_mine.core.policy.model import DecisionContext

__all__ = [
    "CONSTRAINT_CONFIG_KEY",
    "CONSTRAINT_CONTEXT_KEY",
    "COST_TABLE_KEY",
    "REQUEST_KEY",
    "AllocationPlanner",
]

#: Absolute floor for the greedy's budget/window float comparisons (matches the IR verifier).
_EPS = 1.0e-9

#: ``DecisionContext.extras`` key carrying the :class:`~astro_mine.allocate.AllocationRequest`
#: to solve this decision step (the blessed downstream-owned channel; core.md
#: ``policy/model.py``). Absent ⇒ nothing to allocate ⇒ an empty ``ActionBatch``.
REQUEST_KEY = "allocation.request"

#: ``DecisionContext.extras`` key carrying the :class:`~astro_mine.allocate.ConstraintContext`
#: of Core-typed upstream handles (Link/Worlds/Prospect/Fleet). Threaded through in
#: RM-P1-ALLOC-01; read by the physics constraint builders in RM-P1-ALLOC-03.
CONSTRAINT_CONTEXT_KEY = "allocation.constraint_context"

#: ``DecisionContext.extras`` key carrying an optional
#: :class:`~astro_mine.allocate.constraints.ConstraintConfig` (the declared modeling policy the
#: RM-P1-ALLOC-03 builders apply). Absent ⇒ the default policy.
CONSTRAINT_CONFIG_KEY = "allocation.constraint_config"

#: ``DecisionContext.extras`` key carrying an optional
#: :class:`~astro_mine.allocate.constraints.CostTable` (cached per-pair duration/energy costs).
#: Absent ⇒ an empty table (builders fall back to declared defaults, marking the build degraded).
COST_TABLE_KEY = "allocation.cost_table"

#: Backend id recorded in provenance for the RM-P1-ALLOC-01 stub (CP-SAT is RM-P1-ALLOC-02).
#: Aliases the registry's canonical constant — one spelling of the id, so a rename cannot leave
#: the planner's default pointing at a backend the registry no longer knows.
_STUB_BACKEND = TRIVIAL_STUB_BACKEND


def _topological_order(request: AllocationRequest) -> list[str]:
    """Task ids in a precedence-respecting order, ties broken by id (deterministic).

    The request validator guarantees the precedence relation is acyclic, so Kahn's
    algorithm consumes every task.
    """
    preds = {t.task_id: set(t.precedence) for t in request.tasks}
    order: list[str] = []
    ready = sorted(tid for tid, ps in preds.items() if not ps)
    while ready:
        tid = ready.pop(0)
        order.append(tid)
        for other, ps in preds.items():
            if tid in ps:
                ps.discard(tid)
                if not ps and other not in order and other not in ready:
                    ready.append(other)
        ready.sort()
    return order


class AllocationPlanner:
    """The allocation sub-interface: an :class:`~astro_mine.core.policy.protocol.Allocator`.

    Constructed with the solver ``backend`` id it records in provenance (the RM-P1-ALLOC-01
    stub; a CP-SAT-backed planner is RM-P1-ALLOC-02). Stateless and deterministic: the same
    request yields the same plan (allocate.md §8).
    """

    def __init__(self, *, backend: str = _STUB_BACKEND) -> None:
        # Validate against the registry at construction. An unresolvable id used to fall through
        # to the built-in greedy and still be stamped into `provenance.backend`, so a plan solved
        # by Allocate's own heuristic claimed a backend that never ran (RM-P1-ALLOC-07 —
        # "recorded seeds + pinned solver" is only worth anything if the record is true).
        if backend not in known_backends():
            raise ValueError(
                f"unknown solver backend {backend!r}; known backends: {known_backends()}"
            )
        self._backend = backend

    # --- Core Policy/Planner contract --------------------------------------------

    def decide(
        self, observations: Mapping[AgentId, Observation], context: DecisionContext
    ) -> ActionBatch:
        """Solve the request in ``context.extras`` and emit scheduled ``TASK`` actions.

        Reads the :class:`~astro_mine.allocate.AllocationRequest` (and the optional
        :class:`~astro_mine.allocate.ConstraintContext`) from the Core-blessed
        ``DecisionContext.extras`` channel, solves it, and adapts the plan into a
        Sim-consumable :class:`~astro_mine.core.messages.model.ActionBatch`. With no request
        to allocate this step, returns an empty batch (a valid no-op).
        """
        request = context.extras.get(REQUEST_KEY)
        if request is None:
            return ActionBatch(actions=[])
        if not isinstance(request, AllocationRequest):
            raise TypeError(
                f"DecisionContext.extras[{REQUEST_KEY!r}] must be an AllocationRequest, "
                f"got {type(request).__name__}"
            )
        constraint_context = context.extras.get(CONSTRAINT_CONTEXT_KEY)
        if constraint_context is not None and not isinstance(constraint_context, ConstraintContext):
            raise TypeError(
                f"DecisionContext.extras[{CONSTRAINT_CONTEXT_KEY!r}] must be a ConstraintContext, "
                f"got {type(constraint_context).__name__}"
            )
        allocation = self.solve(
            request,
            context=constraint_context,
            config=context.extras.get(CONSTRAINT_CONFIG_KEY),
            costs=context.extras.get(COST_TABLE_KEY),
        )
        return self._to_action_batch(allocation)

    # --- solve (RM-P1-ALLOC-01 trivial-feasible stub; CP-SAT is RM-P1-ALLOC-02) ---

    def solve(
        self,
        request: AllocationRequest,
        *,
        context: ConstraintContext | None = None,
        config: ConstraintConfig | None = None,
        costs: CostTable | None = None,
        hints: Mapping[str, float] | None = None,
    ) -> Allocation:
        """Return a feasible :class:`~astro_mine.allocate.Allocation` (or an honest INFEASIBLE).

        The ``backend`` this planner was constructed with selects the solve path:

        - ``"cp-sat"`` (RM-P1-ALLOC-02) runs the OR-Tools CP-SAT search behind the ``Solver``
          strategy, returning an **optimized** plan with a real ``optimality_gap`` (``OPTIMAL`` when
          proven), accepting solver ``hints`` (a warm start CP-SAT verifies) — over the augmented IR
          when a ``context`` is supplied, else the structural skeleton.
        - ``"trivial-stub"`` (default; RM-P1-ALLOC-01/03) runs the deterministic constraint-aware
          greedy: without a ``context`` the structural stub, with one the terrain/comms/power-aware
          greedy. It is *feasibility* only (``optimality_gap`` unknown) and ignores ``hints``.
        - **Any other id** is a plugin backend advertised under
          :data:`~astro_mine.allocate.solvers.registry.SOLVER_ENTRY_POINT_GROUP`. It is resolved
          through the registry and driven over the *same* incumbent stream as CP-SAT — a
          third-party backend is a first-class citizen, not a second tier.

        Every backend returns a feasible plan or an explicit
        :class:`~astro_mine.allocate.InfeasibilityCertificate`, and every emitted feasible plan is
        independently re-checked against the IR by
        :func:`~astro_mine.allocate.verify_feasible` — Allocate is not the safety authority
        (allocate.md §9), and that check is exactly what makes accepting a third-party solver safe.
        """
        if self._backend == TRIVIAL_STUB_BACKEND:
            if context is not None:
                return self._solve_constrained(request, context, config, costs)
            return self._solve_unconstrained(request)
        return self._solve_streaming(request, context, config, costs, hints)

    def solve_anytime(
        self,
        request: AllocationRequest,
        *,
        context: ConstraintContext | None = None,
        config: ConstraintConfig | None = None,
        costs: CostTable | None = None,
        hints: Mapping[str, float] | None = None,
    ) -> Iterator[Allocation]:
        """Stream feasible :class:`~astro_mine.allocate.Allocation`\\ s as CP-SAT improves.

        The anytime contract (RM-P1-ALLOC-05; allocate.md §2 principle 2): each yielded plan is a
        feasible incumbent with a monotonically improving bound and an explicit optimality gap, so
        a caller can stop at any deadline and take the best plan found. Every incumbent is
        independently re-checked by :func:`~astro_mine.allocate.verify_feasible`; the expensive
        binding-constraint / objective-decomposition explanation is attached only to the terminal
        plan (an intermediate incumbent is superseded before the deadline). The last item is the
        terminal outcome — ``OPTIMAL``/``FEASIBLE`` with a plan, or ``TIMEOUT``/``INFEASIBLE``
        (with a certificate) honestly. ``solve`` returns exactly this terminal item.

        The greedy ``trivial-stub`` has no incumbent trajectory to stream, so ``solve_anytime``
        drives the configured *solver* backend — CP-SAT, or a plugin advertised under
        :data:`~astro_mine.allocate.solvers.registry.SOLVER_ENTRY_POINT_GROUP`, which reaches the
        exact same stream (the ``hints`` warm start is the RM-P1-ALLOC-05 online re-solve seam,
        verified and never trusted by the exact layer).
        """
        yield from self._stream_backend(request, context, config, costs, hints)

    def _solve_unconstrained(self, request: AllocationRequest) -> Allocation:
        """The RM-P1-ALLOC-01 structural stub: greedy assignment over the skeleton IR only.

        Honors the two temporal families the skeleton carries: a task starts inside one of its
        (possibly disjoint) time windows — snapped *out of* an availability gap, never merely inside
        the bounding envelope — and an asset is never double-booked (its next task waits until it is
        free; RM-P1-ALLOC-02).
        """
        ir = compile_request(request)
        assets = sorted(request.assets, key=lambda a: a.asset_id)
        tasks_by_id = {t.task_id: t for t in request.tasks}

        assignment: dict[str, str] = {}
        starts: dict[str, float] = {}
        busy_until: dict[str, float] = {}

        for tid in _topological_order(request):
            task = tasks_by_id[tid]
            required = {str(c) for c in task.required_capabilities}
            not_before = max((starts[p] for p in task.precedence), default=0.0)

            placed: tuple[str, float] | None = None
            for asset in assets:
                if not required <= {str(c) for c in asset.capability_tags}:
                    continue
                start = earliest_start_in_windows(
                    task, max(not_before, busy_until.get(asset.asset_id, 0.0))
                )
                if start is None:
                    continue  # this asset is busy past every window that could host the task
                placed = (asset.asset_id, start)
                break

            if placed is None:
                return self._infeasible(request, ir)
            chosen, start = placed
            assignment[tid], starts[tid] = chosen, start
            busy_until[chosen] = start + task.duration_s

        plan = self._assemble_plan(
            request, assignment, starts, {t.task_id: t.duration_s for t in request.tasks}
        )
        return self._feasible_allocation(
            status=AllocationStatus.FEASIBLE,
            plan=plan,
            realized_objective=self._objective_value(ir, assignment),
            optimality_gap=None,
            ir=ir,
            provenance=self._provenance(request, ir),
        )

    def _assemble_plan(
        self,
        request: AllocationRequest,
        assignment: Mapping[str, str],
        starts: Mapping[str, float],
        durations: Mapping[str, float],
    ) -> list[AssetSchedule]:
        """Map a greedy assignment + starts onto per-asset, time-ordered, non-overlapping schedules.

        A scheduled task's ``end_s`` is its start plus the duration the *model* reserved for it, so
        the plan reports the occupancy the ``NO_OVERLAP`` constraint (and
        :func:`~astro_mine.allocate.verify_feasible`) actually reasons about — a schedule that
        under-reports its end would look conflict-free while double-booking the asset.
        """
        tasks_by_id = {t.task_id: t for t in request.tasks}
        by_asset: dict[str, list[ScheduledTask]] = {a.asset_id: [] for a in request.assets}
        for tid, asset_id in assignment.items():
            task = tasks_by_id[tid]
            by_asset[asset_id].append(
                ScheduledTask(
                    task_id=tid,
                    kind=task.kind,
                    start_s=starts[tid],
                    end_s=starts[tid] + durations.get(tid, 0.0),
                )
            )
        return [
            AssetSchedule(
                asset_id=asset_id, tasks=sorted(sts, key=lambda s: (s.start_s, s.task_id))
            )
            for asset_id, sts in sorted(by_asset.items())
        ]

    # --- constrained solve (RM-P1-ALLOC-03: terrain / comms / power builders) -----

    def _solve_constrained(
        self,
        request: AllocationRequest,
        context: ConstraintContext,
        config: ConstraintConfig | None,
        costs: CostTable | None,
    ) -> Allocation:
        """Greedy assignment honoring the terrain/comms/power constraints, verified against the IR.

        Compiles the augmented IR (:func:`compile_with_constraints`), then assigns each task — in
        precedence order — to the first eligible asset that is not kept out, has remaining energy
        budget, is **free** at the task's earliest start (the per-asset ``NO_OVERLAP`` constraint),
        and has a comms window compatible with it. The start is placed inside one of the task's
        (possibly disjoint) availability windows, never merely inside their bounding envelope. The
        finished plan is independently re-checked against the augmented IR: a plan that fails
        :func:`verify_feasible` is never returned as feasible (a compromised solver cannot smuggle
        an infeasible plan past the shield).
        """
        config = config or ConstraintConfig()
        costs = costs or CostTable()
        comp = compile_with_constraints(request, context, config=config, costs=costs)
        ir, report = comp.ir, comp.report
        extra_hashes = [config.content_hash(), costs.content_hash()]

        tasks_by_id = {t.task_id: t for t in request.tasks}
        pairs = assignment_pairs(ir)
        used_energy: dict[str, float] = {}
        busy_until: dict[str, float] = {}
        assignment: dict[str, str] = {}
        starts: dict[str, float] = {}
        chosen_durations: dict[str, float] = {}

        for tid in _topological_order(request):
            task = tasks_by_id[tid]
            _, base_hi = window_envelope(task)
            prec_lo = max((starts[p] for p in task.precedence), default=0.0)

            candidates = [a for a in pairs.get(tid, []) if (tid, a) not in report.forbidden]
            if not candidates:
                return self._infeasible_cert(
                    request,
                    ir,
                    report,
                    extra_hashes,
                    task_id=tid,
                    reason=f"every eligible asset for {tid} is kept out (terrain/comms)",
                )

            chosen: str | None = None
            for asset_id in candidates:
                cap = report.energy_capacity.get(asset_id)
                cost = report.energy_costs.get((tid, asset_id), 0.0)
                if cap is not None and used_energy.get(asset_id, 0.0) + cost > cap + _EPS:
                    continue
                win = report.pair_windows.get((tid, asset_id))
                not_before = max(prec_lo, busy_until.get(asset_id, 0.0), win[0] if win else 0.0)
                hi = min(base_hi, win[1]) if win else base_hi
                start = earliest_start_in_windows(task, not_before)
                if start is None or start > hi + _EPS:
                    continue
                duration = report.durations.get((tid, asset_id), 0.0)
                chosen, starts[tid], chosen_durations[tid] = asset_id, start, duration
                used_energy[asset_id] = used_energy.get(asset_id, 0.0) + cost
                busy_until[asset_id] = start + duration
                break

            if chosen is None:
                return self._infeasible_cert(
                    request,
                    ir,
                    report,
                    extra_hashes,
                    task_id=tid,
                    reason=f"no eligible asset for {tid} satisfies its energy budget, its comms "
                    "window, and the asset's remaining idle time",
                )
            assignment[tid] = chosen

        plan = self._assemble_plan(request, assignment, starts, chosen_durations)

        allocation = self._feasible_allocation(
            status=AllocationStatus.FEASIBLE,
            plan=plan,
            realized_objective=self._objective_value(ir, assignment),
            optimality_gap=None,
            ir=ir,
            provenance=self._provenance(request, ir, extra_hashes=extra_hashes),
        )
        # Independent re-check: never emit a feasible-labelled plan that does not verify.
        if not verify_feasible(allocation, ir):  # pragma: no cover - defensive; greedy is correct
            return self._infeasible_cert(
                request,
                ir,
                report,
                extra_hashes,
                task_id=None,
                reason="internal: greedy plan failed the independent IR feasibility re-check",
            )
        return allocation

    def _infeasible_cert(
        self,
        request: AllocationRequest,
        ir: AllocationIR,
        report: ConstraintReport,
        extra_hashes: list[str],
        *,
        task_id: str | None,
        reason: str,
    ) -> Allocation:
        """An INFEASIBLE result carrying an explicit certificate built from the builder findings.

        Names the conflicting constraints/tasks and joins their explanations (e.g. "no
        contact window long enough to relay the haul"). This is the RM-P1-ALLOC-03 structural
        certificate; the full
        irreducible-infeasible-set is RM-P1-ALLOC-06 (allocate.md §10).
        """
        relevant = [f for f in report.findings if task_id is None or f.task_id == task_id]
        constraint_ids = sorted({f.constraint_id for f in relevant if f.constraint_id})
        task_ids = sorted(
            {f.task_id for f in relevant if f.task_id} | ({task_id} if task_id else set())
        )
        details = [f.detail for f in relevant]
        details.append(reason)
        return Allocation(
            status=AllocationStatus.INFEASIBLE,
            plan=None,
            realized_objective=None,
            provenance=self._provenance(request, ir, extra_hashes=extra_hashes),
            infeasibility_certificate=InfeasibilityCertificate(
                constraint_ids=constraint_ids,
                task_ids=task_ids,
                explanation="; ".join(details),
            ),
        )

    # --- solver-backed solve (RM-P1-ALLOC-02: CP-SAT and any plugin, one Solver strategy) ---

    def _backend_version(self) -> str:
        """The pinned solver version string this planner's backend records in provenance.

        CP-SAT pins OR-Tools; a plugin backend pins its own distribution where one can be
        resolved, so "which solver produced this plan" stays answerable for a third-party backend
        too (RM-P1-ALLOC-07; allocate.md §5). Falls back to the Allocate version alone only when
        the plugin advertises no resolvable distribution."""
        if self._backend == CPSAT_BACKEND:
            return _cpsat_backend_version()
        provider = backend_provider(self._backend) or f"plugin {self._backend} (provider unknown)"
        return f"{provider}; astro-mine-allocate {_allocate_version()}"

    def _solve_streaming(
        self,
        request: AllocationRequest,
        context: ConstraintContext | None,
        config: ConstraintConfig | None,
        costs: CostTable | None,
        hints: Mapping[str, float] | None,
    ) -> Allocation:
        """Run the anytime solver stream and return its terminal ``Allocation``.

        ``solve`` semantics are unchanged: it consumes the same incumbent stream
        :meth:`solve_anytime` exposes and returns the best plan at the deadline (the last item),
        so a caller that does not care about the trajectory keeps the one-shot contract.
        """
        return list(self._stream_backend(request, context, config, costs, hints))[-1]

    def _stream_backend(
        self,
        request: AllocationRequest,
        context: ConstraintContext | None,
        config: ConstraintConfig | None,
        costs: CostTable | None,
        hints: Mapping[str, float] | None,
    ) -> Iterator[Allocation]:
        """Compile the (augmented or skeleton) IR, run CP-SAT, stream one Allocation per incumbent.

        Bounds are clamped monotone and gaps recomputed by the anytime tracker; the terminal
        incumbent carries the full explanation, an unresolved terminal maps to an honest
        ``TIMEOUT``/``INFEASIBLE``. Every feasible incumbent is re-checked against the same IR by
        :func:`verify_feasible` before it is labelled feasible — a solver bug can never smuggle an
        infeasible or mis-scored plan past the independent shield (allocate.md §9).
        """
        if context is not None:
            config = config or ConstraintConfig()
            costs = costs or CostTable()
            comp = compile_with_constraints(request, context, config=config, costs=costs)
            ir, report = comp.ir, comp.report
            durations = dict(report.durations)
            extra_hashes = [config.content_hash(), costs.content_hash()]
        else:
            # The skeleton reserves each task's *declared nominal* duration; the decoded plan must
            # report that same occupancy, or its end times would contradict the NO_OVERLAP intervals
            # the solver actually scheduled against.
            ir, report, extra_hashes = compile_request(request), None, []
            durations = _nominal_durations(request)

        task_kinds = {t.task_id: t.kind for t in request.tasks}
        solver = resolve_solver(self._backend, task_kinds=task_kinds, durations=durations)
        incumbents = list(
            stream_incumbents(solver, ir, request.budget, ir.objective_sense, hints=hints)
        )
        prov = self._provenance(
            request,
            ir,
            extra_hashes=extra_hashes,
            backend_version=self._backend_version(),
            budget_consumed_s=getattr(solver, "last_wall_time_s", None),
        )
        for index, incumbent in enumerate(incumbents):
            yield self._incumbent_allocation(
                request, ir, report, prov, incumbent, terminal=index == len(incumbents) - 1
            )

    def _incumbent_allocation(
        self,
        request: AllocationRequest,
        ir: AllocationIR,
        report: ConstraintReport | None,
        prov: AllocationProvenance,
        incumbent: Incumbent,
        *,
        terminal: bool,
    ) -> Allocation:
        """Map one tracked incumbent onto a Core-typed :class:`~astro_mine.allocate.Allocation`."""
        if not incumbent.is_feasible:
            return self._cpsat_unsolved(request, ir, report, prov, incumbent.status)

        allocation = self._feasible_allocation(
            status=incumbent.status,
            plan=incumbent.plan,
            realized_objective=incumbent.objective,
            optimality_gap=incumbent.gap,
            ir=ir,
            provenance=prov,
            explain=terminal,  # defer the expensive explanation to the plan taken at the deadline
        )
        if not verify_feasible(
            allocation, ir
        ):  # pragma: no cover - defensive; CP-SAT + verify agree
            return Allocation(
                status=AllocationStatus.INFEASIBLE,
                plan=None,
                realized_objective=None,
                provenance=prov,
                infeasibility_certificate=InfeasibilityCertificate(
                    explanation="internal: CP-SAT plan failed the independent IR feasibility check"
                ),
            )
        return allocation

    def _cpsat_unsolved(
        self,
        request: AllocationRequest,
        ir: AllocationIR,
        report: ConstraintReport | None,
        prov: AllocationProvenance,
        status: AllocationStatus,
    ) -> Allocation:
        """A CP-SAT solve that returned no usable plan, mapped to an honest anytime outcome.

        A proven-infeasible model is ``INFEASIBLE`` with an IIS certificate; a search that ran out
        of budget with no incumbent and no infeasibility proof is an explicit ``TIMEOUT``
        (:func:`~astro_mine.allocate.anytime.finalize_status`) — never a plan, never a false
        certificate (a late optimal answer is a wrong answer in operations, principle 1)."""
        final = finalize_status(status)
        certificate = (
            self._cpsat_certificate(request, ir, report)
            if final is AllocationStatus.INFEASIBLE
            else None
        )
        return Allocation(
            status=final,
            plan=None,
            realized_objective=None,
            provenance=prov,
            infeasibility_certificate=certificate,
        )

    @staticmethod
    def _cpsat_certificate(
        request: AllocationRequest, ir: AllocationIR, report: ConstraintReport | None
    ) -> InfeasibilityCertificate:
        """The irreducible-infeasible-set certificate for a CP-SAT INFEASIBLE result (ALLOC-06).

        Extracts the IIS through CP-SAT's assumption machinery (:func:`extract_iis`), naming the
        *minimal* conflicting constraints/tasks — not a bare "infeasible" — and joining the
        RM-P1-ALLOC-03 builder findings as the human explanation when a constrained compile
        supplied a report. Deterministic (single worker + the request seed + sorted output) so the
        same infeasible model yields the same certificate (allocate.md §10; RM-P1-ALLOC-07)."""
        return extract_iis(ir, seed=request.budget.seed, report=report)

    # --- Allocation -> ActionBatch adapter (fully implemented + tested) -----------

    @staticmethod
    def _to_action_batch(allocation: Allocation) -> ActionBatch:
        """Map a plan into an ``ActionBatch`` of scheduled ``TASK`` actions (deterministic).

        Each :class:`~astro_mine.allocate.ScheduledTask` becomes one
        :class:`~astro_mine.core.messages.model.Action` of kind ``TASK`` for its asset, in
        asset-then-start order. The directive is a **``CUSTOM`` assignment directive** — the
        allocator decides *who does what, when* (the assignment), while the concrete
        motion-feasible task payload (a typed goto/excavate/haul body) is the tactical
        :class:`~astro_mine.core.policy.protocol.TaskMotionPlanner`'s job downstream (core.md
        ``policy/protocol.py``). The intended Core task kind and the schedule ride in the
        directive ``params`` so the assignment is self-describing. An empty/absent plan yields
        an empty batch.
        """
        actions: list[Action] = []
        for asset_schedule in allocation.plan or []:
            for st in asset_schedule.tasks:
                directive = TaskDirective(
                    task_kind=TaskKind.CUSTOM,
                    deadline_s=st.end_s,
                    directive=st.task_id,
                    params={
                        "allocated_kind": st.kind.value,
                        "asset_id": asset_schedule.asset_id,
                        "start_s": repr(st.start_s),
                        "end_s": repr(st.end_s),
                    },
                )
                actions.append(
                    Action(
                        agent_id=asset_schedule.asset_id,
                        kind=ActionKind.TASK,
                        sim_time_s=st.start_s,
                        task=directive,
                    )
                )
        return ActionBatch(actions=actions)

    # --- helpers -----------------------------------------------------------------

    @staticmethod
    def _feasible_allocation(
        *,
        status: AllocationStatus,
        plan: list[AssetSchedule],
        realized_objective: float,
        optimality_gap: float | None,
        ir: AllocationIR,
        provenance: AllocationProvenance,
        explain: bool = True,
    ) -> Allocation:
        """A feasible :class:`~astro_mine.allocate.Allocation` with its explanation attached.

        The single builder every feasible planner path routes through, so a plan's binding
        constraints (which comms window / power floor / slope limit is tight) and objective
        decomposition (the info-gain-vs-ROI split) are computed once, over the solver-neutral IR,
        from the same variable-value mapping the verifier uses (RM-P1-ALLOC-06; allocate.md §10).
        ``explain=False`` skips the explanation for a streamed intermediate incumbent, which is
        superseded before the deadline (RM-P1-ALLOC-05).
        """
        return Allocation(
            status=status,
            plan=plan,
            realized_objective=realized_objective,
            optimality_gap=optimality_gap,
            binding_constraints=binding_constraints(plan, ir) if explain else [],
            objective_decomposition=decompose_objective(plan, ir) if explain else None,
            provenance=provenance,
            infeasibility_certificate=None,
        )

    def _objective_value(self, ir: AllocationIR, assignment: Mapping[str, str]) -> float:
        """Realized objective ``sum(coefficient * variable_value)`` over the IR objective terms.

        Computed from the *IR* (single source of truth) so it is exactly the value
        :func:`~astro_mine.allocate.verify_feasible` re-derives — the two never disagree.
        """
        var_by_id = {v.id: v for v in ir.variables}
        total = 0.0
        for term in ir.objective_terms:
            var = var_by_id[term.var_ref]
            placed = var.task_ref is not None and assignment.get(var.task_ref) == var.asset_ref
            total += term.coefficient * (1.0 if placed else 0.0)
        return total

    def _provenance(
        self,
        request: AllocationRequest,
        ir: AllocationIR,
        *,
        extra_hashes: list[str] | None = None,
        backend_version: str | None = None,
        budget_consumed_s: float | None = 0.0,
    ) -> AllocationProvenance:
        # The constrained path pins the config + cost-table hashes alongside the request so a plan
        # reproduces exactly the modeling policy and cost inputs it was compiled under. The CP-SAT
        # path additionally pins the pinned OR-Tools version and the wall-clock it consumed.
        return AllocationProvenance(
            input_hashes=[request.content_hash(), *(extra_hashes or [])],
            ir_version=ir.ir_version,
            backend=self._backend,
            backend_version=backend_version or _allocate_version(),
            seed=request.budget.seed,
            budget_consumed_s=budget_consumed_s,
            core_interface_versions=dict(CORE_INTERFACES),
        )

    def _infeasible(self, request: AllocationRequest, ir: AllocationIR) -> Allocation:
        # The infeasibility certificate slot is reserved here; the IIS is populated by
        # RM-P1-ALLOC-06 (allocate.md §10).
        return Allocation(
            status=AllocationStatus.INFEASIBLE,
            plan=None,
            realized_objective=None,
            provenance=self._provenance(request, ir),
            infeasibility_certificate=None,
        )


def _nominal_durations(request: AllocationRequest) -> dict[Pair, float]:
    """Each ``(task, asset)`` pair's interval length from the task's declared nominal duration.

    The skeleton's counterpart to the constrained compile's resolved per-pair durations: a task's
    ``duration_s`` occupies whichever asset takes it, so a decoded plan's ``end_s`` matches the
    interval the ``NO_OVERLAP`` constraint reserved. Zero-duration (point) tasks are omitted.
    """
    return {
        (task.task_id, asset.asset_id): task.duration_s
        for task in request.tasks
        if task.duration_s > 0.0
        for asset in request.assets
    }


def _allocate_version() -> str:
    from astro_mine.allocate import __version__

    return __version__


def _cpsat_backend_version() -> str:
    """The pinned CP-SAT backend version string recorded in a plan's provenance.

    The OR-Tools version (the solver whose determinism the golden gate depends on) plus the
    Allocate version — so a plan pins the exact solver + compiler it was produced against
    (allocate.md §5; conventions.md §5)."""
    from importlib.metadata import version

    return f"ortools {version('ortools')}; astro-mine-allocate {_allocate_version()}"
