# SPDX-License-Identifier: Apache-2.0
"""The evaluation-batch orchestrator — plan → enforce → dispatch → collect (RM-P1-BENCH-11).

:func:`run_evaluation_batch` is the top-level scale-out entry point: it plans the fan-out
(:func:`~astro_mine.bench.eval._plan.plan_batch`), enforces Bench's two rails — **fair-share
back-pressure** on the submission queue (Cloud's ``QueueAdmission``) and **hard per-submission
compute budgets** (Cloud's ``BudgetLedger``, charged per seed; ``BudgetExceeded`` halts the sweep) —
dispatches each submission through a :class:`~astro_mine.bench.eval._dispatch.BatchDispatcher`,
emits a run-completion event per seed on Cloud's ``astro-mine.cloud.runs`` subject, and collects the
MCAP + Parquet into ranked leaderboard submissions (bench.md §6, §7, §8).

The async lifecycle is honoured but not required: on a real deployment the completion events flow
over NATS/JetStream and a consumer triggers ingestion; in CI/local ``submit()`` (via the
``DryRunClient``) returns the ``RunResult`` synchronously, so the collector ingests directly — the
NATS consumer is deployment-only plumbing (bench.md §6). ``astro_mine.cloud`` is imported lazily
(the ``[cloud]`` extra); Bench never imports Sim.

Backlog: RM-P1-BENCH-11 — astro-mine-bench#19
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from astro_mine.bench.eval._collect import collect_submission
from astro_mine.bench.eval._plan import (
    DEFAULT_HOURS_PER_SEED,
    DEFAULT_RUNNER,
    REFERENCE_ROLLOUT_IMAGE,
    EvaluationTarget,
    plan_batch,
)

if TYPE_CHECKING:
    from astro_mine.bench.eval._dispatch import BatchDispatcher
    from astro_mine.bench.leaderboard._models import Submission
    from astro_mine.bench.leaderboard._store import LeaderboardStore
    from astro_mine.bench.scenario import ScenarioSpec
    from astro_mine.cloud.runs import EventPublisher
    from astro_mine.cloud.sched import CostRates
    from astro_mine.core.artifacts import ArtifactStore

__all__ = ["AdmissionDenied", "assert_batch_reproducible", "run_evaluation_batch"]


class AdmissionDenied(RuntimeError):
    """Raised when a submission cannot be admitted within its tenant's fair-share quota.

    The back-pressure signal (bench.md §8): the tenant is at quota, so the work waits rather than
    starving other tenants. In the synchronous CI path this surfaces as an exception the caller
    re-queues on; in a deployment it is the Kueue admission gate.
    """


def run_evaluation_batch(
    spec: ScenarioSpec,
    targets: Sequence[EvaluationTarget],
    *,
    seeds: Sequence[int],
    dispatcher: BatchDispatcher,
    leaderboard_store: LeaderboardStore,
    artifact_store: ArtifactStore,
    publisher: EventPublisher | None = None,
    quotas: Mapping[str, Mapping[str, float]] | None = None,
    max_parallel: int | None = None,
    hours_per_seed: float = DEFAULT_HOURS_PER_SEED,
    cost_rates: CostRates | None = None,
    spot: bool = True,
    base_env: Mapping[str, str] | None = None,
    runner: str = DEFAULT_RUNNER,
) -> list[Submission]:
    """Run a scale-out evaluation of ``targets`` over ``seeds`` and rank the results.

    Plans one sweep per target, enforces fair-share admission (if ``quotas`` given) and
    per-submission budgets, dispatches through ``dispatcher``, emits completion events on
    ``publisher`` (default: Cloud's ``NullPublisher``), collects each submission's MCAP/Parquet, and
    persists it to ``leaderboard_store``. Returns the persisted submissions in plan order.

    ``runner`` names the runner each fanned-out worker rolls with (default: the trace fixture). Its
    resolved id is stamped on every collected scorecard, so a Sim-rolled batch is distinguishable
    from a fixture one by provenance and not only by value (bench#64).

    Raises :class:`AdmissionDenied` when a submission exceeds its tenant's quota, and propagates
    Cloud's ``BudgetExceeded`` when a submission would exceed its compute cap — halting the batch so
    no submission runs past its budget.
    """
    # --- why Bench knows about Cloud's scheduling at all (platform#5 §2, third bullet) ---
    #
    # The question that issue asks is whether `EventPublisher`, `BudgetLedger` and `CostRates`
    # belong at the waist, or whether Bench should not know about them at all. The answer is
    # neither, and the reasoning is worth keeping next to the code rather than in a merged PR:
    #
    # *They do not move to Core.* `EventPublisher` is a Protocol with users in two components, so
    # §3.3 would put it at the waist — except that it names `CompletionEvent`, and that names
    # `RunStatus`. Moving the protocol drags Cloud's whole run-lifecycle vocabulary into the narrow
    # waist to buy one annotation, and core.md §2 principle 1 prices that correctly: every addition
    # to Core is permanent. `CostRates` and `BudgetLedger` are worse candidates still — what a
    # GPU-hour costs and how a budget is charged are scheduling *policy*, which §2.2 keeps out of
    # Core by construction.
    #
    # *Bench already inverts the two that matter.* `publisher` and `cost_rates` arrive as injected
    # parameters and default to nothing; every import of them is `TYPE_CHECKING` or deferred, so
    # the local tier runs with no bus, no budget and no cluster, and Bench pays nothing at import
    # time (§8). That is the shape §3.2 rule 4 describes, not a defect.
    #
    # *One genuine remainder.* The two constructions below are Bench enforcing admission and
    # charging budget — scheduling, done by the benchmark, around Cloud's primitives rather than
    # by Cloud. The fix is to hand this function an already-configured dispatcher that enforces
    # both, which is a restructure of the eval loop (AdmissionDenied ordering, BudgetExceeded
    # halting) rather than a move, and it is filed as its own issue rather than smuggled into a
    # refactor whose whole claim is that every step was a move plus a signature change.
    from astro_mine.cloud.runs import NullPublisher, emit_completion
    from astro_mine.cloud.sched import BudgetLedger, QueueAdmission

    sink: EventPublisher = publisher if publisher is not None else NullPublisher()
    plan = plan_batch(
        spec,
        targets,
        seeds=seeds,
        max_parallel=max_parallel,
        hours_per_seed=hours_per_seed,
        cost_rates=cost_rates,
        spot=spot,
        base_env=base_env,
        runner=runner,
    )
    ledger = BudgetLedger(plan.budget_caps) if plan.budget_caps else None
    admission = QueueAdmission(quotas) if quotas else None

    submissions: list[Submission] = []
    for planned in plan.evaluations:
        if admission is not None and not admission.admit(planned.tenant, planned.admission_request):
            raise AdmissionDenied(
                f"{planned.request.policy_ref!r} exceeds tenant {planned.tenant!r} quota "
                f"{planned.admission_request} — queued (back-pressure)"
            )
        try:
            if ledger is not None and planned.cap_key is not None:
                for _ in planned.seeds:
                    ledger.charge(planned.cap_key, planned.cost_per_seed)

            results = dispatcher.dispatch(planned, store=artifact_store)
            for result in results:
                emit_completion(
                    sink,
                    result.run_context,
                    "completed" if result.ok else "failed",
                    tenant=planned.tenant,
                )
            submission = collect_submission(
                spec, planned.request, results, artifact_store=artifact_store
            )
            leaderboard_store.add_submission(submission)
            submissions.append(submission)
        finally:
            if admission is not None:
                admission.release(planned.tenant, planned.admission_request)

    return submissions


def assert_batch_reproducible(
    spec: ScenarioSpec,
    *,
    policy_ref: str,
    seeds: Sequence[int],
    image: str = REFERENCE_ROLLOUT_IMAGE,
    method: str | None = None,
) -> str:
    """The scale-out determinism gate: a Cloud-collected scorecard equals the workstation run.

    Fans ``policy_ref`` over ``seeds`` through Cloud's ``DryRunClient`` — which compiles the real
    Argo/KubeRay manifest **and** executes locally — collects the MCAP/Parquet, and asserts the
    collected submission's scorecard hash is byte-identical to the in-process
    :func:`~astro_mine.bench.baseline.run` scorecard. This is the RM-P1-BENCH-11 acceptance ("a
    cluster run reproduces the workstation run for the same inputs + seed") as a CI-runnable gate.
    Raises :class:`~astro_mine.bench.harness.DeterminismError` on drift; returns the shared hash.
    """
    import tempfile

    from astro_mine.bench.baseline import run
    from astro_mine.bench.eval._dispatch import CloudBatchDispatcher
    from astro_mine.bench.harness import DeterminismError
    from astro_mine.bench.leaderboard._eval import resolve_policy
    from astro_mine.bench.leaderboard._models import SubmissionRequest
    from astro_mine.bench.leaderboard._store import InMemoryStore
    from astro_mine.cloud.artifacts.store import FilesystemArtifactStore
    from astro_mine.cloud.submission.cluster import DryRunClient

    seed_list = tuple(seeds)
    local_hash = run(spec, resolve_policy(policy_ref), seeds=seed_list).content_hash
    request = SubmissionRequest(scenario_id=spec.scenario_id, policy_ref=policy_ref, method=method)
    target = EvaluationTarget(request=request, image=image)
    with tempfile.TemporaryDirectory(prefix="astro-mine-eval-gate-") as tmp:
        collected = run_evaluation_batch(
            spec,
            [target],
            seeds=seed_list,
            dispatcher=CloudBatchDispatcher(client=DryRunClient()),
            leaderboard_store=InMemoryStore(),
            artifact_store=FilesystemArtifactStore(root=tmp),
        )[0].scorecard_hash

    if collected != local_hash:
        raise DeterminismError(
            f"{spec.scenario_id!r} eval batch did not reproduce the local run: "
            f"{collected} != {local_hash}"
        )
    return collected
