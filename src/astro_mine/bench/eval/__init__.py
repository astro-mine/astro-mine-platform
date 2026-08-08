"""Scale-out evaluation on Cloud — fan seeds x submissions out and collect (RM-P1-BENCH-11).

Bench *composes*, it owns neither the compute fabric ([Cloud](cloud.md)) nor the simulator
([Sim](sim.md)) (bench.md §2.2): this package **dispatches** evaluation batches to Cloud — which
executes Sim rollouts as Cloud-scheduled images — and collects the resulting **MCAP** traces and
**Parquet** metrics into the leaderboard, so the public leaderboard (RM-P1-BENCH-10) scales to
community volume (bench.md §7, §12). The pipeline:

- :func:`plan_batch` — one Cloud ``SweepSpec`` per submission: CPU seeds fan out via **Argo**, GPU
  rollouts route to **Ray/KubeRay** with a MIG ``ResourceRequest`` (:mod:`._plan`);
- :class:`BatchDispatcher` — the dispatch seam; :class:`CloudBatchDispatcher` runs it through Cloud
  ``ClusterBackend`` (inject ``DryRunClient`` for the no-cluster CI path),
  :class:`LocalBatchDispatcher` is the minimal local fake (:mod:`._dispatch`);
- :func:`run_worker` — the single-seed rollout argv Cloud runs, wired to the ``eval-worker`` CLI
  subcommand (:mod:`._worker`);
- :func:`collect_submission` — pull each seed's MCAP/Parquet back by content address and aggregate
  into a byte-identical :class:`~astro_mine.bench.metrics.Scorecard` → leaderboard
  :class:`~astro_mine.bench.leaderboard._models.Submission` (:mod:`._collect`);
- :func:`run_evaluation_batch` — the orchestrator: plan → fair-share admission + per-submission
  budgets → dispatch → completion events → collect → rank (:mod:`._batch`).

The dependency-clean surface (planner, seam, collector, orchestration) imports only ``core +
pydantic``; ``astro_mine.cloud`` and ``pyarrow`` ride the ``[cloud]`` extra and are imported lazily,
and Bench still **never imports Sim**.

Backlog: RM-P1-BENCH-11 — astro-mine-bench#19
"""

from __future__ import annotations

from astro_mine.bench.eval._batch import (
    AdmissionDenied,
    assert_batch_reproducible,
    run_evaluation_batch,
)
from astro_mine.bench.eval._collect import collect_submission, read_metrics_parquet
from astro_mine.bench.eval._dispatch import (
    BatchDispatcher,
    CloudBatchDispatcher,
    LocalBatchDispatcher,
)
from astro_mine.bench.eval._plan import (
    DEFAULT_HOURS_PER_SEED,
    METRICS_OUTPUT,
    REFERENCE_ROLLOUT_IMAGE,
    SEED_ENV,
    TRACE_OUTPUT,
    EvaluationPlan,
    EvaluationTarget,
    PlannedEvaluation,
    plan_batch,
)
from astro_mine.bench.eval._worker import run_worker

__all__ = [
    "DEFAULT_HOURS_PER_SEED",
    "METRICS_OUTPUT",
    "REFERENCE_ROLLOUT_IMAGE",
    "SEED_ENV",
    "TRACE_OUTPUT",
    "AdmissionDenied",
    "BatchDispatcher",
    "CloudBatchDispatcher",
    "EvaluationPlan",
    "EvaluationTarget",
    "LocalBatchDispatcher",
    "PlannedEvaluation",
    "assert_batch_reproducible",
    "collect_submission",
    "plan_batch",
    "read_metrics_parquet",
    "run_evaluation_batch",
    "run_worker",
]
