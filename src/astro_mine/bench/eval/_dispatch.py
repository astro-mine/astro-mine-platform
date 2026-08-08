"""The dispatch seam — hand a planned evaluation to Cloud for execution (RM-P1-BENCH-11).

:class:`BatchDispatcher` is the Core/Cloud-typed seam Bench dispatches through: a
:class:`~astro_mine.bench.eval._plan.PlannedEvaluation` in, a list of Cloud
:class:`~astro_mine.cloud.submission.result.RunResult` out. Two implementations ship:

- :class:`CloudBatchDispatcher` runs the sweep through Cloud's ``ClusterBackend``: CPU seeds compile
  to an **Argo** fan-out ``Workflow`` (recorded on :attr:`~CloudBatchDispatcher.sweep_manifests`,
  ``max_parallel`` = back-pressure), and every seed is dispatched via an injectable
  :class:`~astro_mine.cloud.submission.cluster.ClusterClient` — GPU rollouts route to **KubeRay**,
  CPU seeds to a plain **K8s Job** (``select_engine``). Injecting Cloud's ``DryRunClient``
  compiles the *real* manifest **and** runs the job locally, returning a byte-identical RunResult —
  the no-cluster CI executor that proves a cluster run reproduces the workstation run (bench.md §7).
- :class:`LocalBatchDispatcher` is the minimal fake: each seed runs through Cloud's ``LocalBackend``
  directly, no manifest compilation — the sacred single-workstation tier (conventions.md §7).

``astro_mine.cloud`` is imported **lazily** (behind the ``[cloud]`` extra) so the base package stays
dependency-clean (core + pydantic) and Bench still never imports Sim.

Backlog: RM-P1-BENCH-11 — astro-mine-bench#19
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from astro_mine.bench.eval._plan import PlannedEvaluation
    from astro_mine.cloud.k8s import Manifest
    from astro_mine.cloud.submission.cluster import ClusterClient
    from astro_mine.cloud.submission.result import RunResult
    from astro_mine.core.artifacts import ArtifactStore

__all__ = ["BatchDispatcher", "CloudBatchDispatcher", "LocalBatchDispatcher"]

#: Cloud's execution namespace for compiled manifests (Bench's evaluation tenant on the cluster).
DEFAULT_NAMESPACE = "bench"


@runtime_checkable
class BatchDispatcher(Protocol):
    """Executes a planned evaluation's seeds and returns one RunResult per seed."""

    def dispatch(self, planned: PlannedEvaluation, *, store: ArtifactStore) -> list[RunResult]: ...


class CloudBatchDispatcher:
    """Dispatch a sweep to Cloud: Argo fan-out for CPU seeds, KubeRay for GPU rollouts.

    *client* is Cloud's :class:`~astro_mine.cloud.submission.cluster.ClusterClient` dispatch seam;
    it defaults to :class:`~astro_mine.cloud.submission.cluster.DryRunClient`, which compiles the
    real per-seed manifest and executes locally (the no-cluster path). Pass
    :class:`~astro_mine.cloud.submission.cluster.KubectlClusterClient` (opt-in ``cluster`` tests) to
    apply to a live cluster.
    """

    def __init__(self, *, client: ClusterClient | None = None, namespace: str = DEFAULT_NAMESPACE):
        from astro_mine.cloud.submission.cluster import DryRunClient

        self._client: ClusterClient = client if client is not None else DryRunClient()
        self._namespace = namespace
        #: The Argo fan-out ``Workflow`` compiled per CPU evaluation (the DAG + parallelism cap).
        self.sweep_manifests: list[Manifest] = []

    @property
    def client(self) -> ClusterClient:
        """The injected dispatch client (a ``DryRunClient`` records per-seed manifests on it)."""
        return self._client

    def dispatch(self, planned: PlannedEvaluation, *, store: ArtifactStore) -> list[RunResult]:
        """Run every seed of *planned* via Cloud's ``ClusterBackend`` and return the RunResults.

        CPU evaluations additionally record the whole-sweep Argo ``Workflow`` (the fan-out shape and
        ``max_parallel`` back-pressure); GPU rollouts route each seed to a KubeRay ``RayJob`` via
        ``select_engine`` (``distributed=True``). Byte-identical to a local run under a dry run.
        """
        from astro_mine.cloud.engines import compile_sweep
        from astro_mine.cloud.submission.cluster import ClusterBackend

        if not planned.distributed:
            self.sweep_manifests.append(compile_sweep(planned.sweep, namespace=self._namespace))
        backend = ClusterBackend(client=self._client, namespace=self._namespace)
        return [backend.run(job, store=store) for job in planned.sweep.expand()]


class LocalBatchDispatcher:
    """The minimal CI fake: run every seed through Cloud's ``LocalBackend`` (no manifests)."""

    def dispatch(self, planned: PlannedEvaluation, *, store: ArtifactStore) -> list[RunResult]:
        from astro_mine.cloud.submission.local import LocalBackend

        backend = LocalBackend()
        return [backend.run(job, store=store) for job in planned.sweep.expand()]


