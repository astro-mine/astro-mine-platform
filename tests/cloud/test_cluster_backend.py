"""The cluster backend: the call site, and KubectlClusterClient's result collection.

The dispatch *logic* -- apply, wait for a terminal state, scan the pods for the harness
sentinels, load the RunContext back out of the shared store -- is driven here against a fake
``KubectlRunner``, so everything except the ten-line subprocess shim is covered with no cluster.
What a live cluster adds is checked by the opt-in ``cluster``-marked tests in ``tests/cluster/``.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from astro_mine.cloud.artifacts.store import FilesystemArtifactStore
from astro_mine.cloud.packaging import ImageRef
from astro_mine.cloud.runs.events import CollectingPublisher, RunObserver
from astro_mine.cloud.submission import (
    ClusterBackend,
    ClusterDispatchError,
    DryRunClient,
    JobSpec,
    registered_backends,
    submit,
)
from astro_mine.cloud.submission.backend import register_backend
from astro_mine.cloud.submission.cluster import CommandResult, KubectlClusterClient
from astro_mine.cloud.submission.harness import EXIT_CODE_SENTINEL, RUN_CONTEXT_SENTINEL
from astro_mine.cloud.submission.harness import run as run_harness
from astro_mine.cloud.submission.local import LocalBackend

IMAGE = ImageRef.parse("ghcr.io/astro-mine/astro-mine-bench@sha256:" + "78" * 32)
_WORKLOAD = (
    "import os, pathlib;"
    "o=pathlib.Path(os.environ['ASTRO_MINE_OUTPUTS']);"
    "(o/'y.txt').write_text(os.environ['ASTRO_MINE_SEED'])"
)
POD = "pod/amc-job-x-abcde"


def _job(**overrides: object) -> JobSpec:
    job = JobSpec(
        image=IMAGE, command=[sys.executable, "-c", _WORKLOAD], outputs=["y.txt"], seed=42
    )
    return job.model_copy(update=overrides) if overrides else job


class FakeKubectl:
    """A scripted ``kubectl``: records every argv, answers status/pod/log queries from a plan.

    *statuses* is the sequence of status stdouts successive polls see (the last one repeats),
    *pods* the pod names a selector resolves to, *logs* each pod's log text.
    """

    def __init__(
        self,
        *,
        statuses: Sequence[str] = ("Complete=True;",),
        pods: Sequence[str] = (POD,),
        logs: dict[str, str] | None = None,
        apply_returncode: int = 0,
    ) -> None:
        self.argvs: list[list[str]] = []
        self.stdin: bytes | None = None
        self.polls = 0
        self._statuses = list(statuses)
        self._pods = list(pods)
        self._logs = logs or {}
        self._apply_returncode = apply_returncode

    def run(self, argv: Sequence[str], *, stdin: bytes | None = None) -> CommandResult:
        self.argvs.append(list(argv))
        verb = argv[1]
        if verb == "apply":
            self.stdin = stdin
            return CommandResult(self._apply_returncode, "the server rejected the manifest")
        if verb == "logs":
            return CommandResult(0, self._logs.get(argv[4], ""))
        if verb == "get" and argv[2] == "pods":
            return CommandResult(0, "\n".join(self._pods))
        status = self._statuses[min(self.polls, len(self._statuses) - 1)]  # then hold on the last
        self.polls += 1
        return CommandResult(0, status)


def _sentinels(address: str, exit_code: int = 0) -> str:
    return f"starting\n{RUN_CONTEXT_SENTINEL}{address}\n{EXIT_CODE_SENTINEL}{exit_code}\n"


def _be_the_pod(store: FilesystemArtifactStore, job: JobSpec) -> str:
    """Run *job* through the in-pod harness against *store* and return the logs it would print.

    This is exactly what a real workload container does -- the harness writes the outputs and the
    RunContext into the shared store, then prints the sentinels. The fake kubectl serves those
    logs back, so the client under test collects a *real* run's provenance, not a stub's.
    """
    return _sentinels(run_harness(job, store).run_context_address)


def _client(runner: FakeKubectl, *, timeout: float = 900.0) -> KubectlClusterClient:
    # No real sleeping and a clock we control, so poll/timeout tests are instant.
    ticks = iter(range(10_000))
    return KubectlClusterClient(
        runner=runner,
        timeout=timeout,
        poll_interval=0.0,
        sleep=lambda _seconds: None,
        monotonic=lambda: float(next(ticks)),
    )


# --- registration + the DryRun (no-cluster) path -------------------------------------------


def test_cluster_is_registered_with_a_real_default_client() -> None:
    assert "cluster" in registered_backends()
    assert isinstance(ClusterBackend()._client, KubectlClusterClient)


def test_dry_run_records_a_manifest_embedding_digest_and_seed(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    client = DryRunClient()
    ClusterBackend(client=client).run(_job(), store=store)
    manifest = client.dispatched[0]
    assert manifest["kind"] == "Job"
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    assert container["image"] == IMAGE.reference
    seed_env = next(e for e in container["env"] if e["name"] == "ASTRO_MINE_SEED")
    assert seed_env["value"] == "42"


def test_dry_run_reruns_locally_so_it_cannot_prove_cluster_equivalence(tmp_path: Path) -> None:
    """DryRunClient *is* the local backend -- it never leaves the workstation.

    Equal results here therefore say nothing about a cluster (that is ``tests/cluster/``'s job).
    What they do prove is that the ``backend="cluster"`` call site compiles and runs a JobSpec.
    """
    store = FilesystemArtifactStore(tmp_path)
    job = _job()
    local = LocalBackend().run(job, store=store)
    cluster = ClusterBackend(client=DryRunClient()).run(job, store=store)
    assert cluster.ok
    assert local.outputs == cluster.outputs


def test_engine_override_selects_ray(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    client = DryRunClient()
    ClusterBackend(client=client, engine="ray").run(_job(), store=store)
    assert client.dispatched[0]["kind"] == "RayJob"


def test_same_submit_call_site_runs_local_and_cluster(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    job = _job()
    register_backend("cluster-dryrun", ClusterBackend(client=DryRunClient()), replace=True)
    local = submit(job, store=store)
    cluster = submit(job, backend="cluster-dryrun", store=store)
    assert local.run_context.content_address() == cluster.run_context.content_address()


# --- applying ------------------------------------------------------------------------------


def test_dispatch_applies_the_rendered_manifest_on_stdin(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    job = _job()
    runner = FakeKubectl(logs={POD: _be_the_pod(store, job)})

    ClusterBackend(client=_client(runner), namespace="astro-mine").run(job, store=store)

    assert runner.argvs[0] == ["kubectl", "apply", "-f", "-"]
    assert runner.stdin is not None
    manifest = runner.stdin.decode()
    assert "kind: Job" in manifest
    assert IMAGE.reference in manifest


def test_a_failed_apply_raises_rather_than_waiting_forever(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    runner = FakeKubectl(apply_returncode=1)
    with pytest.raises(ClusterDispatchError, match="kubectl apply failed"):
        ClusterBackend(client=_client(runner)).run(_job(), store=store)


# --- waiting -------------------------------------------------------------------------------


def test_dispatch_polls_until_the_job_is_complete(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    job = _job()
    runner = FakeKubectl(
        statuses=["", "Complete=False;", "Complete=True;"], logs={POD: _be_the_pod(store, job)}
    )

    result = ClusterBackend(client=_client(runner)).run(job, store=store)

    assert result.ok
    assert runner.polls == 3  # it really waited


def test_dispatch_polls_a_rayjob_on_its_entrypoint_status(tmp_path: Path) -> None:
    """`.status.jobStatus` is the *entrypoint's* outcome; the RayCluster is ready long before."""
    store = FilesystemArtifactStore(tmp_path)
    job = _job(distributed=True)
    submitter = "pod/amc-ray-x-submitter"
    runner = FakeKubectl(
        statuses=["PENDING", "RUNNING", "SUCCEEDED"],
        pods=[submitter],
        logs={submitter: _be_the_pod(store, job)},
    )

    result = ClusterBackend(client=_client(runner)).run(job, store=store)

    assert result.ok
    status_argv = next(a for a in runner.argvs if a[1] == "get" and a[2] == "rayjob")
    assert status_argv[-1] == "jsonpath={.status.jobStatus}"


@pytest.mark.parametrize(
    ("distributed", "status"),
    [(False, "Failed=True;"), (True, "FAILED"), (True, "STOPPED")],
)
def test_a_failed_object_is_still_collected(tmp_path: Path, distributed: bool, status: str) -> None:
    """A failed run is a *result*, not an exception -- the pod still recorded its provenance."""
    store = FilesystemArtifactStore(tmp_path)
    job = _job(distributed=distributed, command=[sys.executable, "-c", "import sys; sys.exit(3)"])
    logs = _sentinels(run_harness(job, store).run_context_address, exit_code=3)
    runner = FakeKubectl(statuses=[status], logs={POD: logs})

    result = ClusterBackend(client=_client(runner)).run(job, store=store)

    assert not result.ok
    assert result.status == "failed"
    assert result.exit_code == 3
    assert result.outputs == {}


def test_a_job_that_never_finishes_times_out(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    runner = FakeKubectl(statuses=[""])  # forever pending
    with pytest.raises(ClusterDispatchError, match="timed out"):
        ClusterBackend(client=_client(runner, timeout=3.0)).run(_job(), store=store)


# --- collecting the run's real result -------------------------------------------------------


def test_the_collected_result_is_the_pods_own_run_context(tmp_path: Path) -> None:
    """Nothing is recomputed host-side: the RunContext is loaded back from the shared store."""
    store = FilesystemArtifactStore(tmp_path)
    job = _job()
    in_pod = run_harness(job, store)  # be the pod
    runner = FakeKubectl(logs={POD: _sentinels(in_pod.run_context_address)})

    result = ClusterBackend(client=_client(runner)).run(job, store=store)

    assert result.run_context_address == in_pod.run_context_address
    assert result.run_context == in_pod.run_context
    assert result.outputs == in_pod.outputs
    assert store.get(result.outputs["y.txt"]) == b"42"


def test_a_resumed_run_collects_the_attempt_that_finished(tmp_path: Path) -> None:
    """The killed pod printed nothing; the retry that completed the job printed the sentinels.

    kubectl is asked for the pods oldest-first, so the collection that survives is the *last* one
    to report -- exactly the semantics checkpoint-resume needs.
    """
    store = FilesystemArtifactStore(tmp_path)
    job = _job()
    runner = FakeKubectl(
        pods=["pod/amc-job-x-killed", "pod/amc-job-x-retry"],
        logs={
            "pod/amc-job-x-killed": "step 1\nstep 2\n",
            "pod/amc-job-x-retry": _be_the_pod(store, job),
        },
    )

    result = ClusterBackend(client=_client(runner)).run(job, store=store)

    assert result.ok
    pod_query = next(a for a in runner.argvs if a[1] == "get" and a[2] == "pods")
    assert "--sort-by=.metadata.creationTimestamp" in pod_query  # oldest-first is load-bearing


def test_a_run_with_no_sentinel_is_a_loud_error_not_a_silent_pass(tmp_path: Path) -> None:
    """An image that is not running the harness must never look like a successful run."""
    store = FilesystemArtifactStore(tmp_path)
    runner = FakeKubectl(logs={POD: "hello from some other image\n"})
    with pytest.raises(ClusterDispatchError, match="no run-context sentinel"):
        ClusterBackend(client=_client(runner)).run(_job(), store=store)


def test_unreadable_pods_and_logs_do_not_crash_the_collector(tmp_path: Path) -> None:
    """A force-deleted pod has no logs left to read -- that is the resume case, not a failure."""
    store = FilesystemArtifactStore(tmp_path)
    job = _job(distributed=True)
    submitter = "pod/amc-ray-x-submitter"

    class Flaky(FakeKubectl):
        def run(self, argv: Sequence[str], *, stdin: bytes | None = None) -> CommandResult:
            if argv[1] == "logs" and argv[4] == "pod/gone":
                return CommandResult(1, "Error from server (NotFound): pods not found")
            if argv[1] == "get" and argv[2] == "pods" and argv[6].startswith("ray.io/"):
                return CommandResult(1, "error: no matching resources")
            return super().run(argv, stdin=stdin)

    runner = Flaky(
        statuses=["SUCCEEDED"],
        pods=["pod/gone", submitter],
        logs={submitter: _be_the_pod(store, job)},
    )

    assert ClusterBackend(client=_client(runner)).run(job, store=store).ok


# --- lifecycle events -----------------------------------------------------------------------


def test_dispatch_emits_started_then_completed_under_one_run_pin(tmp_path: Path) -> None:
    store = FilesystemArtifactStore(tmp_path)
    job = _job()
    runner = FakeKubectl(logs={POD: _be_the_pod(store, job)})
    publisher = CollectingPublisher()

    ClusterBackend(client=_client(runner)).run(
        job, store=store, observer=RunObserver(publisher=publisher)
    )

    assert [event.status for _subject, event in publisher.events] == ["started", "completed"]
    # one run identity across the lifecycle, even though the pod's own observer is inert
    assert len({event.run_address for _subject, event in publisher.events}) == 1
