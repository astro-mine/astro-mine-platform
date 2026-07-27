"""Chaos: kill the pod (and then the node) mid-run; checkpoint-resume must still finish the run.

``autoscale/checkpoint.py`` models a spot eviction with a ``Preempted`` exception raised by its
own loop -- a simulation of the failure, in the same process, on the happy path. Nothing had ever
taken a real pod away from a real workload. These tests do, and assert the two things that
actually matter (``cloud.md`` §2 principle 5, §8):

  1. the Job **completes** -- Kubernetes retried it, because ``compile_job`` now gives a
     checkpointing job a ``backoffLimit`` (it did not before: the budget was hardcoded to 0, so a
     killed pod ended the run); and
  2. the resumed run's output is **identical** to the uninterrupted run's -- resume loses at most
     one interval and reproduces byte-for-byte, or "reproducible" was never true.

The workload checkpoints a hash chain into the shared object store by content address, so the
test can watch its progress from outside the cluster (probe for the address of step *k*) and can
predict its final output without trusting the run that produced it. See ``workloads.py``.
"""

from __future__ import annotations

import threading

import pytest

from astro_mine.cloud.artifacts.addressing import content_address
from astro_mine.cloud.artifacts.s3 import S3ArtifactStore
from astro_mine.cloud.engines import compile_job
from astro_mine.cloud.packaging import ImageRef
from astro_mine.cloud.submission import (
    CheckpointPolicy,
    ClusterBackend,
    JobSpec,
    KubectlClusterClient,
    RunResult,
    submit,
)
from astro_mine.cloud.submission.backend import register_backend
from tests.cloud.cluster import workloads
from tests.cloud.cluster.conftest import CLUSTER_MARKS, NAMESPACE, Kubectl, run

pytestmark = CLUSTER_MARKS

STEPS = 12
STEP_SECONDS = "2"
#: Kill after this step has been committed -- far enough in that resume has real work to skip,
#: early enough that the retry has real work left to do.
KILL_AFTER_STEP = 3
#: Distinct seeds keep the two tests' checkpoint chains disjoint in the shared store. Sharing them
#: would let the second test "resume" straight to the last step of the first -- and pass without
#: ever running, let alone being interrupted. (It also gives the two Jobs distinct deterministic
#: names, so the second `kubectl apply` creates an object instead of no-op'ing over a finished one.)
POD_KILL_SEED = 101
NODE_KILL_SEED = 202


@pytest.fixture(scope="module", autouse=True)
def _cluster_backend() -> None:
    # A generous wait: a killed pod must be noticed, rescheduled, and re-run to completion.
    register_backend(
        "live",
        ClusterBackend(namespace=NAMESPACE, client=KubectlClusterClient(timeout=1800.0)),
        replace=True,
    )


def _job(image: ImageRef, env: dict[str, str], *, seed: int) -> JobSpec:
    return JobSpec(
        image=image,
        command=["python", "-c", workloads.CHECKPOINTED],
        env={**env, "WORKLOAD_STEPS": str(STEPS), "WORKLOAD_STEP_SECONDS": STEP_SECONDS},
        outputs=["state.txt"],
        seed=seed,
        # This is what earns the Job a retry budget + a podFailurePolicy that does not charge it
        # for a disruption. Without it the backoffLimit is 0 and the killed pod ends the run.
        checkpoint=CheckpointPolicy(interval_seconds=1),
    )


def _await_progress(store: S3ArtifactStore, kubectl: Kubectl, *, seed: int, step: int) -> None:
    """Wait until the workload has committed *step*'s checkpoint into the shared store.

    Progress is observed through the store, not the logs: a checkpoint's address is a pure
    function of (seed, step), so the host can simply ask whether it exists yet.
    """
    address = content_address(workloads.chain_state(seed, step))
    kubectl.wait_until(lambda: store.exists(address), timeout=240.0, interval=2.0)


def _pods(kubectl: Kubectl, job_name: str) -> list[str]:
    listed = kubectl("get", "pods", "-n", NAMESPACE, "-l", f"job-name={job_name}", "-o", "name")
    return listed.stdout.split()


def _node_ready(kubectl: Kubectl, node: str) -> bool | None:
    """``True``/``False`` for the node's Ready condition, ``None`` when it is Unknown or absent.

    A node whose kubelet has stopped reporting goes to Ready=**Unknown**, not Ready=False -- which
    is why "wait for not-ready" has to be a poll over three states rather than a match on one.
    """
    conditions = kubectl.json("get", "node", node)["status"]["conditions"]
    status = next((c["status"] for c in conditions if c["type"] == "Ready"), None)
    return {"True": True, "False": False}.get(str(status))


class _Submission(threading.Thread):
    """Runs ``submit()`` off the main thread and *keeps its exception*.

    A bare ``Thread(target=lambda: result.append(submit(...)))`` drops anything ``submit()`` raises:
    the list stays empty, the test dies on ``result[0]`` with an ``IndexError``, and the real
    failure -- the one worth reading -- is gone. Re-raise it on the main thread instead.
    """

    def __init__(self, job: JobSpec, store: S3ArtifactStore) -> None:
        super().__init__()
        self._job = job
        self._store = store
        self.result: RunResult | None = None
        self.error: BaseException | None = None

    def run(self) -> None:
        try:
            self.result = submit(self._job, backend="live", store=self._store)
        except BaseException as exc:
            self.error = exc

    def outcome(self) -> RunResult:
        if self.error is not None:
            raise AssertionError("submit() raised while the chaos was in flight") from self.error
        assert self.result is not None, "submit() neither returned nor raised"
        return self.result


def _assert_resumed_and_reproduced(result: RunResult, store: S3ArtifactStore, *, seed: int) -> None:
    assert result.ok, f"the interrupted job never completed (exit {result.exit_code})"
    expected = workloads.chain_state(seed, STEPS)
    assert store.get(result.outputs["state.txt"]) == expected
    # the resumed run's output address equals the uninterrupted run's -- computed independently
    assert result.outputs["state.txt"] == content_address(expected)


def test_a_pod_killed_mid_run_is_retried_and_resumes_from_its_last_checkpoint(
    kubectl: Kubectl,
    workload_image: ImageRef,
    store: S3ArtifactStore,
    pod_store_env: dict[str, str],
) -> None:
    """The primary chaos test: fast, deterministic, and it exercises the whole resume path."""
    job = _job(workload_image, pod_store_env, seed=POD_KILL_SEED)
    name = compile_job(job, namespace=NAMESPACE)["metadata"]["name"]

    killed: str | None = None
    thread = _Submission(job, store)
    thread.start()
    try:
        _await_progress(store, kubectl, seed=POD_KILL_SEED, step=KILL_AFTER_STEP)
        victims = _pods(kubectl, name)
        assert victims, "the job never scheduled a pod"
        killed = victims[0]
        kubectl("delete", killed, "-n", NAMESPACE, "--force", "--grace-period=0")
    finally:
        thread.join(timeout=1800)

    assert not thread.is_alive(), "submit() never returned after the pod was killed"
    _assert_resumed_and_reproduced(thread.outcome(), store, seed=POD_KILL_SEED)

    # It really *was* interrupted -- a green test here must not be a test that never killed
    # anything.
    served = kubectl.json("get", "job", name, "-n", NAMESPACE)
    assert served["status"].get("failed", 0) >= 1, "no pod failure was recorded; nothing was killed"

    # ...and a *replacement* is what finished it. Counting pods cannot show that:
    # `--force --grace-period=0` drops the victim from the API immediately, so the replacement is
    # the only pod left and a `>= 2` count can never hold. Compare identities instead.
    survivors = _pods(kubectl, name)
    assert survivors, "no pod remains; the job scheduled no replacement"
    assert killed not in survivors, "the killed pod is still listed; nothing was replaced"


@pytest.mark.slow
def test_losing_a_whole_node_mid_run_still_completes(
    kubectl: Kubectl,
    workload_image: ImageRef,
    store: S3ArtifactStore,
    pod_store_env: dict[str, str],
) -> None:
    """True node loss -- the spot-eviction shape -- by killing the kind worker container itself.

    Slower and rougher than a pod kill (the control plane must notice the node is gone before it
    reschedules), which is why the pod kill above is the primary test. The cluster has two workers
    precisely so one can be taken away.
    """
    job = _job(workload_image, pod_store_env, seed=NODE_KILL_SEED)
    name = compile_job(job, namespace=NAMESPACE)["metadata"]["name"]

    thread = _Submission(job, store)
    thread.start()
    node: str | None = None
    try:
        _await_progress(store, kubectl, seed=NODE_KILL_SEED, step=KILL_AFTER_STEP)
        pods = _pods(kubectl, name)
        assert pods
        node = kubectl.json("get", pods[0], "-n", NAMESPACE)["spec"]["nodeName"]
        assert node is not None and "control-plane" not in node, f"refusing to kill {node}"
        killed = run(["docker", "kill", node])
        assert killed.returncode == 0, killed.stderr

        # Wait for the control plane to actually notice, so the kill is a *node* loss and not a
        # race we won before anyone looked.
        #
        # Polled, not `kubectl wait`: the node's Ready condition flips to Unknown (not False) when
        # the kubelet simply stops reporting, so `--for=condition=Ready=false` never matches, sits
        # there for its full 300s, and collides with the subprocess timeout -- which is how this
        # arrived as a bare TimeoutExpired. "Not Ready" is the condition; "False" is only one of
        # the two ways to be it.
        kubectl.wait_until(
            lambda: _node_ready(kubectl, node) is not True, timeout=300.0, interval=5.0
        )

        # Then give the node back -- while submit() is still waiting, which is the whole point.
        #
        # A pod on an unreachable node is not deleted: it sits Unknown, and the Job keeps counting
        # it as active, so the Job cannot reach Complete however many replacements succeed. Holding
        # the node down for the entire wait therefore deadlocks by construction -- the run we are
        # blocking on can never finish, and submit() times out having proven nothing. (It did: 30
        # minutes, with a healthy replacement pod sitting Completed beside the stuck one.)
        #
        # A preempted spot node comes back. So does this one. The workload's container died with
        # the node either way, so the retry and the resume are real -- that is what is under test,
        # not Kubernetes' behaviour toward permanently absent hardware.
        run(["docker", "start", node])
        kubectl("wait", "--for=condition=Ready", f"node/{node}", "--timeout=300s", check=False)
    finally:
        thread.join(timeout=1800)
        if node is not None:  # belt and braces: the tests that follow need every node back
            run(["docker", "start", node])
            kubectl("wait", "--for=condition=Ready", f"node/{node}", "--timeout=300s", check=False)

    assert not thread.is_alive(), "submit() never returned after the node was killed"
    _assert_resumed_and_reproduced(thread.outcome(), store, seed=NODE_KILL_SEED)
