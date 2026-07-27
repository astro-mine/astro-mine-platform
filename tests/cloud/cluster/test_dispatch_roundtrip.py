"""A real K8s Job and a real RayJob, dispatched through the same ``submit()`` a laptop uses.

The call site here is character-for-character the one in ``tests/test_submit_local.py`` -- only
``backend=`` differs. That is the ``cloud.md`` §2 principle 2 claim, and until now nothing
executed it: ``KubectlClusterClient.dispatch`` applied the manifest and then raised
``NotImplementedError``.

``distributed`` routes the engine (``engines/selection.py``): ``False`` -> a plain Job, ``True``
-> a KubeRay RayJob. Both are dispatched, waited on, and *collected* -- the RunResult that comes
back is the pod's own, loaded from the shared object store by content address.
"""

from __future__ import annotations

import pytest

from astro_mine.cloud.artifacts.runcontext import RunContext
from astro_mine.cloud.artifacts.s3 import S3ArtifactStore
from astro_mine.cloud.engines import HARNESS_COMMAND, compile_job
from astro_mine.cloud.packaging import ImageRef
from astro_mine.cloud.submission import ClusterBackend, JobSpec, submit
from astro_mine.cloud.submission.backend import register_backend
from tests.cloud.cluster import workloads
from tests.cloud.cluster.conftest import CLUSTER_MARKS, NAMESPACE, Kubectl

pytestmark = CLUSTER_MARKS

SEED = 7
X = 6


@pytest.fixture(scope="module", autouse=True)
def _cluster_backend() -> None:
    """Register the live cluster backend under the name the tests submit to."""
    register_backend("live", ClusterBackend(namespace=NAMESPACE), replace=True)


def _job(
    image: ImageRef, store: S3ArtifactStore, env: dict[str, str], *, distributed: bool
) -> JobSpec:
    return JobSpec(
        image=image,
        command=["python", "-c", workloads.DETERMINISTIC],
        env=env,
        inputs={"x.txt": store.put(str(X).encode())},
        outputs=["y.txt"],
        seed=SEED,
        distributed=distributed,
    )


@pytest.mark.parametrize(
    ("distributed", "kind"), [(False, "Job"), (True, "RayJob")], ids=["k8sjob", "rayjob"]
)
def test_submit_dispatches_and_collects_a_real_cluster_run(
    workload_image: ImageRef,
    store: S3ArtifactStore,
    pod_store_env: dict[str, str],
    distributed: bool,
    kind: str,
) -> None:
    job = _job(workload_image, store, pod_store_env, distributed=distributed)

    result = submit(job, backend="live", store=store)  # the laptop call site, one word changed

    assert result.ok, f"{kind} did not succeed: exit {result.exit_code}"
    assert result.exit_code == 0

    # The output the *pod* produced, captured back into the shared store by content address.
    expected = workloads.deterministic_output(X, SEED)
    assert store.get(result.outputs["y.txt"]) == expected

    # And the provenance envelope the pod wrote -- reloaded from the store, not reconstructed.
    assert RunContext.load(store, result.run_context_address) == result.run_context
    assert result.run_context.image_digest == workload_image.reference
    assert result.run_context.seed == SEED
    assert result.run_context.outputs == result.outputs


def test_a_failing_workload_comes_back_as_a_failed_result_not_a_hang(
    workload_image: ImageRef, store: S3ArtifactStore, pod_store_env: dict[str, str]
) -> None:
    """A nonzero exit is a *result*: the pod still recorded provenance, the Job still terminated."""
    job = JobSpec(
        image=workload_image,
        command=["python", "-c", "import sys; sys.exit(3)"],
        env=pod_store_env,
        seed=SEED,
    )

    result = submit(job, backend="live", store=store)

    assert not result.ok
    assert result.status == "failed"
    assert result.exit_code == 3
    assert result.outputs == {}


def test_the_compiled_job_really_runs_the_harness(
    kubectl: Kubectl,
    workload_image: ImageRef,
    store: S3ArtifactStore,
    pod_store_env: dict[str, str],
) -> None:
    """The API server holds the object we think it does: harness command, digest-pinned image.

    A Kubernetes ``command`` overrides the image ENTRYPOINT, so if this were ``job.command`` the
    pod would run the workload *directly* -- unstaged, uncaptured, unrecorded. Asserting it on the
    *server's* copy (not the compiled dict) is the point.
    """
    job = _job(workload_image, store, pod_store_env, distributed=False)
    result = submit(job, backend="live", store=store)
    assert result.ok

    name = compile_job(job, namespace=NAMESPACE)["metadata"]["name"]
    served = kubectl.json("get", "job", name, "-n", NAMESPACE)
    container = served["spec"]["template"]["spec"]["containers"][0]
    assert container["command"] == HARNESS_COMMAND
    assert container["image"] == workload_image.reference
    assert "@sha256:" in container["image"]
