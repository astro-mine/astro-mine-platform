"""Real Kueue back-pressure: a tenant at quota *waits*, it does not overrun.

``sched/kueue.QueueAdmission`` is an in-process dict of counters. It has never been checked
against Kueue -- and until now it could not have been, because nothing wired a compiled Job to a
``LocalQueue`` at all: there was no ``kueue.x-k8s.io/queue-name`` label anywhere in the repo, so
Kueue never saw a single Astro-Mine object and the whole quota model was decorative.

``compile_job`` now labels a tenant-scoped job with its LocalQueue and compiles it **suspended**;
Kueue un-suspends it when the tenant is within quota. This asserts the consequence: with a
one-slot quota, the second job stays suspended until the first releases -- and then runs
(``cloud.md`` §2 principle 9, §9: degrade, don't collapse).
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from astro_mine.cloud.engines import compile_job
from astro_mine.cloud.k8s import LABEL_QUEUE_NAME, to_yaml
from astro_mine.cloud.packaging import ImageRef
from astro_mine.cloud.sched.kueue import cluster_queue, resource_flavor
from astro_mine.cloud.submission import JobSpec, ResourceRequest
from astro_mine.cloud.tenancy.namespace import tenant_manifests, tenant_namespace
from tests.cloud.cluster.conftest import CLUSTER_MARKS, Kubectl

pytestmark = CLUSTER_MARKS

TENANT = "backpressure"
FLAVOR = "backpressure-flavor"
COHORT = "backpressure"
#: One slot: the first job takes it, the second must wait for it.
QUOTA_CPU = "500m"
#: The ClusterQueue MUST cover memory as well as CPU, even though the jobs only ask for CPU.
#:
#: `tenant_manifests` ships a LimitRange whose `defaultRequest` is `{cpu: 100m, memory: 128Mi}`, so
#: Kubernetes injects a memory request into every container that does not declare one. Kueue will
#: not admit a workload whose resources its ClusterQueue does not cover, so a queue quoted in CPU
#: alone admits *nothing*: both workloads sat Pending with "couldn't assign flavors to pod set main:
#: resource memory unavailable in ClusterQueue", and the back-pressure under test never ran.
#:
#: This is a real trap in the library's tenancy builders, not a quirk of this test: a tenant built
#: entirely from `tenant_manifests` + `cluster_queue` is unusable with Kueue unless the quota covers
#: every resource the LimitRange can inject. Generous on purpose -- this bounds nothing the test
#: asserts, which is CPU back-pressure.
QUOTA_MEMORY = "4Gi"
HOLD_SECONDS = 45


@pytest.fixture(scope="module")
def tenant(kubectl: Kubectl) -> Iterator[str]:
    """A tenant with a one-slot ClusterQueue, built by the library's own manifest builders."""
    ns = tenant_namespace(TENANT)
    kubectl.apply(
        to_yaml(
            [
                resource_flavor(FLAVOR),
                cluster_queue(
                    TENANT,
                    cohort=COHORT,
                    quotas={"cpu": QUOTA_CPU, "memory": QUOTA_MEMORY},
                    flavor=FLAVOR,
                    allow_borrowing=False,  # a hard ceiling: nothing to borrow from
                ),
                *tenant_manifests(
                    TENANT, quota={"cpu": "4", "memory": "8Gi"}, cluster_queue=TENANT
                ),
            ]
        )
    )
    # Kueue must have accepted the ClusterQueue before anything can be admitted through it.
    kubectl("wait", "--for=condition=Active", f"clusterqueue/{TENANT}", "--timeout=120s")
    try:
        yield ns
    finally:
        kubectl("delete", "namespace", ns, "--wait=false", check=False)
        kubectl("delete", "clusterqueue", TENANT, check=False)
        kubectl("delete", "resourceflavor", FLAVOR, check=False)


def _job(image: ImageRef, env: dict[str, str], *, seed: int) -> JobSpec:
    return JobSpec(
        image=image,
        # Hold the quota slot long enough for the second job to be observably queued.
        command=["python", "-c", f"import time; time.sleep({HOLD_SECONDS}); print('done')"],
        env=env,
        seed=seed,
        tenant=TENANT,
        resource_request=ResourceRequest(cpu=QUOTA_CPU),
    )


def _suspended(kubectl: Kubectl, ns: str, name: str) -> bool:
    return bool(kubectl.json("get", "job", name, "-n", ns)["spec"].get("suspend"))


def test_a_tenant_at_quota_queues_instead_of_overrunning(
    kubectl: Kubectl,
    tenant: str,
    workload_image: ImageRef,
    pod_store_env: dict[str, str],
) -> None:
    first = _job(workload_image, pod_store_env, seed=1)
    second = _job(workload_image, pod_store_env, seed=2)
    names = [compile_job(j, namespace=tenant)["metadata"]["name"] for j in (first, second)]

    # Both are born suspended and queue-labelled -- that is what puts them under Kueue at all.
    for job, name in zip((first, second), names, strict=True):
        manifest = compile_job(job, namespace=tenant)
        assert manifest["spec"]["suspend"] is True
        assert manifest["metadata"]["labels"][LABEL_QUEUE_NAME] == tenant
        kubectl.apply(to_yaml(manifest))
        assert name  # (the compiled name is what we poll on below)

    # Exactly ONE is admitted; the other is held at quota. Which one is Kueue's business, not ours:
    # it does not promise to admit in submission order, and naming the winner in advance is how this
    # test failed while the back-pressure it tests was working perfectly. (It waited for names[0],
    # which Kueue had queued *second*; by the time that was admitted the other had long since run,
    # finished, and been un-suspended -- so "the second job is still suspended" was false, and the
    # test reported an absence of back-pressure that the cluster had in fact applied.)
    def _admitted() -> list[str]:
        return [n for n in names if not _suspended(kubectl, tenant, n)]

    kubectl.wait_until(lambda: len(_admitted()) >= 1, timeout=180.0)
    running = _admitted()
    assert len(running) == 1, (
        f"both jobs were admitted while the tenant was at quota ({running}) -- "
        "there is no back-pressure"
    )
    (queued,) = [n for n in names if n not in running]

    # ...and once the admitted one releases its slot, the queued one runs rather than starving.
    kubectl.wait_until(lambda: not _suspended(kubectl, tenant, queued), timeout=300.0, interval=5.0)
    names = [running[0], queued]  # (the completion wait below polls the one that had to wait)
    kubectl("wait", "--for=condition=Complete", f"job/{names[1]}", "-n", tenant, "--timeout=300s")
