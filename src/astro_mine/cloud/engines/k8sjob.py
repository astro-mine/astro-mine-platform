# SPDX-License-Identifier: Apache-2.0
"""The plain-Kubernetes-Job engine -- trivial one-shot containers.

Compiles a :class:`~astro_mine.cloud.submission.jobspec.JobSpec` to a ``batch/v1`` Job (or an
Indexed Job for a fixed fan-out), the right shape for a trivial one-shot container that needs
no gang scheduling or actor topology (``cloud.md`` §2 principle 3, §4).

Two policies the JobSpec has always carried become *real Kubernetes behaviour* here, instead of
staying in-process models nothing enforces:

- **``checkpoint``** (:class:`~astro_mine.cloud.submission.jobspec.CheckpointPolicy`) makes the
  Job resumable. ``backoffLimit`` stays 0 by default -- a silent retry could change a
  nondeterministic outcome, and Cloud treats "reproduces only sometimes" as a bug (§2 principle
  4) -- but a job that checkpoints has *made itself* deterministic under retry: it resumes from
  its last content-addressed checkpoint and reproduces the uninterrupted result byte-for-byte
  (§8). So it earns a retry budget, plus a ``podFailurePolicy`` that does not *charge* that
  budget for a disruption, which is the failure it exists to absorb.
- **``tenant``** puts the Job under Kueue; see :func:`~astro_mine.cloud.engines.base.queue_labels`.

Backlog: RM-P1-CLOUD-01/02/03 -- astro-mine-cloud#12
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_mine.cloud.engines.base import (
    container_spec,
    deterministic_name,
    input_annotations,
    queue_labels,
    register_engine,
)
from astro_mine.cloud.k8s import Manifest, object_meta

if TYPE_CHECKING:
    from astro_mine.cloud.submission.jobspec import JobSpec

__all__ = ["RESUME_BACKOFF_LIMIT", "K8sJobEngine", "compile_job", "resume_backoff_limit"]

#: Retries granted to a job that checkpoints and resumes -- enough to ride out a couple of spot
#: evictions, small enough that a genuinely broken job still fails fast.
RESUME_BACKOFF_LIMIT = 3

#: Do not charge the retry budget for a pod killed by a *disruption* (preemption, node drain,
#: eviction): that is precisely the failure checkpoint-resume absorbs, so it must not consume the
#: budget reserved for real failures (``cloud.md`` §8). Requires ``restartPolicy: Never``.
_IGNORE_DISRUPTION: Manifest = {
    "rules": [
        {"action": "Ignore", "onPodConditions": [{"type": "DisruptionTarget", "status": "True"}]}
    ]
}


def resume_backoff_limit(job: JobSpec) -> int:
    """The Job's ``backoffLimit``: a retry budget iff *job* checkpoints and resumes, else 0."""
    policy = job.checkpoint
    return RESUME_BACKOFF_LIMIT if policy is not None and policy.resume else 0


def compile_job(
    job: JobSpec,
    *,
    namespace: str,
    indexed: bool = False,
    completions: int = 1,
    parallelism: int | None = None,
    backoff_limit: int | None = None,
) -> Manifest:
    """Compile *job* to a ``batch/v1`` Job manifest (Indexed if *indexed*).

    *backoff_limit* overrides the policy-derived budget (:func:`resume_backoff_limit`).

    A job naming a ``tenant`` is compiled **suspended** and labelled with that tenant's Kueue
    ``LocalQueue``: Kueue un-suspends it only once the tenant is within quota, so the
    back-pressure is a real scheduler's rather than an in-process model of one. A tenant-scoped
    job therefore *requires* Kueue and a ``LocalQueue`` named ``tenant-<name>`` on the target
    cluster -- without them it stays suspended forever. Stand both up with
    :func:`~astro_mine.cloud.tenancy.namespace.tenant_manifests`.
    """
    meta = object_meta(
        deterministic_name(job, prefix="amc-job"),
        namespace=namespace,
        tenant=job.tenant,
        component="workload",
        annotations=input_annotations(job),
    )
    # The queue label goes on the *Job* -- the object Kueue admits. The pod template keeps the
    # plain platform labels, so pod selection stays about tenant / run / component.
    pod_labels = dict(meta["labels"])
    queue = queue_labels(job)
    meta["labels"].update(queue)

    pod_spec: Manifest = {"restartPolicy": "Never", "containers": [container_spec(job)]}
    spec: Manifest = {
        "backoffLimit": resume_backoff_limit(job) if backoff_limit is None else backoff_limit,
        "template": {"metadata": {"labels": pod_labels}, "spec": pod_spec},
    }
    if job.checkpoint is not None and job.checkpoint.resume:
        spec["podFailurePolicy"] = _IGNORE_DISRUPTION
    if queue:
        spec["suspend"] = True
    if indexed:
        spec["completionMode"] = "Indexed"
        spec["completions"] = completions
        spec["parallelism"] = parallelism if parallelism is not None else completions
    return {"apiVersion": "batch/v1", "kind": "Job", "metadata": meta, "spec": spec}


class K8sJobEngine:
    """The ``k8sjob`` engine: JobSpec -> a plain Kubernetes Job."""

    name = "k8sjob"

    def compile(self, job: JobSpec, *, namespace: str) -> Manifest:
        return compile_job(job, namespace=namespace)


register_engine(K8sJobEngine())
