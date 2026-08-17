# SPDX-License-Identifier: Apache-2.0
"""The Ray / KubeRay engine -- tightly-coupled stateful work.

Compiles a distributed :class:`~astro_mine.cloud.submission.jobspec.JobSpec` to a KubeRay
``RayJob`` (a RayCluster lifecycle + an entrypoint), the right shape for RL training, actor
rollout fleets, and distributed solves that need gang scheduling and shared actor state
(``cloud.md`` §2 principle 3, §3, §6). The job's command becomes the RayJob ``entrypoint``;
head and worker containers run ``ray start`` from the *same* digest-pinned image, so a
distributed run is the same container with a bigger executor. :class:`RayJobHandle` is the
value object returned to the caller.

Backlog: RM-P1-CLOUD-01/02 -- astro-mine-cloud#12
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING

from astro_mine.cloud.engines.base import (
    HARNESS_COMMAND,
    container_spec,
    deterministic_name,
    input_annotations,
    queue_labels,
    register_engine,
)
from astro_mine.cloud.k8s import Manifest, object_meta

if TYPE_CHECKING:
    from astro_mine.cloud.submission.jobspec import JobSpec

__all__ = ["RayEngine", "RayJobHandle", "compile_rayjob"]


def compile_rayjob(
    job: JobSpec,
    *,
    namespace: str,
    workers: int = 2,
    min_workers: int = 0,
    max_workers: int | None = None,
) -> Manifest:
    """Compile *job* to a KubeRay ``RayJob`` with an elastic worker group.

    The ``entrypoint`` is the **driver** -- the program KubeRay runs on the Ray head -- and that
    is the in-pod harness (:data:`~astro_mine.cloud.engines.base.HARNESS_COMMAND`), not
    ``job.command``. The driver is the only place that can stage the job's content-addressed
    inputs and capture its declared outputs, and the harness launches ``job.command`` itself once
    it has. The workload's argv rides in :data:`~astro_mine.cloud.k8s.ENV_JOBSPEC` on the head
    container, whose environment the driver process inherits.

    A RayJob entrypoint is a *shell command line*, not an argv, so it is built with
    :func:`shlex.join`: a naive ``" ".join`` silently mangles any argument holding a space or a
    quote, which would make the driver a different command than the one it was asked to be.

    A job naming a ``tenant`` is labelled with its Kueue ``LocalQueue``. Unlike a plain Job it is
    *not* pre-suspended here: Kueue queues RayJobs only when its ``ray.io/rayjob`` integration is
    enabled, and suspending an object nothing will admit would hang the run forever. With that
    integration on, Kueue's own webhook suspends the labelled RayJob.
    """
    entrypoint = shlex.join(HARNESS_COMMAND)
    head = container_spec(job, name="ray-head", include_command=False)
    worker = container_spec(job, name="ray-worker", include_command=False)
    meta = object_meta(
        deterministic_name(job, prefix="amc-ray"),
        namespace=namespace,
        tenant=job.tenant,
        component="ray",
        annotations=input_annotations(job),
    )
    meta["labels"].update(queue_labels(job))
    worker_group: Manifest = {
        "groupName": "workers",
        "replicas": workers,
        "minReplicas": min_workers,
        "maxReplicas": max_workers if max_workers is not None else workers,
        "rayStartParams": {},
        "template": {"spec": {"containers": [worker]}},
    }
    spec: Manifest = {
        "entrypoint": entrypoint,
        "shutdownAfterJobFinishes": True,
        "rayClusterSpec": {
            "headGroupSpec": {
                "rayStartParams": {"dashboard-host": "0.0.0.0"},
                "template": {"spec": {"containers": [head]}},
            },
            "workerGroupSpecs": [worker_group],
        },
    }
    return {"apiVersion": "ray.io/v1", "kind": "RayJob", "metadata": meta, "spec": spec}


@dataclass(frozen=True)
class RayJobHandle:
    """A handle to a compiled/submitted RayJob (name, namespace, entrypoint, manifest)."""

    name: str
    namespace: str
    entrypoint: str
    manifest: Manifest

    @classmethod
    def from_manifest(cls, manifest: Manifest) -> RayJobHandle:
        """Build a handle from a compiled RayJob manifest."""
        return cls(
            name=manifest["metadata"]["name"],
            namespace=manifest["metadata"].get("namespace", "default"),
            entrypoint=manifest["spec"]["entrypoint"],
            manifest=manifest,
        )


class RayEngine:
    """The ``ray`` engine: a distributed JobSpec -> a KubeRay RayJob."""

    name = "ray"

    def compile(self, job: JobSpec, *, namespace: str) -> Manifest:
        return compile_rayjob(job, namespace=namespace)


register_engine(RayEngine())
