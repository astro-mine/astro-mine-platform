"""The submit() backend-equivalence contract (local backend).

The same call site runs in-process on a workstation as it later will on a cluster -- a
backend swap, not a code fork (``cloud.md`` §2 principle 2, §3). ``submit(job)`` resolves
a backend by name and runs the :class:`JobSpec` against a content-addressed store; Phase 0
registers the dependency-free ``"local"`` (subprocess) and ``"docker"`` (container)
backends, and defaults to a local :class:`FilesystemArtifactStore` so the sacred local
tier needs no cloud and no account.

```python
from astro_mine.cloud import submit
from astro_mine.cloud.packaging import ImageRef
from astro_mine.cloud.submission import JobSpec

job = JobSpec(image=ImageRef.parse("ghcr.io/astro-mine/astro-mine-bench@sha256:..."),
              command=["python", "-m", "astro_mine.bench"], seed=42)
result = submit(job)                 # local subprocess; or submit(job, backend="docker")
assert result.ok
```

Backlog: RM-P0-CLOUD-02 -- https://github.com/astro-mine/astro-mine-cloud/issues/2
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_mine.cloud.artifacts.store import FilesystemArtifactStore
from astro_mine.cloud.runs.events import NullPublisher, RunObserver
from astro_mine.cloud.submission._run import pre_run_context
from astro_mine.cloud.submission.backend import (
    Backend,
    get_backend,
    register_backend,
    registered_backends,
)

# Importing the backend modules registers "local", "docker", and "cluster" as a side effect.
from astro_mine.cloud.submission.cluster import (
    ClusterBackend,
    ClusterDispatchError,
    DryRunClient,
    KubectlClusterClient,
    KubectlRunner,
)
from astro_mine.cloud.submission.docker import DockerBackend
from astro_mine.cloud.submission.jobspec import CheckpointPolicy, JobSpec, ResourceRequest
from astro_mine.cloud.submission.local import LocalBackend
from astro_mine.cloud.submission.result import RunResult
from astro_mine.cloud.submission.sweepspec import SweepSpec
from astro_mine.cloud.submission.workflowspec import WorkflowSpec, WorkflowStep

if TYPE_CHECKING:
    from astro_mine.cloud.artifacts.store import ArtifactStore
    from astro_mine.cloud.runs.events import EventPublisher
    from astro_mine.cloud.runs.status import JobStatusStore

__all__ = [
    "Backend",
    "CheckpointPolicy",
    "ClusterBackend",
    "ClusterDispatchError",
    "DockerBackend",
    "DryRunClient",
    "JobSpec",
    "KubectlClusterClient",
    "KubectlRunner",
    "LocalBackend",
    "ResourceRequest",
    "RunResult",
    "SweepSpec",
    "WorkflowSpec",
    "WorkflowStep",
    "get_backend",
    "register_backend",
    "registered_backends",
    "submit",
]


def submit(
    job: JobSpec,
    *,
    backend: str = "local",
    store: ArtifactStore | None = None,
    publisher: EventPublisher | None = None,
    status_store: JobStatusStore | None = None,
) -> RunResult:
    """Submit *job* through the named *backend* (default ``"local"``).

    With no *store*, a local :class:`FilesystemArtifactStore` is used -- no cloud, no
    account. The call site is identical whichever backend runs the job.

    *publisher* / *status_store* wire the lifecycle-eventing substrate (RM-P1-CLOUD-06): the run
    emits ``submitted`` here, then ``started``/``completed``/``failed`` from the harness. Both
    default to the inert local-tier behaviour (a :class:`NullPublisher`, no status store), so a
    laptop submit needs no broker; a deployment injects a
    :class:`~astro_mine.cloud.runs.nats.NatsEventPublisher` and a
    :class:`~astro_mine.cloud.runs.status.RedisJobStatusStore`.
    """
    if store is None:
        store = FilesystemArtifactStore()
    observer = RunObserver(
        publisher=publisher if publisher is not None else NullPublisher(),
        status_store=status_store,
        tenant=job.tenant,
    )
    observer.transition(pre_run_context(job, store), "submitted")
    return get_backend(backend).run(job, store=store, observer=observer)
