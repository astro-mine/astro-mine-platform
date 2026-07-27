"""The engine seam -- the ``Engine`` protocol, registry, and shared container builders.

An :class:`Engine` compiles a single :class:`~astro_mine.cloud.submission.jobspec.JobSpec` to
a Kubernetes manifest ``dict`` for one workload shape. Engines register under a name
(``k8sjob``, ``ray``), so the cluster backend selects one by name -- a swap, not a code fork
(``cloud.md`` §3 extension points). The shared builders here (:func:`container_spec`,
:func:`job_resources`, :func:`input_annotations`) turn a JobSpec's image / command / env /
resources / content-addressed I/O into the pod-level pieces every engine reuses, so a job
runs the *same* container whichever engine schedules it.

Backlog: RM-P1-CLOUD-01/02 -- https://github.com/astro-mine/astro-mine-cloud/issues/12
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from astro_mine.cloud.artifacts.addressing import content_address, hex_of
from astro_mine.cloud.k8s import (
    ENV_JOBSPEC,
    LABEL_QUEUE_NAME,
    Manifest,
    env_var_list,
    sanitize_name,
)
from astro_mine.cloud.tenancy.namespace import tenant_namespace

if TYPE_CHECKING:
    from astro_mine.cloud.submission.jobspec import JobSpec

__all__ = [
    "ANNOTATION_INPUTS",
    "ANNOTATION_OUTPUTS",
    "HARNESS_COMMAND",
    "Engine",
    "container_spec",
    "deterministic_name",
    "get_engine",
    "input_annotations",
    "job_resources",
    "queue_labels",
    "register_engine",
    "registered_engines",
    "workload_env",
]

#: Annotations recording a job's content-addressed I/O on its compiled object, so the
#: workload's entrypoint can stage inputs from and write outputs to the object store.
ANNOTATION_INPUTS = "astro-mine.org/inputs"
ANNOTATION_OUTPUTS = "astro-mine.org/outputs"

#: What a cluster container actually runs: the in-pod harness
#: (:mod:`astro_mine.cloud.submission.harness`). It reads the JobSpec back out of
#: :data:`~astro_mine.cloud.k8s.ENV_JOBSPEC`, stages the job's content-addressed inputs from the
#: object store, launches ``job.command``, captures the declared outputs back, and records the
#: RunContext -- through the *same* ``execute()`` the local and docker backends call, which is
#: what makes a cluster run's content address equal a laptop run's by construction.
#:
#: Every cluster workload image must therefore have ``astro_mine.cloud`` installed and ``python``
#: on its ``PATH``; see ``platform/kind/workload.Dockerfile``.
HARNESS_COMMAND = ["python", "-m", "astro_mine.cloud.submission.harness"]


@runtime_checkable
class Engine(Protocol):
    """Compiles a JobSpec to a manifest for one workload shape."""

    name: str

    def compile(self, job: JobSpec, *, namespace: str) -> Manifest: ...


_REGISTRY: dict[str, Engine] = {}


def register_engine(engine: Engine, *, replace: bool = False) -> None:
    """Register *engine* under its ``name``; raise on a duplicate unless *replace* is set."""
    if engine.name in _REGISTRY and not replace:
        raise ValueError(f"engine {engine.name!r} is already registered")
    _REGISTRY[engine.name] = engine


def get_engine(name: str) -> Engine:
    """Return the engine registered under *name*, or raise with the known names."""
    try:
        return _REGISTRY[name]
    except KeyError:
        known = ", ".join(registered_engines()) or "(none)"
        raise ValueError(f"unknown engine {name!r}; registered: {known}") from None


def registered_engines() -> tuple[str, ...]:
    """Return the registered engine names, sorted."""
    return tuple(sorted(_REGISTRY))


def workload_env(job: JobSpec) -> dict[str, str]:
    """The container env for a cluster run: the JobSpec + ``job.env`` + seed + Core version.

    Mirrors the local backend's :func:`~astro_mine.cloud.submission._run.build_env` seed
    contract; the input/output *paths* are not bind-mounted on a cluster, so instead of
    ``ASTRO_MINE_INPUTS``/``ASTRO_MINE_OUTPUTS`` the pod gets the whole JobSpec as JSON under
    :data:`~astro_mine.cloud.k8s.ENV_JOBSPEC` and
    :mod:`~astro_mine.cloud.submission.harness` -- the in-pod entrypoint -- stages the
    content-addressed inputs from the object store itself, into a run directory it owns. The
    container image is therefore identical between backends: only the *staging* differs.

    The JobSpec is serialised *before* this variable is added, so the spec a pod reconstructs is
    the one the caller submitted -- ``ENV_JOBSPEC`` is not recursively embedded in itself.
    """
    env = dict(job.env)
    if job.seed is not None:
        env["ASTRO_MINE_SEED"] = str(job.seed)
    if job.core_interface_version is not None:
        env["ASTRO_MINE_CORE_INTERFACE_VERSION"] = job.core_interface_version
    env[ENV_JOBSPEC] = job.model_dump_json()
    return env


def queue_labels(job: JobSpec) -> dict[str, str]:
    """The Kueue ``LocalQueue`` label for a tenant-scoped *job* (empty when it has no tenant).

    A job that names a tenant is *queue-managed*: it is admitted through that tenant's
    ``LocalQueue`` -- ``tenant-<name>``, the queue
    :func:`~astro_mine.cloud.tenancy.namespace.tenant_manifests` stands up -- so a flood from one
    tenant queues behind its quota instead of starving the others (``cloud.md`` §4, §9). Without
    this label Kueue never sees the object and the quota model is decorative.
    """
    if job.tenant is None:
        return {}
    return {LABEL_QUEUE_NAME: tenant_namespace(job.tenant)}


def job_resources(job: JobSpec) -> dict[str, dict[str, str]]:
    """The container ``resources`` block: typed :class:`ResourceRequest` or the raw map."""
    if job.resource_request is not None:
        return job.resource_request.to_k8s_resources()
    if job.resources:
        return {"requests": dict(job.resources), "limits": dict(job.resources)}
    return {}


def input_annotations(job: JobSpec) -> dict[str, str]:
    """Record content-addressed inputs/outputs as annotations (sorted, byte-stable)."""
    annotations: dict[str, str] = {}
    if job.inputs:
        annotations[ANNOTATION_INPUTS] = json.dumps(job.inputs, sort_keys=True)
    if job.outputs:
        annotations[ANNOTATION_OUTPUTS] = json.dumps(sorted(job.outputs))
    return annotations


def container_spec(
    job: JobSpec, *, name: str = "workload", include_command: bool = True
) -> Manifest:
    """Build the container spec (image / command / env / resources) shared by all engines.

    The container runs :data:`HARNESS_COMMAND`, **not** ``job.command``. A Kubernetes
    ``container.command`` overrides the image's ENTRYPOINT, so naming the workload's command here
    would run it *directly* -- bypassing the harness, and with it the input staging, the output
    capture and the provenance record, none of which a pod can get any other way (there is no
    bind-mount on a cluster). The workload's own argv travels inside
    :data:`~astro_mine.cloud.k8s.ENV_JOBSPEC` and the harness launches it.

    *include_command* is ``False`` for a Ray head/worker, whose containers KubeRay makes run
    ``ray start``; there the harness is the RayJob ``entrypoint`` instead (``cloud.md`` §3).
    """
    container: Manifest = {
        "name": name,
        "image": job.image.reference,  # digest-pinned; never a floating tag
        "env": env_var_list(workload_env(job)),
    }
    if include_command:
        container["command"] = list(HARNESS_COMMAND)
    resources = job_resources(job)
    if resources:
        container["resources"] = resources
    return container


def deterministic_name(job: JobSpec, *, prefix: str) -> str:
    """A stable, unique object name for *job*: ``<prefix>-<first 12 hex of its address>``.

    Content-addressing the JobSpec means equal jobs get equal names and distinct jobs get
    distinct ones -- reproducible by construction (``cloud.md`` §2 principle 4), with no
    clock or counter that would break determinism gates.
    """
    digest = hex_of(content_address(job.model_dump(mode="json")))
    return sanitize_name(f"{prefix}-{digest[:12]}")
