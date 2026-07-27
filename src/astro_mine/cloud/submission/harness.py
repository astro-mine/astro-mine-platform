"""The in-container run harness -- the entrypoint a *cluster* workload actually runs.

The compiled manifests carry a job's content-addressed I/O as annotations
(``astro-mine.org/inputs`` / ``astro-mine.org/outputs``) precisely so "the workload's entrypoint
can stage inputs from and write outputs to the object store". This module *is* that entrypoint.
It reads the :class:`~astro_mine.cloud.submission.jobspec.JobSpec` back from
``$ASTRO_MINE_JOBSPEC`` (:data:`~astro_mine.cloud.k8s.ENV_JOBSPEC`, which the engines compile
into the container env), builds the shared artifact store from the environment, and hands off to
:func:`~astro_mine.cloud.submission._run.execute` -- **the very function the local and docker
backends call**.

That reuse is the whole design. Backend equivalence is not asserted here, it is *constructed*:
the cluster run stages inputs, launches the command, captures declared outputs, and records the
:class:`~astro_mine.cloud.artifacts.runcontext.RunContext` through the same code path as the
laptop run, so the two content addresses agree by construction rather than by coincidence
(``cloud.md`` §2 principle 2, principle 4).

Two things the *image* must get right, because both feed the RunContext content address
(``conventions.md`` §5) -- a mismatch in either fails the determinism gate for reasons that have
nothing to do with the run:

- ``code_version`` -- the installed ``astro-mine-cloud`` version. hatch-vcs derives it from
  ``git describe``, so an image built without the repo's git state gets a *different* version.
  Pin it at build time with ``SETUPTOOLS_SCM_PRETEND_VERSION``; see
  ``platform/kind/workload.Dockerfile``.
- ``env_lockfile`` -- the content address of the active ``uv.lock``.
  :func:`~astro_mine.cloud.submission._run._active_uv_lock` walks up from the CWD, so the image
  must ship the *same* ``uv.lock`` bytes under its ``WORKDIR`` (or set
  ``ASTRO_MINE_ENV_LOCKFILE``), else the pin is ``None`` in the pod and set on the host.

The host-side collector (:class:`~astro_mine.cloud.submission.cluster.KubectlClusterClient`)
never re-derives the result: it reads the two sentinel lines this module prints on stdout and
loads the RunContext back out of the shared store by its content address.

Backlog: RM-P1-CLOUD-02 -- https://github.com/astro-mine/astro-mine-cloud/issues/21
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

from astro_mine.cloud.artifacts.store import DEFAULT_ROOT_ENV, FilesystemArtifactStore
from astro_mine.cloud.k8s import ENV_JOBSPEC
from astro_mine.cloud.submission._run import execute
from astro_mine.cloud.submission.jobspec import JobSpec

# The *same* launcher the local backend uses: the harness is a subprocess launch inside a pod,
# which is exactly what the local backend is on a workstation. Sharing it (rather than
# re-implementing it) is what makes the two runs equivalent by construction.
from astro_mine.cloud.submission.local import _SubprocessLauncher

if TYPE_CHECKING:
    from collections.abc import Mapping

    from astro_mine.cloud.artifacts.store import ArtifactStore
    from astro_mine.cloud.submission.result import RunResult

__all__ = [
    "ENV_JOBSPEC",
    "EXIT_CODE_SENTINEL",
    "RUN_CONTEXT_SENTINEL",
    "S3_BUCKET_VAR",
    "S3_ENDPOINT_VAR",
    "build_store",
    "main",
    "parse_sentinels",
    "run",
]

#: The shared object store the pod stages inputs from and writes outputs + provenance to. These
#: ride in ``job.env``, which :func:`~astro_mine.cloud.engines.base.workload_env` already
#: forwards into the container -- so pointing a pod at its store needs no engine change. With no
#: bucket the harness falls back to the filesystem store (honoring ``ASTRO_MINE_ARTIFACT_ROOT``),
#: which is what makes this module testable with no cluster and no object store at all.
S3_BUCKET_VAR = "ASTRO_MINE_S3_BUCKET"
S3_ENDPOINT_VAR = "ASTRO_MINE_S3_ENDPOINT"

#: Sentinels printed on stdout so the host-side collector can recover the run from pod logs
#: alone -- no API-server watch, no shared filesystem. The RunContext travels by *address*
#: (the envelope itself is in the shared store); the exit code travels literally, because the
#: RunContext does not carry one.
RUN_CONTEXT_SENTINEL = "ASTRO_MINE_RUN_CONTEXT="
EXIT_CODE_SENTINEL = "ASTRO_MINE_EXIT_CODE="


def build_store(env: Mapping[str, str] | None = None) -> ArtifactStore:
    """Build the run's artifact store from *env* (default: the process environment).

    An ``ASTRO_MINE_S3_BUCKET`` selects the S3-compatible store -- MinIO in-cluster, or any S3
    backend, no code change either way (``cloud.md`` §5). boto3 picks its credentials up from the
    ambient ``AWS_*`` variables. Otherwise the dependency-free
    :class:`~astro_mine.cloud.artifacts.store.FilesystemArtifactStore` is used.
    """
    environ = os.environ if env is None else env
    bucket = environ.get(S3_BUCKET_VAR)
    if not bucket:
        return FilesystemArtifactStore(environ.get(DEFAULT_ROOT_ENV))
    from astro_mine.cloud.artifacts.s3 import S3ArtifactStore

    return S3ArtifactStore(bucket, endpoint_url=environ.get(S3_ENDPOINT_VAR) or None)


def run(job: JobSpec, store: ArtifactStore) -> RunResult:
    """Run *job* against *store* through the shared harness -- the local backend's own path."""
    return execute(job, store, _SubprocessLauncher())


def parse_sentinels(text: str) -> tuple[str, int] | None:
    """Parse ``(run_context_address, exit_code)`` out of a pod's log *text*.

    Returns ``None`` when either sentinel is absent -- a pod killed mid-run, or an image that is
    not running this harness at all. The *last* occurrence wins, so a container whose attempts
    streamed into one log reports its final state.
    """
    address: str | None = None
    exit_code: int | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith(RUN_CONTEXT_SENTINEL):
            address = stripped[len(RUN_CONTEXT_SENTINEL) :].strip()
        elif stripped.startswith(EXIT_CODE_SENTINEL):
            value = stripped[len(EXIT_CODE_SENTINEL) :].strip()
            if value.lstrip("-").isdigit():
                exit_code = int(value)
    if not address or exit_code is None:
        return None
    return address, exit_code


def main() -> int:
    """The container entrypoint: run the JobSpec in ``$ASTRO_MINE_JOBSPEC``, print the sentinels.

    Exits with the *workload's* exit code, so a pod's success or failure is the job's. That is
    what makes a Job's ``Complete``/``Failed`` condition mean what the caller expects, and what
    lets Kubernetes retry a preempted run (``cloud.md`` §8).
    """
    raw = os.environ.get(ENV_JOBSPEC)
    if not raw:
        print(
            f"{ENV_JOBSPEC} is not set: the cluster harness reads its JobSpec as JSON from that "
            "environment variable, which the engines compile into the container env.",
            file=sys.stderr,
        )
        return 2
    result = run(JobSpec.model_validate_json(raw), build_store())
    print(f"{RUN_CONTEXT_SENTINEL}{result.run_context_address}", flush=True)
    print(f"{EXIT_CODE_SENTINEL}{result.exit_code}", flush=True)
    return result.exit_code


if __name__ == "__main__":  # pragma: no cover - the container entrypoint
    raise SystemExit(main())
