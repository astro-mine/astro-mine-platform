# SPDX-License-Identifier: Apache-2.0
"""The docker backend -- the same job in a container (``docker compose`` tier).

``DockerBackend`` runs the *same* :class:`~astro_mine.cloud.submission.jobspec.JobSpec` in
its digest-pinned image via ``docker run`` (``conventions.md`` §7 tier 1: "docker
compose"). The run directory is bind-mounted (``/inputs`` read-only, ``/outputs``
writable) and the network is disabled, so a container run is reproducible and byte-for-byte
equivalent to the local run (``cloud.md`` §2 principle 2, §4 principle 4).

The argv construction (:func:`build_docker_argv`) is pure and unit-tested; only the
default runner shells out to Docker, so it runs solely in the opt-in ``docker``-marked
tests. Inject a *runner* to test or to retarget the container engine.

Backlog: RM-P0-CLOUD-02 -- astro-mine-cloud#2
"""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from astro_mine.cloud.submission._run import build_env, execute
from astro_mine.cloud.submission.backend import register_backend

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from astro_mine.cloud.runs.events import RunObserver
    from astro_mine.cloud.submission.jobspec import JobSpec
    from astro_mine.cloud.submission.result import RunResult
    from astro_mine.core.artifacts import ArtifactStore

    Runner = Callable[[Sequence[str]], int]

__all__ = ["DockerBackend", "build_docker_argv"]


def build_docker_argv(job: JobSpec, inputs_dir: Path, outputs_dir: Path) -> list[str]:
    """Build the ``docker run`` argv for *job* with the run directory bind-mounted."""
    env = build_env(job, inputs_dir, outputs_dir, container=True)
    argv = [
        "docker",
        "run",
        "--rm",
        "--network=none",
        "-v",
        f"{inputs_dir}:/inputs:ro",
        "-v",
        f"{outputs_dir}:/outputs",
    ]
    for key, value in sorted(env.items()):
        argv += ["-e", f"{key}={value}"]
    argv.append(job.image.reference)  # digest-pinned; never a floating tag
    argv += job.command
    return argv


def _docker_run(argv: Sequence[str]) -> int:  # pragma: no cover - requires a Docker daemon
    return subprocess.run(list(argv), check=False).returncode


class _DockerLauncher:
    def __init__(self, runner: Runner) -> None:
        self._runner = runner

    def launch(self, *, job: JobSpec, inputs_dir: Path, outputs_dir: Path) -> int:
        if not job.command:
            raise ValueError("the docker backend requires a non-empty job.command")
        return self._runner(build_docker_argv(job, inputs_dir, outputs_dir))


class DockerBackend:
    """Runs a job in its digest-pinned image via ``docker run`` (injectable *runner*)."""

    def __init__(self, *, runner: Runner | None = None) -> None:
        self._runner = runner or _docker_run

    def run(
        self, job: JobSpec, *, store: ArtifactStore, observer: RunObserver | None = None
    ) -> RunResult:
        return execute(job, store, _DockerLauncher(self._runner), observer=observer)


register_backend("docker", DockerBackend())
