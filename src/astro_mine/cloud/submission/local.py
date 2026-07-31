"""The local backend -- a subprocess on the workstation (the sacred tier).

``LocalBackend`` runs ``job.command`` as a subprocess in the local Python env
(``conventions.md`` §7 tier 1: "a single Python env") -- no cluster, no Docker, no
account. It is the default backend and the reason the local tier "MUST always work"
(``cloud.md`` §1, §2 principle 2). The workload reads inputs from ``$ASTRO_MINE_INPUTS``
and writes outputs to ``$ASTRO_MINE_OUTPUTS``.

Backlog: RM-P0-CLOUD-02 -- https://github.com/astro-mine/astro-mine-cloud/issues/2
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING

from astro_mine.cloud.submission._run import build_env, execute
from astro_mine.cloud.submission.backend import register_backend

if TYPE_CHECKING:
    from pathlib import Path

    from astro_mine.cloud.runs.events import RunObserver
    from astro_mine.cloud.submission.jobspec import JobSpec
    from astro_mine.cloud.submission.result import RunResult
    from astro_mine.core.artifacts import ArtifactStore

__all__ = ["LocalBackend"]


def _local_argv(command: Sequence[str]) -> list[str]:
    """Retarget a leading bare ``python`` at the interpreter that is actually running.

    A JobSpec's command is written for the **container** -- Bench's fan-out emits
    ``["python", "-m", "astro_mine.bench", "eval-worker", …]``, and the workload image puts
    the venv's interpreter first on ``PATH`` (``docker/Dockerfile``), so bare ``python`` is the
    venv's. This backend runs that same argv on the host, where ``PATH`` is the developer's:
    typically a system or conda interpreter with no ``astro_mine`` installed, so the job dies
    with ``No module named 'astro_mine'`` and the caller reports something far less obvious --
    for Bench, "no successful rollouts to score in this evaluation batch".

    Here rather than in a dispatcher because *every* cluster-less path arrives here: the local
    backend, and the Cloud dispatcher running against a ``DryRunClient``. Fixing one dispatcher
    fixed one of them, which is how this was first mis-diagnosed (astro-mine-platform#3).

    Only a bare ``python`` is rewritten. An absolute path, an explicit ``python3.12``, or any
    other executable is the caller's stated choice and is left alone.
    """
    argv = list(command)
    if argv and argv[0] == "python":
        argv[0] = sys.executable
    return argv


class _SubprocessLauncher:
    def launch(self, *, job: JobSpec, inputs_dir: Path, outputs_dir: Path) -> int:
        if not job.command:
            raise ValueError("the local backend requires a non-empty job.command")
        # Inherit the ambient env (PATH etc.) so non-absolute commands resolve, then
        # overlay the job's stable I/O contract.
        env = {**os.environ, **build_env(job, inputs_dir, outputs_dir, container=False)}
        completed = subprocess.run(
            _local_argv(job.command), cwd=str(outputs_dir.parent), env=env, check=False
        )
        return completed.returncode


class LocalBackend:
    """Runs a job as a subprocess in the local Python env (no cluster, no Docker)."""

    def run(
        self, job: JobSpec, *, store: ArtifactStore, observer: RunObserver | None = None
    ) -> RunResult:
        return execute(job, store, _SubprocessLauncher(), observer=observer)


register_backend("local", LocalBackend())
