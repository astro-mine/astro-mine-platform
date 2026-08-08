"""The containerized sandbox backend — the recommended posture for a public deployment.

bench.md §9: *"Submitted policies/plugins run out-of-process in sandboxed containers
(seccomp/gVisor), with no network egress and strict CPU/GPU/memory/time limits."* This is that
backend. It runs the **same eval-worker argv** as
:class:`~astro_mine.bench.sandbox.SubprocessSandbox`
— the one Cloud already fans out (RM-P1-BENCH-11) — but inside a container, which adds the two
boundaries a bare subprocess cannot draw:

- a **rootfs boundary** (``--read-only`` + a ``noexec,nosuid`` tmpfs, and a single writable bind
  mount for the result channel), so a submission cannot read the host filesystem, and
- a **network namespace with no interfaces** (``--network=none``), so no-egress holds for the whole
  process tree at the namespace level rather than syscall-by-syscall.

Everything else is defence in depth on top: ``--cap-drop=ALL``, ``--security-opt
no-new-privileges``, an explicit seccomp profile when one is configured, ``--pids-limit``,
``--memory``, ``--cpus``, ``--ulimit``, and an unprivileged ``--user``. A **gVisor** deployment adds
``runtime_flags=("--runtime=runsc",)``, which is the forward-looking sandbox bench.md §9 names —
it needs no code change here, only the flag.

The container image is the **signed, digest-pinned evaluation runner** (bench.md §9: *"the
evaluation runner image is signed and pinned"*), the same image the Cloud rollout path schedules.

Execution is injected (``execute``) so the argv this backend builds — which *is* the security
posture — is unit-tested exactly as it will be issued, without a container runtime present.

Backlog: bench#30 — astro-mine-bench#30
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from astro_mine.bench.sandbox._channel import (
    WORKER_RESULT,
    ResourceUsage,
    SandboxOutcome,
    SandboxStatus,
    WorkerInvocation,
    read_worker_result,
)
from astro_mine.bench.sandbox._limits import DEFAULT_LIMITS, NetworkPolicy, SandboxLimits
from astro_mine.bench.sandbox._subprocess import (
    STDERR_CAPTURE_BYTES,
    SandboxUnavailable,
    worker_argv,
)

__all__ = ["CONTAINER_OUTPUT_DIR", "ContainerSandbox", "Execute", "container_runtime_available"]

#: Where the writable result-channel mount lands inside the container. It is the **only** writable
#: path in the container's view, and the only thing that crosses back out.
CONTAINER_OUTPUT_DIR = "/out"

#: The unprivileged uid:gid the worker runs as inside the container (``nobody``).
_UNPRIVILEGED_USER = "65534:65534"


class Execute(Protocol):
    """The container-runtime invoker — injected so the compiled argv is testable without Docker."""

    def __call__(self, argv: Sequence[str], timeout: float) -> subprocess.CompletedProcess[bytes]:
        """Run the container runtime with ``argv``, capturing its streams; raise on timeout."""
        ...


def _run_container(argv: Sequence[str], timeout: float) -> subprocess.CompletedProcess[bytes]:
    """The default invoker: run the container runtime and capture its streams."""
    return subprocess.run(
        list(argv),
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def container_runtime_available(runtime: str = "docker") -> bool:
    """Whether ``runtime`` (``docker`` / ``podman``) is on the PATH — the sandbox needs it."""
    return shutil.which(runtime) is not None


class ContainerSandbox:
    """Run a submitted policy in a network-less, read-only, capability-stripped container.

    ``image`` is the digest-pinned evaluation-runner image (bench.md §9). ``runtime_flags`` are
    passed straight to the runtime — ``("--runtime=runsc",)`` selects **gVisor**.
    ``seccomp_profile`` is a path to a profile the runtime applies on top of its default; leaving it
    ``None`` keeps the runtime's own default profile (already restrictive), never ``unconfined``.
    """

    def __init__(
        self,
        image: str,
        *,
        limits: SandboxLimits = DEFAULT_LIMITS,
        runtime: str = "docker",
        runtime_flags: Sequence[str] = (),
        seccomp_profile: str | None = None,
        user: str = _UNPRIVILEGED_USER,
        execute: Execute = _run_container,
    ) -> None:
        if not image:
            raise ValueError("ContainerSandbox needs an evaluation-runner image")
        self._image = image
        self._limits = limits
        self._runtime = runtime
        self._runtime_flags = tuple(runtime_flags)
        self._seccomp_profile = seccomp_profile
        self._user = user
        self._execute = execute

    @property
    def limits(self) -> SandboxLimits:
        return self._limits

    @property
    def image(self) -> str:
        """The digest-pinned evaluation-runner image submissions execute inside."""
        return self._image

    def container_argv(self, invocation: WorkerInvocation, *, host_output_dir: str) -> list[str]:
        """Compile the runtime command line — this argv *is* the sandbox's security posture.

        Unit-tested flag by flag: the no-egress namespace, the read-only rootfs, the dropped
        capabilities, the frozen privileges, and every resource cap must all be present, because a
        container that is merely *out-of-process* is not a sandbox.
        """
        limits = self._limits
        argv = [self._runtime, "run", "--rm", *self._runtime_flags]

        # No egress: the container gets a namespace with no interface but loopback (bench.md §9).
        # NetworkPolicy.ALLOW is an explicit, auditable opt-in for a trusted, self-authored policy.
        if limits.network is NetworkPolicy.DENY:
            argv.append("--network=none")

        argv += [
            # No writable rootfs, no privileges, no capabilities, no privilege escalation.
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--user={self._user}",
            # Scratch space the policy may use, but never execute from.
            "--tmpfs=/tmp:rw,noexec,nosuid,size=64m",
            # Strict CPU / memory / process / descriptor / file-size caps.
            f"--cpus={limits.cpu_seconds / max(limits.wall_seconds, 1.0):.3f}",
            f"--memory={limits.memory_bytes}b",
            f"--memory-swap={limits.memory_bytes}b",
            f"--pids-limit={limits.max_processes}",
            f"--ulimit=nofile={limits.max_open_files}:{limits.max_open_files}",
            f"--ulimit=fsize={limits.output_bytes}:{limits.output_bytes}",
            "--ulimit=core=0:0",
        ]
        if self._seccomp_profile is not None:
            argv.append(f"--security-opt=seccomp={self._seccomp_profile}")
        # A submitted policy gets no GPU unless the envelope explicitly grants one.
        if limits.gpus > 0:
            argv.append(f"--gpus={limits.gpus}")

        # The single writable mount: the structured result-hand-back channel.
        argv += [
            f"--volume={host_output_dir}:{CONTAINER_OUTPUT_DIR}:rw",
            f"--env=ASTRO_MINE_OUTPUTS={CONTAINER_OUTPUT_DIR}",
            "--env=PYTHONHASHSEED=0",
            self._image,
        ]
        argv += worker_argv(invocation, python="python", output_dir=CONTAINER_OUTPUT_DIR)
        return argv

    def run(self, invocation: WorkerInvocation) -> SandboxOutcome:
        """Roll one seed inside the container and parse the result document off the bind mount."""
        # Fail closed: with the *default* invoker, an absent runtime means the submission would run
        # unsandboxed, so refuse. An injected invoker owns its own execution (a remote runner, or a
        # test), so the PATH of *this* host says nothing about whether it can be contained.
        if self._execute is _run_container and not container_runtime_available(self._runtime):
            raise SandboxUnavailable(
                f"container runtime {self._runtime!r} is not on PATH; refusing to execute an "
                "untrusted submission without the container sandbox"
            )
        host_output = tempfile.mkdtemp(prefix="astro-mine-bench-container-")
        try:
            return self._run_in(host_output, invocation)
        finally:
            shutil.rmtree(host_output, ignore_errors=True)

    def _run_in(self, host_output: str, invocation: WorkerInvocation) -> SandboxOutcome:
        argv = self.container_argv(invocation, host_output_dir=host_output)
        started = time.monotonic()
        timed_out = False
        stderr = b""
        returncode: int | None = None
        try:
            completed = self._execute(argv, self._limits.wall_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
        except FileNotFoundError as exc:  # pragma: no cover - guarded by the PATH check above
            raise SandboxUnavailable(
                f"container runtime {self._runtime!r} not found: {exc}"
            ) from exc
        else:
            returncode, stderr = completed.returncode, completed.stderr or b""
        elapsed = time.monotonic() - started

        result = read_worker_result(Path(host_output) / WORKER_RESULT)
        reported = result.usage if result is not None else None
        usage = ResourceUsage(
            wall_seconds=elapsed,
            cpu_seconds=None if reported is None else reported.cpu_seconds,
            max_rss_bytes=None if reported is None else reported.max_rss_bytes,
        )

        if timed_out:
            status: SandboxStatus = SandboxStatus.TIMEOUT
            detail: str | None = (
                f"wall-clock limit of {self._limits.wall_seconds}s exceeded; container terminated"
            )
        elif result is None:
            status, detail = (
                SandboxStatus.CRASHED,
                f"the container exited {returncode} without a parseable result document",
            )
        elif not result.ok:
            status, detail = SandboxStatus.FAILED, result.error
        else:
            status, detail = SandboxStatus.OK, None

        return SandboxOutcome(
            status=status,
            invocation_seed=invocation.seed,
            result=result,
            exit_code=returncode,
            signal=None,
            usage=usage,
            detail=detail,
            stderr=stderr.decode("utf-8", errors="replace")[-STDERR_CAPTURE_BYTES:],
        )
