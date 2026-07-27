"""The out-of-process sandbox backend: rlimits + seccomp, no egress (bench.md §9; bench#30).

The default backend the hosted leaderboard runs submitted policies under. It executes the **existing
eval-worker argv** (:mod:`astro_mine.bench.eval._worker` — the same command Cloud fans out per seed,
RM-P1-BENCH-11) in a fresh child process that is, before it ever reaches the submission's code:

- **kernel-capped** — ``RLIMIT_CPU`` (CPU seconds), ``RLIMIT_AS`` (address space), ``RLIMIT_FSIZE``
  (bytes written), ``RLIMIT_NPROC`` (no fork bombs), ``RLIMIT_NOFILE``, ``RLIMIT_CORE`` = 0;
- **network-less** — a seccomp-BPF filter denies every socket syscall
  (:mod:`astro_mine.bench.sandbox._seccomp`), inherited across ``exec`` and unrevocable;
- **filesystem-confined** — a Landlock allowlist (:mod:`astro_mine.bench.sandbox._landlock`) grants
  the worker only the interpreter, its libraries, the submission's import roots, and the run's
  scratch directory, so the embargoed held-out seeds — and the rest of the host — are simply
  unreadable (bench#36), likewise inherited across ``exec`` and unrevocable;
- **privilege-frozen** — ``PR_SET_NO_NEW_PRIVS``, so no setuid binary can raise its privileges;
- **environment-scrubbed** — the child inherits an explicit allowlist, never the evaluator's
  environment, so a submission cannot read the deployment's database URL, Hub token, or OIDC
  secrets out of ``os.environ`` (and, with ``/proc`` outside the Landlock allowlist, cannot recover
  them from ``/proc/<pid>/environ`` either);
- **session-isolated** — its own process group, so the wall-clock cap kills the whole tree, not
  just the leader.

It also enforces the wall-clock cap the kernel cannot (``RLIMIT_CPU`` does not fire on a policy that
merely sleeps), and it **fails closed**: if the no-egress filter or the Landlock confinement cannot
be installed on this platform, :meth:`SubprocessSandbox.run` raises :class:`SandboxUnavailable`
instead of running untrusted code without the guarantee.

What it does **not** isolate — the process's view of the host's PID/process table, and the kernel
attack surface — is stated in ``TRUST_BOUNDARY.md``;
:class:`~astro_mine.bench.sandbox.ContainerSandbox` is the tier that closes those (namespaces +
gVisor), and is the recommended posture for a public deployment.

Backlog: bench#30, bench#36 — https://github.com/astro-mine/astro-mine-bench/issues/36
"""

from __future__ import annotations

import os
import resource
import shutil
import subprocess
import sys
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
    WorkerResult,
    read_worker_result,
)
from astro_mine.bench.sandbox._landlock import (
    SYSTEM_WRITE_PATHS,
    filesystem_read_roots,
    landlock_supported,
    restrict_filesystem,
)
from astro_mine.bench.sandbox._limits import (
    DEFAULT_LIMITS,
    FilesystemPolicy,
    NetworkPolicy,
    SandboxLimits,
)
from astro_mine.bench.sandbox._seccomp import (
    SeccompUnsupported,
    egress_filter_supported,
    install_egress_filter,
)

__all__ = [
    "STDERR_CAPTURE_BYTES",
    "Sandbox",
    "SandboxError",
    "SandboxUnavailable",
    "SubprocessSandbox",
    "rlimit_settings",
    "sandbox_environment",
    "worker_argv",
]

#: How much of a worker's stderr is kept for the audit trail. It is untrusted text: captured and
#: stored, never parsed, never executed.
STDERR_CAPTURE_BYTES = 4096

#: The only environment variables a sandboxed worker inherits from the evaluator. Everything else —
#: database URLs, registry paths, tokens, OIDC settings — is withheld (conventions.md §9: no
#: secrets reachable by untrusted code).
_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "SSL_CERT_FILE", "SSL_CERT_DIR")


class SandboxError(Exception):
    """A sandboxed run could not be trusted — the submission is rejected, never scored."""


class SandboxUnavailable(SandboxError):
    """The requested isolation cannot be enforced here, so nothing was run (fail-closed).

    Raised rather than degrading to a weaker sandbox: an unenforced boundary that *looks* enforced
    is worse than no boundary at all, because the leaderboard would run community code believing it
    was contained.
    """


class Sandbox(Protocol):
    """Runs one seed of one submission out-of-process, under an enforced resource envelope."""

    @property
    def limits(self) -> SandboxLimits:
        """The envelope this backend enforces (bench.md §9)."""
        ...

    def run(self, invocation: WorkerInvocation) -> SandboxOutcome:
        """Roll ``invocation`` in the sandbox and hand back its structured outcome."""
        ...


def worker_argv(
    invocation: WorkerInvocation, *, python: str, output_dir: str, module: str = "astro_mine.bench"
) -> list[str]:
    """The eval-worker command line for ``invocation`` (the argv Cloud already fans out).

    ``--emit json`` selects the dependency-clean hand-back: the worker writes only the
    :class:`~astro_mine.bench.sandbox.WorkerResult` document, skipping the Parquet/MCAP artifacts
    (and their ``pyarrow``/``mcap`` imports) that the Cloud scale-out path collects.
    """
    return [
        python,
        "-m",
        module,
        "eval-worker",
        "--scenario-id",
        invocation.scenario_id,
        "--policy-ref",
        invocation.policy_ref,
        "--seed",
        str(invocation.seed),
        "--output-dir",
        output_dir,
        "--emit",
        "json",
    ]


def rlimit_settings(limits: SandboxLimits) -> tuple[tuple[int, tuple[int, int]], ...]:
    """Map :class:`SandboxLimits` onto the POSIX rlimits the child sets before ``exec``.

    A pure function so the mapping is unit-tested directly: the code that *applies* it runs in a
    forked child, where a test cannot observe it.
    """
    return (
        # A CPU-seconds cap: SIGXCPU at the soft limit, SIGKILL one second later at the hard limit.
        (resource.RLIMIT_CPU, (limits.cpu_seconds, limits.cpu_seconds + 1)),
        # Address space: an over-allocating policy fails its allocation instead of the host OOMing.
        (resource.RLIMIT_AS, (limits.memory_bytes, limits.memory_bytes)),
        # The largest file the worker may write — it only ever needs to write result.json.
        (resource.RLIMIT_FSIZE, (limits.output_bytes, limits.output_bytes)),
        # Fork bombs: the child may not spawn beyond this many processes.
        (resource.RLIMIT_NPROC, (limits.max_processes, limits.max_processes)),
        (resource.RLIMIT_NOFILE, (limits.max_open_files, limits.max_open_files)),
        # No core dumps: a crashing submission must not write the host's disk full.
        (resource.RLIMIT_CORE, (0, 0)),
    )


def sandbox_environment(
    limits: SandboxLimits, *, workdir: str, python_path: Sequence[str] = ()
) -> dict[str, str]:
    """The scrubbed environment the sandboxed worker sees — an allowlist, never an inheritance.

    ``HOME``/``TMPDIR`` are pointed at the run's own scratch directory, ``PYTHONHASHSEED`` is fixed
    (the worker must be deterministic — bench.md §9), and ``CUDA_VISIBLE_DEVICES`` is emptied unless
    the envelope actually grants GPUs.
    """
    env = {name: os.environ[name] for name in _ENV_ALLOWLIST if name in os.environ}
    env.update(
        {
            "HOME": workdir,
            "TMPDIR": workdir,
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }
    )
    if python_path:
        env["PYTHONPATH"] = os.pathsep.join(str(entry) for entry in python_path)
    if limits.gpus == 0:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env


def _child_setup(  # pragma: no cover - runs in the forked child
    limits: SandboxLimits, *, read_roots: Sequence[str], write_roots: Sequence[str]
) -> None:
    """Applied in the child between ``fork`` and ``exec``: rlimits, no-egress, then confinement.

    Order matters only in that the Landlock confinement goes **last**: its own syscalls
    (``landlock_*``) are not among those the seccomp filter denies, and it must still be able to
    open the O_PATH handles for ``read_roots``/``write_roots`` — which it does here, before the
    ruleset takes effect. Any failure raises, which makes ``Popen`` fail the spawn — the submission
    does not run half-sandboxed.
    """
    for which, (soft, hard) in rlimit_settings(limits):
        resource.setrlimit(which, (soft, hard))
    os.umask(0o077)
    if limits.network is NetworkPolicy.DENY:
        install_egress_filter()
    if limits.filesystem is FilesystemPolicy.CONFINE:
        restrict_filesystem(read_roots, write_roots)


class SubprocessSandbox:
    """Run a submitted policy out-of-process under rlimits + a seccomp no-egress filter.

    ``python_path`` are extra import roots forwarded to the worker (the scrubbed environment does
    not inherit ``PYTHONPATH``); a deployment that installs submissions into the worker image needs
    none. They are also the roots the Landlock confinement grants read access to, so a submission
    can import itself but nothing outside them. ``limits`` is the enforced envelope — the default
    denies egress, confines the filesystem, and exposes no GPU.
    """

    def __init__(
        self,
        *,
        limits: SandboxLimits = DEFAULT_LIMITS,
        python: str | None = None,
        python_path: Sequence[str] = (),
    ) -> None:
        self._limits = limits
        self._python = python or sys.executable
        self._python_path = tuple(str(entry) for entry in python_path)
        #: The Landlock read allowlist is fixed for a given (interpreter, import roots) pair — it
        #: probes the interpreter, so it is computed once and reused across seeds.
        self._read_roots: tuple[str, ...] | None = None

    @property
    def limits(self) -> SandboxLimits:
        return self._limits

    def _preflight(self) -> None:
        """Refuse to run untrusted code if the requested isolation cannot be enforced here."""
        if self._limits.network is NetworkPolicy.DENY and not egress_filter_supported():
            raise SandboxUnavailable(
                "cannot enforce NetworkPolicy.DENY on "
                f"{sys.platform}: no seccomp egress filter for this platform. Run the leaderboard "
                "evaluator on Linux, or use ContainerSandbox (--network=none); refusing to execute "
                "an untrusted submission unsandboxed."
            )
        if self._limits.filesystem is FilesystemPolicy.CONFINE and not landlock_supported():
            raise SandboxUnavailable(
                "cannot enforce FilesystemPolicy.CONFINE on "
                f"{sys.platform}: Landlock is unavailable on this kernel. Run the leaderboard "
                "evaluator on a Linux kernel with Landlock (>=5.13, in CONFIG_LSM), or use "
                "ContainerSandbox (--read-only); refusing to execute an untrusted submission with "
                "an unconfined filesystem."
            )

    def _confinement_roots(self, workdir: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """The Landlock (read, write) allowlists for a run in ``workdir`` (empty when unconfined).

        The read set — interpreter, libraries, and the submission's import roots — is cached; the
        write set is the run's own scratch directory plus the one writable device a worker needs.
        The repo root is never in either, which is what keeps ``embargo/`` unreadable.
        """
        if self._limits.filesystem is not FilesystemPolicy.CONFINE:
            return (), ()
        if self._read_roots is None:
            self._read_roots = filesystem_read_roots(self._python, extra_roots=self._python_path)
        return self._read_roots, (workdir, *SYSTEM_WRITE_PATHS)

    def run(self, invocation: WorkerInvocation) -> SandboxOutcome:
        """Roll one seed in a fresh, capped, network-less child and parse its result document."""
        self._preflight()
        workdir = tempfile.mkdtemp(prefix="astro-mine-bench-sandbox-")
        try:
            return self._run_in(workdir, invocation)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _run_in(self, workdir: str, invocation: WorkerInvocation) -> SandboxOutcome:
        argv = worker_argv(invocation, python=self._python, output_dir=workdir)
        env = sandbox_environment(self._limits, workdir=workdir, python_path=self._python_path)
        read_roots, write_roots = self._confinement_roots(workdir)
        started = time.monotonic()
        try:
            process = subprocess.Popen(
                argv,
                cwd=workdir,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
                # Its own session: the wall-clock cap kills the whole process tree, and the child
                # cannot signal or read the terminal of the evaluator that spawned it.
                start_new_session=True,
                preexec_fn=lambda: _child_setup(
                    self._limits, read_roots=read_roots, write_roots=write_roots
                ),
            )
        except SeccompUnsupported as exc:
            raise SandboxUnavailable(f"the no-egress filter could not be installed: {exc}") from exc
        except OSError as exc:
            raise SandboxUnavailable(f"the sandboxed worker could not be spawned: {exc}") from exc

        timed_out = False
        try:
            _, stderr = process.communicate(timeout=self._limits.wall_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            self._kill_tree(process)
            _, stderr = process.communicate()
        elapsed = time.monotonic() - started

        result = read_worker_result(Path(workdir) / WORKER_RESULT)
        return self._outcome(
            invocation,
            result=result,
            returncode=process.returncode,
            stderr=stderr,
            elapsed=elapsed,
            timed_out=timed_out,
        )

    @staticmethod
    def _kill_tree(process: subprocess.Popen[bytes]) -> None:
        """SIGKILL the worker's whole process group — a submission cannot outlive its wall cap."""
        try:
            os.killpg(os.getpgid(process.pid), 9)
        except (OSError, ProcessLookupError):  # pragma: no cover - the child already exited
            process.kill()

    def _outcome(
        self,
        invocation: WorkerInvocation,
        *,
        result: WorkerResult | None,
        returncode: int | None,
        stderr: bytes,
        elapsed: float,
        timed_out: bool,
    ) -> SandboxOutcome:
        """Classify how the child ended (:class:`SandboxStatus`; every non-OK is a refusal)."""
        reported = result.usage if result is not None else None
        usage = ResourceUsage(
            wall_seconds=elapsed,
            cpu_seconds=None if reported is None else reported.cpu_seconds,
            max_rss_bytes=None if reported is None else reported.max_rss_bytes,
        )
        text = stderr.decode("utf-8", errors="replace")[-STDERR_CAPTURE_BYTES:]
        signal_number = -returncode if returncode is not None and returncode < 0 else None

        status: SandboxStatus
        detail: str | None
        if timed_out:
            status, detail = (
                SandboxStatus.TIMEOUT,
                f"wall-clock limit of {self._limits.wall_seconds}s exceeded; process group killed",
            )
        elif signal_number is not None:
            status, detail = (
                SandboxStatus.KILLED,
                f"killed by signal {signal_number} (a CPU/memory limit or the seccomp arch guard)",
            )
        elif result is None:
            status, detail = (
                SandboxStatus.CRASHED,
                f"the worker exited {returncode} without a parseable result document",
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
            signal=signal_number,
            usage=usage,
            detail=detail,
            stderr=text,
        )
