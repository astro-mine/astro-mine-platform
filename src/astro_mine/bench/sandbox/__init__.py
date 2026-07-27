"""Sandboxed execution of submitted policies (bench#30; bench.md §9; conventions.md §9).

**A submitted policy is untrusted code.** bench.md §9 calls this "the central safety concern for
Bench: the leaderboard runs arbitrary community code at scale", and requires it to execute
*out-of-process*, sandboxed, "with no network egress and strict CPU/GPU/memory/time limits". This
package is that boundary, and the leaderboard evaluator now runs **every** submission through it —
both the local ``policy_ref`` intake (RM-P0-BENCH-06) and the Hub-digest intake (RM-P1-BENCH-10),
neither of which may import a submission into the evaluator's process.

The pieces:

- :class:`SandboxLimits` / :class:`NetworkPolicy` — the enforced envelope: CPU, wall-clock, memory,
  file size, process and descriptor caps, no GPU, and **no egress by default**;
- :class:`SubprocessSandbox` — the default backend: the *existing eval-worker argv* (the one Cloud
  fans out, RM-P1-BENCH-11) in a capped, environment-scrubbed, session-isolated child, behind a
  **seccomp-BPF filter that denies every socket syscall**, a **Landlock allowlist that confines the
  filesystem** so the embargoed held-out seeds are unreadable (bench#36), and
  ``PR_SET_NO_NEW_PRIVS``;
- :class:`ContainerSandbox` — the recommended posture for a public deployment: the same argv inside
  a ``--network=none --read-only --cap-drop=ALL`` container (``--runtime=runsc`` for **gVisor**),
  which adds the rootfs and namespace boundaries a bare subprocess cannot draw;
- :class:`WorkerResult` / :class:`SandboxOutcome` — the **structured hand-back channel**: the worker
  writes a result document, the evaluator parses it as data. No shared in-process state ever crosses
  back;
- :class:`SandboxScorer` — the :class:`PolicyScorer` the leaderboard evaluates through. It scores a
  ``policy_ref`` *string*, never a live ``Policy``, and aggregates the sandboxed per-seed values
  with the same kernel the local tier uses — so a sandboxed scorecard is **byte-identical** to the
  workstation one.

Everything **fails closed**. A backend that cannot enforce the requested isolation raises
:class:`SandboxUnavailable` rather than running the submission anyway, and a seed that times out, is
killed by a limit, or crashes rejects the whole submission rather than scoring the seeds that
happened to finish.

The trust boundary — what each backend does and does **not** protect against, and the residual risk
— is stated in ``TRUST_BOUNDARY.md`` at the repo root. Read it before deploying.

Dependency-clean (``core + pydantic``): the seccomp filter is built by hand and installed through
``ctypes``, so no sandboxing library enters the base package, and the local scoring tier is
untouched.

Backlog: bench#30 — https://github.com/astro-mine/astro-mine-bench/issues/30
"""

from __future__ import annotations

from astro_mine.bench.sandbox._channel import (
    WORKER_RESULT,
    ResourceUsage,
    SandboxOutcome,
    SandboxStatus,
    WorkerInvocation,
    WorkerMetric,
    WorkerResult,
    read_worker_result,
)
from astro_mine.bench.sandbox._container import (
    CONTAINER_OUTPUT_DIR,
    ContainerSandbox,
    container_runtime_available,
)
from astro_mine.bench.sandbox._landlock import (
    LandlockUnsupported,
    filesystem_read_roots,
    landlock_abi,
    landlock_supported,
    restrict_filesystem,
    supported_access_rights,
)
from astro_mine.bench.sandbox._limits import (
    DEFAULT_LIMITS,
    FilesystemPolicy,
    NetworkPolicy,
    SandboxLimits,
)
from astro_mine.bench.sandbox._score import (
    InProcessScorer,
    PolicyScorer,
    SandboxScorer,
    SubmissionExecutionError,
)
from astro_mine.bench.sandbox._seccomp import (
    DENIED_SYSCALLS,
    SeccompUnsupported,
    build_egress_filter,
    egress_filter_supported,
    install_egress_filter,
)
from astro_mine.bench.sandbox._subprocess import (
    STDERR_CAPTURE_BYTES,
    Sandbox,
    SandboxError,
    SandboxUnavailable,
    SubprocessSandbox,
    rlimit_settings,
    sandbox_environment,
    worker_argv,
)

__all__ = [
    "CONTAINER_OUTPUT_DIR",
    "DEFAULT_LIMITS",
    "DENIED_SYSCALLS",
    "STDERR_CAPTURE_BYTES",
    "WORKER_RESULT",
    "ContainerSandbox",
    "FilesystemPolicy",
    "InProcessScorer",
    "LandlockUnsupported",
    "NetworkPolicy",
    "PolicyScorer",
    "ResourceUsage",
    "Sandbox",
    "SandboxError",
    "SandboxLimits",
    "SandboxOutcome",
    "SandboxScorer",
    "SandboxStatus",
    "SandboxUnavailable",
    "SeccompUnsupported",
    "SubmissionExecutionError",
    "SubprocessSandbox",
    "WorkerInvocation",
    "WorkerMetric",
    "WorkerResult",
    "build_egress_filter",
    "container_runtime_available",
    "egress_filter_supported",
    "filesystem_read_roots",
    "install_egress_filter",
    "landlock_abi",
    "landlock_supported",
    "read_worker_result",
    "restrict_filesystem",
    "rlimit_settings",
    "sandbox_environment",
    "supported_access_rights",
    "worker_argv",
]
