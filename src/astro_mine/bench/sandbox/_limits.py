# SPDX-License-Identifier: Apache-2.0
"""Sandbox resource limits + the network and filesystem postures (bench.md §9; conventions.md §9).

The leaderboard runs **arbitrary community code at scale** — bench.md §9 calls this "the central
safety concern for Bench". :class:`SandboxLimits` is the enforced envelope every submitted policy
executes inside: strict CPU / memory / time / process / file caps, a **no-egress-by-default**
network posture, and a **confined-by-default** filesystem posture. The limits are data, not policy:
each backend (:class:`~astro_mine.bench.sandbox.SubprocessSandbox`,
:class:`~astro_mine.bench.sandbox.ContainerSandbox`) maps them onto the kernel mechanism it has
(POSIX rlimits + seccomp-BPF + Landlock; container cgroups + a network-less, read-only namespace).

:class:`NetworkPolicy` and :class:`FilesystemPolicy` are each deliberately a two-value enum whose
restrictive member is the default and whose permissive member is an *explicit, auditable* opt-in —
there is no "best-effort" middle: a backend that cannot enforce the restrictive posture raises
:class:`~astro_mine.bench.sandbox.SandboxUnavailable` rather than running the code anyway
(fail-closed; conventions.md §9).

Backlog: bench#30, bench#36 — astro-mine-bench#36
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = ["DEFAULT_LIMITS", "FilesystemPolicy", "NetworkPolicy", "SandboxLimits"]


class NetworkPolicy(StrEnum):
    """Whether a sandboxed submission may create sockets at all (bench.md §9: *no network egress*).

    ``DENY`` is the default and the only posture the leaderboard uses for an untrusted submission:
    the backend must enforce it in the kernel or refuse to run. ``ALLOW`` exists so a *trusted*,
    self-authored policy (a local determinism gate, an operator's own baseline) can be run through
    the same machinery; it is never selected for a community submission.
    """

    DENY = "deny"
    ALLOW = "allow"


class FilesystemPolicy(StrEnum):
    """Whether a sandboxed submission's filesystem reach is confined (bench#36; bench.md §9).

    ``CONFINE`` is the default and the only posture the leaderboard uses for an untrusted
    submission: the backend confines the worker to the paths it legitimately needs — the
    interpreter, its libraries, the submission's import roots, and the run's scratch directory — so
    the **embargoed held-out seeds and the rest of the host are simply unreadable**. The subprocess
    backend enforces this with `Landlock <https://docs.kernel.org/userspace-api/landlock.html>`_
    (:mod:`astro_mine.bench.sandbox._landlock`); the container backend with a read-only rootfs.
    A backend that cannot enforce it refuses to run (fail-closed).

    ``HOST`` leaves the process with the service user's full filesystem view. It exists for the
    *trusted* local tier — a policy you wrote, run through the same machinery (the determinism gate,
    an operator's own baseline) — and is **never** selected for a community submission: under
    ``HOST`` a hostile policy can read the held-out seeds and encode them in its metric floats
    (``TRUST_BOUNDARY.md`` §4).
    """

    CONFINE = "confine"
    HOST = "host"


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    """The resource envelope a submitted policy executes inside (bench.md §9).

    ``cpu_seconds`` is a hard CPU-time cap (kernel ``RLIMIT_CPU`` / container CPU quota) and
    ``wall_seconds`` a hard wall-clock cap the *evaluator* enforces — both are needed: a policy that
    sleeps burns no CPU, and a policy that spins burns no extra wall-clock beyond its slice.
    ``memory_bytes`` caps the address space, ``output_bytes`` the size of any file the worker
    writes, ``max_processes`` blocks fork bombs, and ``max_open_files`` bounds descriptor
    exhaustion. ``gpus`` is the GPU cap — **0 exposes no GPU at all**, the default for a submitted
    policy. ``network`` is the egress posture (see :class:`NetworkPolicy`) and ``filesystem`` the
    confinement posture (see :class:`FilesystemPolicy`).
    """

    cpu_seconds: int = 60
    wall_seconds: float = 120.0
    memory_bytes: int = 2 * 1024**3
    output_bytes: int = 256 * 1024**2
    max_processes: int = 64
    max_open_files: int = 256
    gpus: int = 0
    network: NetworkPolicy = NetworkPolicy.DENY
    filesystem: FilesystemPolicy = FilesystemPolicy.CONFINE

    def __post_init__(self) -> None:
        for name in ("cpu_seconds", "memory_bytes", "output_bytes", "max_processes"):
            if getattr(self, name) < 1:
                raise ValueError(f"SandboxLimits.{name} must be >= 1")
        if self.max_open_files < 8:
            raise ValueError("SandboxLimits.max_open_files must be >= 8 (stdio + the result file)")
        if self.wall_seconds <= 0.0:
            raise ValueError("SandboxLimits.wall_seconds must be > 0")
        if self.gpus < 0:
            raise ValueError("SandboxLimits.gpus must be >= 0")


#: The default envelope for an untrusted submission: no egress, no GPU, confined filesystem, bounded
#: everything.
DEFAULT_LIMITS = SandboxLimits()
