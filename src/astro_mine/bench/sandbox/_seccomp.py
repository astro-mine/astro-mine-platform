# SPDX-License-Identifier: Apache-2.0
"""A seccomp-BPF filter that denies network egress to a submitted policy (bench.md §9).

bench.md §9 and conventions.md §9 require submitted policies to run "with **no network egress**",
sandboxed with **seccomp**. This module is that filter, built by hand as classic BPF and installed
with two ``prctl(2)`` calls — so the no-egress guarantee is **enforced by the kernel**, needs no
privileges, no container runtime, and no third-party dependency (the base package stays
``core + pydantic``).

The filter is installed in the *child*, after ``fork`` and before ``exec``
(:mod:`astro_mine.bench.sandbox._subprocess`), together with ``PR_SET_NO_NEW_PRIVS`` — which is
what makes an unprivileged ``PR_SET_SECCOMP`` legal *and* makes the filter survive ``exec`` and
every descendant. A submitted policy therefore cannot escape it, cannot regain the syscalls by
re-executing, and cannot drop it.

**What it denies:** every syscall that could create or use a socket — ``socket``, ``socketpair``,
``connect``, ``bind``, ``listen``, ``accept``/``accept4``, and the ``send*``/``recv*`` family —
plus ``ptrace`` (so a submission cannot attach to another process on the host). Denied calls return
``EPERM`` rather than killing the process, so a submission that probes the network gets a clean,
unrecoverable ``PermissionError`` and its rollout still scores; there is no "partially online"
state to negotiate. A syscall from a **foreign architecture** (including the x32 ABI, whose numbers
share ``AUDIT_ARCH_X86_64``) kills the process outright — that is the classic filter-bypass, and it
is closed here rather than merely denied.

**What it does not do:** it is not a filesystem sandbox and not a rootfs boundary — see
``TRUST_BOUNDARY.md`` for the full statement of what each backend does and does not protect
against. Where the deployment needs those too, run the
:class:`~astro_mine.bench.sandbox.ContainerSandbox` tier.

Backlog: bench#30 — astro-mine-bench#30
"""

from __future__ import annotations

import ctypes
import ctypes.util
import platform
import struct
import sys

__all__ = [
    "DENIED_SYSCALLS",
    "SeccompUnsupported",
    "build_egress_filter",
    "egress_filter_supported",
    "install_egress_filter",
]

# --- prctl / seccomp ABI constants (linux/prctl.h, linux/seccomp.h, linux/filter.h) --------------

_PR_SET_NO_NEW_PRIVS = 38
_PR_SET_SECCOMP = 22
_SECCOMP_MODE_FILTER = 2

_SECCOMP_RET_KILL_PROCESS = 0x80000000
_SECCOMP_RET_ERRNO = 0x00050000
_SECCOMP_RET_ALLOW = 0x7FFF0000
_EPERM = 1

_BPF_LD = 0x00
_BPF_W = 0x00
_BPF_ABS = 0x20
_BPF_JMP = 0x05
_BPF_JEQ = 0x10
_BPF_JGE = 0x30
_BPF_K = 0x00
_BPF_RET = 0x06

#: Byte offsets into ``struct seccomp_data``: the syscall number, then the calling architecture.
_OFFSET_NR = 0
_OFFSET_ARCH = 4

#: ``AUDIT_ARCH_*`` for the architectures the filter knows how to constrain. A machine that is not
#: in this table cannot be filtered, so the sandbox refuses to run untrusted code there.
_AUDIT_ARCH: dict[str, int] = {"x86_64": 0xC000003E, "aarch64": 0xC00000B7}

#: x86_64 syscall numbers carry this bit when issued through the **x32** ABI, which reports the same
#: ``AUDIT_ARCH_X86_64``. Filtering on the number alone would let ``socket|0x40000000`` straight
#: through, so anything at or above this bit is killed, not merely denied.
_X32_SYSCALL_BIT = 0x40000000

#: The syscalls a submitted policy may not make, per architecture: every socket-creating and
#: socket-using call (no egress — bench.md §9), plus ``ptrace`` (no attaching to host processes).
DENIED_SYSCALLS: dict[str, tuple[str, ...]] = {
    "x86_64": (
        "socket",
        "socketpair",
        "connect",
        "bind",
        "listen",
        "accept",
        "accept4",
        "sendto",
        "sendmsg",
        "recvfrom",
        "recvmsg",
        "ptrace",
    ),
    "aarch64": (
        "socket",
        "socketpair",
        "connect",
        "bind",
        "listen",
        "accept",
        "accept4",
        "sendto",
        "sendmsg",
        "recvfrom",
        "recvmsg",
        "ptrace",
    ),
}

_SYSCALL_NR: dict[str, dict[str, int]] = {
    "x86_64": {
        "socket": 41,
        "connect": 42,
        "accept": 43,
        "sendto": 44,
        "recvfrom": 45,
        "sendmsg": 46,
        "recvmsg": 47,
        "bind": 49,
        "listen": 50,
        "socketpair": 53,
        "ptrace": 101,
        "accept4": 288,
    },
    "aarch64": {
        "ptrace": 117,
        "socket": 198,
        "socketpair": 199,
        "bind": 200,
        "listen": 201,
        "accept": 202,
        "connect": 203,
        "sendto": 206,
        "recvfrom": 207,
        "sendmsg": 211,
        "recvmsg": 212,
        "accept4": 242,
    },
}


class SeccompUnsupported(RuntimeError):
    """The kernel/architecture cannot enforce the no-egress filter — the sandbox must refuse."""


class _SockFprog(ctypes.Structure):
    """``struct sock_fprog`` — the (length, program) pair ``PR_SET_SECCOMP`` takes."""

    _fields_ = (("len", ctypes.c_ushort), ("filter", ctypes.c_void_p))


def _statement(code: int, k: int) -> bytes:
    """One ``struct sock_filter`` with no jump targets."""
    return struct.pack("HBBI", code, 0, 0, k)


def _jump(code: int, k: int, jt: int, jf: int) -> bytes:
    """One conditional ``struct sock_filter``; ``jt``/``jf`` are *relative* instruction skips."""
    return struct.pack("HBBI", code, jt, jf, k)


def egress_filter_supported(machine: str | None = None) -> bool:
    """Whether :func:`install_egress_filter` can enforce no-egress on this platform.

    Linux only (seccomp is a Linux facility) and only on an architecture whose syscall numbering is
    pinned above. Anywhere else the sandbox **fails closed** rather than running untrusted code
    without the guarantee.
    """
    if sys.platform != "linux":
        return False
    return (platform.machine() if machine is None else machine) in _AUDIT_ARCH


def build_egress_filter(machine: str | None = None) -> bytes:
    """Assemble the classic-BPF program that denies the socket syscalls, as packed bytes.

    Pure and inspectable — the program is data, so it is unit-tested rather than trusted. Layout::

        0  load  arch
        1  jeq   AUDIT_ARCH_<this>  -> +1 (ok)          else fall through to the kill
        2  ret   KILL_PROCESS                            (foreign architecture)
        3  load  nr
        4  jge   0x40000000         -> kill              (the x32 ABI bypass)
        5.. jeq  <denied nr>        -> deny              (one per denied syscall)
        n  ret   ALLOW
        n+1 ret  ERRNO(EPERM)                            (a denied socket call)
        n+2 ret  KILL_PROCESS

    Raises :class:`SeccompUnsupported` on an architecture the filter does not know.
    """
    name = platform.machine() if machine is None else machine
    if name not in _AUDIT_ARCH:
        raise SeccompUnsupported(f"no seccomp egress filter for architecture {name!r}")
    numbers = [_SYSCALL_NR[name][call] for call in DENIED_SYSCALLS[name]]
    count = len(numbers)

    # Absolute indices of the three terminal returns, used to compute the relative jumps.
    allow_at = 5 + count
    deny_at = allow_at + 1
    kill_at = deny_at + 1

    program = [
        _statement(_BPF_LD | _BPF_W | _BPF_ABS, _OFFSET_ARCH),
        _jump(_BPF_JMP | _BPF_JEQ | _BPF_K, _AUDIT_ARCH[name], 1, 0),
        _statement(_BPF_RET | _BPF_K, _SECCOMP_RET_KILL_PROCESS),
        _statement(_BPF_LD | _BPF_W | _BPF_ABS, _OFFSET_NR),
        _jump(_BPF_JMP | _BPF_JGE | _BPF_K, _X32_SYSCALL_BIT, kill_at - 5, 0),
    ]
    program += [
        _jump(_BPF_JMP | _BPF_JEQ | _BPF_K, number, deny_at - (5 + index) - 1, 0)
        for index, number in enumerate(numbers)
    ]
    program += [
        _statement(_BPF_RET | _BPF_K, _SECCOMP_RET_ALLOW),
        _statement(_BPF_RET | _BPF_K, _SECCOMP_RET_ERRNO | _EPERM),
        _statement(_BPF_RET | _BPF_K, _SECCOMP_RET_KILL_PROCESS),
    ]
    return b"".join(program)


def install_egress_filter() -> None:
    """Install the no-egress filter on the **calling** process, irrevocably.

    Called in the forked child before ``exec`` — never in the evaluator, which would lose its own
    networking. Sets ``PR_SET_NO_NEW_PRIVS`` first (required for an unprivileged
    ``PR_SET_SECCOMP``, and it also denies the child any setuid privilege escalation), then loads
    the filter. Both the filter and ``no_new_privs`` are inherited across ``exec`` and by every
    descendant, so the submitted policy runs — and stays — inside them.

    Raises :class:`SeccompUnsupported` if the platform cannot enforce it or the kernel rejects it;
    the caller **must** treat that as fatal (fail closed) rather than exec'ing unfiltered.

    **Coverage note.** The ``prctl`` block below is marked ``no cover`` because it *cannot* be
    measured: it only ever runs in the forked child (where the coverage tracer does not follow), and
    calling it in the test process would irrevocably strip that process of sockets for the rest of
    the run. It is instead asserted **behaviourally** — ``tests/test_sandbox.py`` runs real hostile
    policies through the sandbox and proves TCP, UDP, and AF_UNIX sockets are all denied — which is
    a stronger check than line coverage would be. The BPF program itself
    (:func:`build_egress_filter`) is a pure function and *is* unit-tested.
    """
    if not egress_filter_supported():
        raise SeccompUnsupported(f"seccomp is unavailable on {sys.platform}/{platform.machine()}")
    _prctl_install(build_egress_filter())


def _prctl_install(program: bytes) -> None:  # pragma: no cover - runs only in the forked child
    """Load ``program`` into the calling process via ``prctl(2)`` — see the coverage note above."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    libc.prctl.restype = ctypes.c_int
    libc.prctl.argtypes = (
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_ulong,
    )
    # PR_SET_NO_NEW_PRIVS is what makes an unprivileged PR_SET_SECCOMP legal, and it also denies
    # the child setuid escalation. Both survive exec and are inherited by every descendant.
    if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        raise SeccompUnsupported(f"PR_SET_NO_NEW_PRIVS failed: errno {ctypes.get_errno()}")

    buffer = ctypes.create_string_buffer(program, len(program))
    fprog = _SockFprog(
        len(program) // ctypes.sizeof(ctypes.c_uint64), ctypes.cast(buffer, ctypes.c_void_p)
    )
    if libc.prctl(_PR_SET_SECCOMP, _SECCOMP_MODE_FILTER, ctypes.addressof(fprog), 0, 0) != 0:
        raise SeccompUnsupported(f"PR_SET_SECCOMP failed: errno {ctypes.get_errno()}")
