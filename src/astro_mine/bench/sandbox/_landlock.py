# SPDX-License-Identifier: Apache-2.0
"""A Landlock allowlist that confines a submitted policy's filesystem reach (bench#36; bench.md §9).

bench#30 shipped the out-of-process sandbox but left one boundary undrawn: a submitted policy runs
as the service user and can ``open()`` any file that user can — including the embargoed
``embargo/*/heldout_seeds.json`` the leaderboard's whole anti-overfitting story depends on. seccomp
denies the *socket* that would exfiltrate it, but not the *read*, and the metric floats the policy
returns are a low-bandwidth channel out (``TRUST_BOUNDARY.md`` §4). This module closes the read.

It is the filesystem counterpart of :mod:`astro_mine.bench.sandbox._seccomp`: a kernel access
control built by hand and installed with raw syscalls through ``ctypes``, in the *child*, after
``fork`` and before ``exec`` — so it needs no privilege, no container runtime, and no third-party
dependency (the base package stays ``core + pydantic``). `Landlock
<https://docs.kernel.org/userspace-api/landlock.html>`_ is **allowlist-only**: a restricted process
may touch only the paths granted to it, and — like seccomp + ``no_new_privs`` — the restriction is
inherited across ``exec`` and by every descendant and can never be dropped.

**What it grants:** read+execute on the interpreter, its standard library and site-packages, the
submission's declared import roots, and a small set of system paths a Python process needs (the C
runtime, the dynamic loader, CA certificates, timezone data, ``/dev/urandom``); read+write on the
run's single scratch/result directory. **Everything else — the repo tree, the operator's home, SSH
keys, and crucially the embargo directory — is simply not on the list, so a read of it returns
``EACCES``.** The evaluator hands the held-out set to the worker as *integers on the argv*, never as
a file the worker opens, so nothing legitimate needs the embargo path (bench#30; ``_eval.py``).

It also, as a free consequence, closes the environment-scrub bypass: ``/proc`` is not granted, so a
same-uid worker can no longer read the evaluator's secrets out of ``/proc/<pid>/environ`` behind the
``_ENV_ALLOWLIST`` (``_subprocess.py``).

**What it does not do:** it is a *filesystem* boundary, not a network or namespace one — no egress
(that is :mod:`._seccomp`), and it does not hide the host's process table or PIDs (that is the
container tier). And it depends on the evaluator's filesystem supporting Landlock: every standard
Linux filesystem (ext4/xfs/btrfs/overlay/tmpfs) does, but some network/9p mounts silently deny even
granted paths — there a confined worker cannot start and the submission is **rejected**, never run
unconfined (fail-closed; ``TRUST_BOUNDARY.md`` §5). Where the deployment needs the process-table and
kernel-surface boundaries too, run the :class:`~astro_mine.bench.sandbox.ContainerSandbox` tier.

Backlog: bench#36 — astro-mine-bench#36
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import stat
import subprocess
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

__all__ = [
    "SYSTEM_READ_PATHS",
    "SYSTEM_WRITE_PATHS",
    "LandlockUnsupported",
    "filesystem_read_roots",
    "interpreter_read_roots",
    "landlock_abi",
    "landlock_supported",
    "restrict_filesystem",
    "supported_access_rights",
]

# --- Landlock ABI constants (linux/landlock.h; syscalls in the arch-generic 444/445/446 range) ----

_SYS_LANDLOCK_CREATE_RULESET = 444
_SYS_LANDLOCK_ADD_RULE = 445
_SYS_LANDLOCK_RESTRICT_SELF = 446

_LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
_LANDLOCK_RULE_PATH_BENEATH = 1

_PR_SET_NO_NEW_PRIVS = 38

#: The filesystem access rights Landlock can gate, newest-last. The bit values are the stable
#: ``LANDLOCK_ACCESS_FS_*`` flags; the grouping by the ABI that introduced each is what keeps the
#: ruleset legal on an older kernel — asking to handle a right the running ABI does not know is
#: ``EINVAL``, so :func:`supported_access_rights` intersects with what this kernel supports.
_ACCESS_FS: dict[str, int] = {
    "execute": 1 << 0,
    "write_file": 1 << 1,
    "read_file": 1 << 2,
    "read_dir": 1 << 3,
    "remove_dir": 1 << 4,
    "remove_file": 1 << 5,
    "make_char": 1 << 6,
    "make_dir": 1 << 7,
    "make_reg": 1 << 8,
    "make_sock": 1 << 9,
    "make_fifo": 1 << 10,
    "make_block": 1 << 11,
    "make_sym": 1 << 12,
    "refer": 1 << 13,  # ABI 2
    "truncate": 1 << 14,  # ABI 3
    "ioctl_dev": 1 << 15,  # ABI 5
}

#: Which rights first become available at each Landlock ABI version.
_RIGHTS_BY_ABI: dict[int, tuple[str, ...]] = {
    1: tuple(r for r in _ACCESS_FS if r not in ("refer", "truncate", "ioctl_dev")),
    2: ("refer",),
    3: ("truncate",),
    5: ("ioctl_dev",),
}

#: Rights that only make sense on a directory — attaching them to a rule whose target is a regular
#: file or a device is ``EINVAL``, so a file-target rule keeps only :data:`_FILE_RIGHTS`.
_FILE_RIGHTS = frozenset({"execute", "read_file", "write_file", "truncate", "ioctl_dev"})

#: Read a submission's code, but never write it: the interpreter, its libraries, and import roots.
_READ_ACCESS = ("execute", "read_file", "read_dir")

#: The scratch/result directory the worker owns — read, write, and create within, but not execute.
_WRITE_ACCESS = (
    "read_file",
    "read_dir",
    "write_file",
    "make_reg",
    "make_dir",
    "make_sym",
    "make_fifo",
    "make_sock",
    "remove_file",
    "remove_dir",
    "truncate",
)

#: System paths a CPython process needs to *start and run* that live outside its own prefix: the C
#: runtime and dynamic loader, CA-certificate bundles (TLS trust, though egress is denied), timezone
#: data, and the kernel RNG. Read-only. ``embargo/`` — and the rest of the host — is deliberately
#: absent; a path that does not exist on a given host is skipped when the rule is added.
SYSTEM_READ_PATHS: tuple[str, ...] = (
    "/usr/lib",
    "/usr/lib64",
    "/usr/local/lib",
    "/lib",
    "/lib64",
    "/usr/bin",
    "/bin",
    "/etc/ld.so.cache",
    "/etc/ld.so.conf",
    "/etc/ld.so.conf.d",
    "/etc/ssl",
    "/etc/pki",
    "/etc/localtime",
    "/usr/share/zoneinfo",
    "/dev/urandom",
    "/dev/random",
    "/dev/zero",
    "/dev/null",
)

#: The single writable device a worker legitimately needs beyond its own scratch directory.
SYSTEM_WRITE_PATHS: tuple[str, ...] = ("/dev/null",)

#: The interpreter probe: ask the *worker's* Python (which may be a different venv) where it imports
#: from, in a clean ``-I`` process so no inherited ``PYTHONPATH`` skews the answer.
_ROOTS_PROBE = (
    "import json,sys,sysconfig;"
    "p=sysconfig.get_paths();"
    "print(json.dumps(sorted({x for x in ("
    "*[e for e in sys.path if e],sys.prefix,sys.base_prefix,sys.exec_prefix,"
    "p['stdlib'],p['platstdlib'],p['purelib'],p['platlib']) if x})))"
)


class LandlockUnsupported(RuntimeError):
    """The kernel cannot enforce a Landlock ruleset here — the sandbox must refuse (fail-closed)."""


def landlock_abi() -> int:
    """The kernel's Landlock ABI version, or ``<= 0`` if Landlock is unavailable.

    A version-probe ``landlock_create_ruleset(NULL, 0, VERSION)`` — the documented way to ask
    without creating anything. Non-Linux returns ``-1`` (the syscall does not exist).
    """
    if sys.platform != "linux":
        return -1
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(
        ctypes.c_long(_SYS_LANDLOCK_CREATE_RULESET),
        None,
        ctypes.c_size_t(0),
        ctypes.c_uint32(_LANDLOCK_CREATE_RULESET_VERSION),
    )
    return int(result)


def landlock_supported() -> bool:
    """Whether :func:`restrict_filesystem` can confine a process on this kernel (Linux, ABI ≥ 1).

    A kernel-level check only. Whether a *particular filesystem* honours the ruleset is a separate
    question the kernel does not answer up front (a granted path on an unsupporting mount is denied
    at access time); a confined worker on such a mount simply cannot start, and is rejected.
    """
    return landlock_abi() >= 1


def supported_access_rights(abi: int) -> dict[str, int]:
    """The ``name -> bit`` filesystem rights a ruleset may ``handle`` at Landlock ABI ``abi``.

    Pure, so the ABI gating is unit-tested directly: handling a right the running kernel does not
    know makes ``landlock_create_ruleset`` fail, so the handled set is clamped to the ABI.
    """
    rights: dict[str, int] = {}
    for version, names in sorted(_RIGHTS_BY_ABI.items()):
        if version <= abi:
            rights.update({name: _ACCESS_FS[name] for name in names})
    return rights


def _mask(names: Iterable[str], rights: dict[str, int], *, is_dir: bool) -> int:
    """OR the named rights that exist at this ABI, dropping dir-only rights on a file target."""
    value = 0
    for name in names:
        if name in rights and (is_dir or name in _FILE_RIGHTS):
            value |= rights[name]
    return value


def interpreter_read_roots(python: str) -> tuple[str, ...]:
    """The import roots ``python`` reads its standard library and installed packages from.

    Runs the target interpreter once (it may be a different venv than the evaluator's) and parses
    the directories it would import from — its ``sys.path`` entries, prefixes, and the ``sysconfig``
    library paths. Only existing directories are returned; a probe failure yields ``()`` and the
    caller falls back to granting nothing extra (the worker then fails to import and is rejected —
    fail-closed, never unconfined).
    """
    try:
        completed = subprocess.run(
            [python, "-I", "-c", _ROOTS_PROBE],
            capture_output=True,
            timeout=30.0,
            check=True,
        )
        entries = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return ()
    return tuple(entry for entry in entries if isinstance(entry, str) and Path(entry).is_dir())


def filesystem_read_roots(python: str, *, extra_roots: Sequence[str] = ()) -> tuple[str, ...]:
    """Every path a confined worker on ``python`` may read: interpreter + system + import roots.

    ``extra_roots`` are the submission's declared import roots (the sandbox's ``python_path``); the
    interpreter binary itself and its realpath are included so ``exec`` of it is permitted. The
    result is deduplicated and order-stable. The repo root — hence ``embargo/`` — is never added
    here: it is not an import root in a real deployment (the package installs into site-packages),
    which is exactly what makes the held-out set unreachable.
    """
    roots: list[str] = [python, os.path.realpath(python)]
    roots.extend(interpreter_read_roots(python))
    roots.extend(SYSTEM_READ_PATHS)
    roots.extend(str(root) for root in extra_roots)
    seen: dict[str, None] = {}
    for root in roots:
        if root and root not in seen:
            seen[root] = None
    return tuple(seen)


class _RulesetAttr(ctypes.Structure):
    """``struct landlock_ruleset_attr`` (the ABI-1 prefix: the handled filesystem-access mask)."""

    _fields_ = (("handled_access_fs", ctypes.c_uint64),)


class _PathBeneathAttr(ctypes.Structure):
    """``struct landlock_path_beneath_attr`` — an allowed-access mask over a parent directory fd."""

    _pack_ = 1
    _fields_ = (("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32))


def restrict_filesystem(read_roots: Sequence[str], write_roots: Sequence[str]) -> None:
    """Confine the **caller** to ``read_roots`` (read+exec) and ``write_roots`` (read+write).

    Called in the forked child before ``exec`` — never in the evaluator, which would lose access to
    its own database and secrets. Sets ``PR_SET_NO_NEW_PRIVS`` first (required for an unprivileged
    ``landlock_restrict_self``, and it also denies the child setuid escalation), builds a ruleset
    handling every filesystem right this kernel's ABI supports, grants each root the appropriate
    subset, and restricts. The ruleset and ``no_new_privs`` survive ``exec`` and are inherited by
    every descendant, so the submitted policy runs — and stays — inside them.

    Raises :class:`LandlockUnsupported` if the platform cannot enforce it or the kernel rejects a
    step; the caller **must** treat that as fatal (fail-closed) rather than exec'ing unconfined.

    **Coverage note.** The syscall block is marked ``no cover`` for the same reason
    :func:`~astro_mine.bench.sandbox._seccomp.install_egress_filter` is: it only ever runs in the
    forked child (which the coverage tracer does not follow), and calling it in the test process
    would irrevocably confine that process for the rest of the run. It is asserted **behaviourally**
    instead — ``tests/test_sandbox.py`` runs a real policy that tries to read the embargo file
    through the sandbox and proves the read fails. The ruleset it builds
    (:func:`supported_access_rights`, :func:`filesystem_read_roots`) is pure and unit-tested.
    """
    abi = landlock_abi()
    if abi < 1:
        raise LandlockUnsupported(f"Landlock is unavailable on {sys.platform} (abi={abi})")
    _restrict(abi, read_roots, write_roots)


def _restrict(  # pragma: no cover - runs only in the forked child; see the coverage note above
    abi: int, read_roots: Sequence[str], write_roots: Sequence[str]
) -> None:
    """Build and enter the Landlock ruleset via raw syscalls — see :func:`restrict_filesystem`."""
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    libc.syscall.restype = ctypes.c_long

    rights = supported_access_rights(abi)
    handled = 0
    for bit in rights.values():
        handled |= bit

    attr = _RulesetAttr(handled_access_fs=handled)
    ruleset_fd = libc.syscall(
        ctypes.c_long(_SYS_LANDLOCK_CREATE_RULESET),
        ctypes.byref(attr),
        ctypes.c_size_t(ctypes.sizeof(attr)),
        ctypes.c_uint32(0),
    )
    if ruleset_fd < 0:
        raise LandlockUnsupported(f"landlock_create_ruleset failed: errno {ctypes.get_errno()}")

    try:
        for paths, access in ((read_roots, _READ_ACCESS), (write_roots, _WRITE_ACCESS)):
            for path in paths:
                _add_path_rule(libc, ruleset_fd, path, access, rights)

        if libc.prctl(_PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
            raise LandlockUnsupported(f"PR_SET_NO_NEW_PRIVS failed: errno {ctypes.get_errno()}")
        if (
            libc.syscall(
                ctypes.c_long(_SYS_LANDLOCK_RESTRICT_SELF),
                ctypes.c_int(ruleset_fd),
                ctypes.c_uint32(0),
            )
            != 0
        ):
            raise LandlockUnsupported(f"landlock_restrict_self failed: errno {ctypes.get_errno()}")
    finally:
        os.close(ruleset_fd)


def _add_path_rule(  # pragma: no cover - runs only in the forked child
    libc: ctypes.CDLL,
    ruleset_fd: int,
    path: str,
    access: Sequence[str],
    rights: dict[str, int],
) -> None:
    """Grant ``access`` beneath ``path``; a path that cannot be opened is skipped, not fatal.

    A non-existent or unreadable root (a system path absent on this host, a stale import root) is
    not an error — the allowlist is a ceiling, and a path missing from it just stays denied.
    """
    try:
        parent_fd = os.open(path, os.O_PATH | os.O_CLOEXEC)
    except OSError:
        return
    try:
        is_dir = stat.S_ISDIR(os.fstat(parent_fd).st_mode)
        allowed = _mask(access, rights, is_dir=is_dir)
        if not allowed:
            return
        beneath = _PathBeneathAttr(allowed_access=allowed, parent_fd=parent_fd)
        if (
            libc.syscall(
                ctypes.c_long(_SYS_LANDLOCK_ADD_RULE),
                ctypes.c_int(ruleset_fd),
                ctypes.c_int(_LANDLOCK_RULE_PATH_BENEATH),
                ctypes.byref(beneath),
                ctypes.c_uint32(0),
            )
            != 0
        ):
            raise LandlockUnsupported(
                f"landlock_add_rule({path!r}) failed: errno {ctypes.get_errno()}"
            )
    finally:
        os.close(parent_fd)
