"""Environment / lockfile pinning for reproducible runs (conventions.md §5, §7; bench.md §10).

The harness folds a ``toolchain`` of pinned inputs — the ``uv.lock`` digest, the bench code version,
and the runner identity — into ``resolve_scenario`` via the slot the resolver reserves for BENCH-04,
so any change to the pinned environment changes the resolved ``scenario_hash``. The machine
fingerprint (interpreter, platform) is recorded separately in an :class:`EnvironmentStamp` and kept
OUT of the hash, so a Result reproduces byte-for-byte across machines (mirrors Sim's split).

The lockfile pins the environment *the run executed in*, which is not Bench's own. When Bench is
installed as a dependency, the dependency set that decides whether a run reproduces belongs to the
consumer — the engines, solvers, and Core pin that produced the physics. So the lockfile is resolved
from the caller, the environment, or the working directory, and never from Bench's install location:
deriving it from ``__file__`` only ever resolved in a source checkout, which broke the
``pip install astro-mine-bench`` tier that bench.md §7 requires to always work.

Backlog: RM-P0-BENCH-04 — https://github.com/astro-mine/astro-mine-bench/issues/4
"""

from __future__ import annotations

import os
import platform
from pathlib import Path

from astro_mine.bench._version import __version__
from astro_mine.bench.harness._models import EnvironmentStamp
from astro_mine.bench.scenario._hash import content_hash

__all__ = [
    "LOCKFILE_ENV",
    "LOCKFILE_NAME",
    "LockfileNotFound",
    "build_toolchain",
    "environment_stamp",
    "lockfile_digest",
    "resolve_lockfile",
]

#: Names the lockfile pinning the environment a run executes in; overrides directory discovery.
LOCKFILE_ENV = "ASTRO_MINE_BENCH_LOCKFILE"

#: The lockfile searched for upward from the working directory.
LOCKFILE_NAME = "uv.lock"


class LockfileNotFound(FileNotFoundError):
    """No lockfile resolved — the determinism gate fails closed rather than skipping the check."""


def resolve_lockfile(path: Path | str | None = None, *, start: Path | None = None) -> Path:
    """Locate the lockfile pinning the environment this run executes in.

    Resolution order: an explicit *path*, then ``$ASTRO_MINE_BENCH_LOCKFILE``, then the nearest
    ``uv.lock`` at or above *start* (the working directory by default) — the lockfile of whichever
    project is being benchmarked. Raises :class:`LockfileNotFound` when none resolves: an unpinned
    environment cannot be attested, so the gate fails closed instead of scoring an unprovenanced run
    (bench.md §2 principle 1).
    """
    if path is not None:
        candidate = Path(path)
        if not candidate.is_file():
            raise LockfileNotFound(f"lockfile {candidate} does not exist")
        return candidate

    override = os.environ.get(LOCKFILE_ENV)
    if override:
        candidate = Path(override)
        if not candidate.is_file():
            raise LockfileNotFound(f"${LOCKFILE_ENV} points at {candidate}, which does not exist")
        return candidate

    origin = (start or Path.cwd()).resolve()
    for directory in (origin, *origin.parents):
        candidate = directory / LOCKFILE_NAME
        if candidate.is_file():
            return candidate

    raise LockfileNotFound(
        f"no {LOCKFILE_NAME} found in {origin} or any parent directory. A Result pins the "
        f"environment it was produced in, so the gate needs the lockfile of the project being "
        f"benchmarked. Pass lockfile=... to reproduce()/assert_reproducible(), or set "
        f"${LOCKFILE_ENV}."
    )


def lockfile_digest(path: Path | str | None = None) -> str:
    """The ``sha256:`` digest of the ``uv.lock`` dependency lockfile."""
    lockfile = resolve_lockfile(path)
    # Keyed by the fixed name rather than the resolved path, so the digest turns on the lockfile's
    # contents alone — a consumer's lockfile hashes the same wherever on disk it happens to live.
    return content_hash({"uv.lock": lockfile.read_text(encoding="utf-8")})


def build_toolchain(runner: str, *, lockfile: Path | str | None = None) -> dict[str, str]:
    """The pinned-input toolchain folded into the resolved scenario hash for a run."""
    return {"bench": __version__, "lockfile": lockfile_digest(lockfile), "runner": runner}


def environment_stamp() -> EnvironmentStamp:
    """The machine fingerprint recorded (but not hashed) for audit."""
    return EnvironmentStamp(python=platform.python_version(), platform=platform.platform())
