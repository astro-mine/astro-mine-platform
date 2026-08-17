# SPDX-License-Identifier: Apache-2.0
"""Deterministic reproducibility harness + determinism gates (RM-P0-BENCH-04).

Containerized, seeded, lockfile-pinned execution with determinism gates that fail CI on
non-reproducibility — the platform's regression oracle (bench.md §10). :func:`reproduce` runs a
scenario ``runs`` times and reports whether it reproduces byte-for-byte; :func:`assert_reproducible`
is the gate that raises on drift; :func:`replay` is sampled re-execution from a stored
:class:`Result`. The harness is runner-agnostic — inject a :class:`Runner` (Sim, once wired); the
pure :func:`reference_runner` drives the oracle today.

Backlog: RM-P0-BENCH-04 — astro-mine-bench#4
"""

from __future__ import annotations

from astro_mine.bench.harness._models import (
    EnvironmentStamp,
    ReproductionReport,
    Result,
    SeedResult,
)
from astro_mine.bench.harness._reproduce import (
    DeterminismError,
    assert_reproducible,
    replay,
    reproduce,
)
from astro_mine.bench.harness._runner import REFERENCE_RUNNER_ID, Runner, reference_runner
from astro_mine.bench.harness._toolchain import (
    LOCKFILE_ENV,
    LOCKFILE_NAME,
    LockfileNotFound,
    build_toolchain,
    environment_stamp,
    lockfile_digest,
    resolve_lockfile,
)

__all__ = [
    "LOCKFILE_ENV",
    "LOCKFILE_NAME",
    "REFERENCE_RUNNER_ID",
    "DeterminismError",
    "EnvironmentStamp",
    "LockfileNotFound",
    "ReproductionReport",
    "Result",
    "Runner",
    "SeedResult",
    "assert_reproducible",
    "build_toolchain",
    "environment_stamp",
    "lockfile_digest",
    "reference_runner",
    "replay",
    "reproduce",
    "resolve_lockfile",
]
