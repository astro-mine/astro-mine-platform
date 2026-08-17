# SPDX-License-Identifier: Apache-2.0
"""The structured result hand-back channel (bench.md §9; bench#30 AC3).

A sandboxed submission hands its result back **as data over a well-defined channel**, never as
shared in-process state: the worker writes a :class:`WorkerResult` document to ``result.json`` in
the one directory the sandbox lets it write to, and the evaluator parses it back. That is the whole
protocol — it crosses a process (and, in the container backend, a kernel-namespace) boundary, so
the evaluator never imports, unpickles, or otherwise executes anything the submission produced.

The channel carries three things the evaluator needs (bench#30): the per-seed **metrics**, a
structured **error** when the policy failed, and the **resource usage** the run cost.
:class:`SandboxOutcome` is the evaluator-side view: the parsed :class:`WorkerResult` (when there is
one) plus the facts only the parent knows — how the child exited, whether a limit killed it, and
the wall-clock it burned.

Everything here is ``core + pydantic`` only, so the worker side (which runs *inside* the sandbox)
stays dependency-clean.

Backlog: bench#30 — astro-mine-bench#30
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, ValidationError

__all__ = [
    "WORKER_RESULT",
    "ResourceUsage",
    "SandboxOutcome",
    "SandboxStatus",
    "WorkerInvocation",
    "WorkerMetric",
    "WorkerResult",
    "read_worker_result",
]

#: The file the sandboxed worker writes its :class:`WorkerResult` to, inside its output directory.
WORKER_RESULT = "result.json"


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


@dataclass(frozen=True, slots=True)
class WorkerInvocation:
    """What to roll in the sandbox: one seed of one scenario under one policy reference.

    ``policy_ref`` is the submitted ``module:attribute`` policy — a **string**, deliberately: it is
    resolved (imported) by the worker *inside* the sandbox and never by the evaluator.
    """

    scenario_id: str
    policy_ref: str
    seed: int


class WorkerMetric(_Model):
    """One metric's value on one seed, as computed inside the sandbox.

    Mirrors the columns of the Cloud eval worker's Parquet row (bench.md §5) so the two evaluation
    paths report the same per-seed record. ``value`` is ``None`` when the metric did not apply.
    """

    metric: str = Field(min_length=1)
    version: str
    unit: str
    direction: str
    aggregation: str
    value: float | None


class ResourceUsage(_Model):
    """What a sandboxed run cost (bench#30 AC3: *metrics, errors, resource usage*).

    ``wall_seconds`` is measured by the **evaluator** (the parent) and is authoritative.
    ``cpu_seconds`` / ``max_rss_bytes`` are ``getrusage`` figures the worker reports about itself;
    they are **advisory telemetry, not a control** — a hostile submission can lie about them. The
    caps themselves are enforced by the kernel (rlimits / cgroups), which no submission can lie its
    way past, so a false report cannot buy a submission more resource than its envelope allows.
    """

    wall_seconds: float = Field(ge=0.0)
    cpu_seconds: float | None = Field(default=None, ge=0.0)
    max_rss_bytes: int | None = Field(default=None, ge=0)


class WorkerResult(_Model):
    """The document the sandboxed worker writes — the only thing that crosses back in.

    ``ok`` distinguishes a scored rollout from a structured failure: an unresolvable ``policy_ref``,
    a policy that raised, or a metric that blew up are all reported *as data* in ``error`` rather
    than as an exception in the evaluator's process.
    """

    ok: bool
    scenario_id: str
    policy_ref: str
    seed: int
    #: The id of the runner that rolled this seed (bench#64) — provenance, not configuration. Only
    #: set on a scored rollout: a failure may not have got far enough to resolve a runner at all.
    runner: str | None = None
    metrics: tuple[WorkerMetric, ...] = ()
    error: str | None = None
    usage: ResourceUsage | None = None


class SandboxStatus(StrEnum):
    """How a sandboxed run ended, from the evaluator's side.

    ``OK`` the worker scored the seed; ``FAILED`` it ran but reported a structured error (a bad
    policy); ``TIMEOUT`` the wall-clock cap fired and the process group was killed; ``KILLED`` the
    kernel killed it (a CPU/memory rlimit, or the seccomp arch guard); ``CRASHED`` it exited without
    a parseable result (a hard interpreter failure, or a submission that scribbled on the channel).
    Every non-``OK`` status is a **rejection**, never a silently-accepted score.
    """

    OK = "ok"
    FAILED = "failed"
    TIMEOUT = "timeout"
    KILLED = "killed"
    CRASHED = "crashed"


class SandboxOutcome(_Model):
    """The evaluator-side view of one sandboxed run: the worker's result plus how it exited."""

    status: SandboxStatus
    invocation_seed: int
    result: WorkerResult | None = None
    exit_code: int | None = None
    #: The signal that killed the worker (POSIX ``-N`` exit), e.g. ``SIGKILL`` on a memory cap.
    signal: int | None = None
    usage: ResourceUsage
    #: A short operator-facing explanation; also the audit-log/OTel detail for a rejected run.
    detail: str | None = None
    #: The worker's stderr, truncated — captured for the audit trail, never executed or parsed.
    stderr: str = ""

    @property
    def scored(self) -> bool:
        """Whether this run produced a trustworthy per-seed score (the only accepting state)."""
        return self.status is SandboxStatus.OK and self.result is not None and self.result.ok


def read_worker_result(path: Path) -> WorkerResult | None:
    """Parse the hand-back document at ``path``; anything unparseable reads back as *no result*.

    The document is written by untrusted code, so it is validated as **data** and never trusted to
    be well-formed: a submission that scribbles garbage over its side of the channel is reported as
    a crash by the evaluator, never as a score.
    """
    if not path.is_file():
        return None
    try:
        return WorkerResult.model_validate_json(path.read_bytes())
    except (ValidationError, ValueError, json.JSONDecodeError, OSError):
        return None
