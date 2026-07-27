"""Async submission lifecycle: job state + rate limiting (RM-P1-BENCH-10; bench.md §7, §9).

A hosted submission is not a synchronous scoring call — it is a **job** that moves through a
lifecycle (``queued`` → ``running`` → ``scored`` → ``ranked``, or a terminal ``flagged`` /
``rejected``) while an evaluation worker resolves it from Hub, runs it under submit-policy-we-run,
verifies its provenance, and ranks it (bench.md §3, §7). This module models that lifecycle plus
the two pieces of shared coordination state a public edge needs:

- :class:`JobStore` — the job-state record keyed by submission id (a **Redis**-backed hash in the
  deployment; :class:`InMemoryJobStore` here);
- :class:`RateLimiter` — a fixed-window counter that bounds brute-force seed-search abuse
  (bench.md §9; a **Redis** ``INCR``/``EXPIRE`` in the deployment; :class:`InMemoryRateLimiter`
  here).

The lifecycle **transport** (NATS/JetStream) is deployment plumbing; the states, transitions, and
their coordination state live here so the *same* lifecycle runs in-process for the local tier and
the tests. A real deployment swaps the backends behind these protocols, not the transitions.

Backlog: RM-P1-BENCH-10 — https://github.com/astro-mine/astro-mine-bench/issues/18
"""

from __future__ import annotations

from collections import deque
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "InMemoryJobQueue",
    "InMemoryJobStore",
    "InMemoryRateLimiter",
    "JobQueue",
    "JobRecord",
    "JobStore",
    "RateLimitError",
    "RateLimiter",
    "SubmissionEnvelope",
    "SubmissionStatus",
]


class SubmissionStatus(StrEnum):
    """The lifecycle of a hosted submission (bench.md §7).

    ``QUEUED`` accepted and awaiting a worker; ``RUNNING`` resolving + executing under
    submit-policy-we-run; ``SCORED`` scored on the held-out seeds; ``RANKED`` verified and placed
    on the board; ``FLAGGED`` an integrity failure (provenance re-execution mismatch); ``REJECTED``
    never ran (bad digest, manifest/interface mismatch, or a rate-limit refusal).
    """

    QUEUED = "queued"
    RUNNING = "running"
    SCORED = "scored"
    RANKED = "ranked"
    FLAGGED = "flagged"
    REJECTED = "rejected"


#: The states from which no transition is allowed — a job is done.
TERMINAL_STATUSES: frozenset[SubmissionStatus] = frozenset(
    {SubmissionStatus.RANKED, SubmissionStatus.FLAGGED, SubmissionStatus.REJECTED}
)


class JobRecord(BaseModel):
    """A submission job's current lifecycle state — status, a human detail, and its result id.

    A hosted submission is accepted as a **job** keyed by a deterministic ``job_id`` ticket (the
    handle the client polls), which links to the content-addressed ``result_id`` (the
    :class:`~astro_mine.bench.leaderboard._models.Submission` id) once the job reaches ``ranked`` /
    ``flagged``. Deliberately **timestamp-free**: the record is part of the reproducible provenance
    surface (the deployment's Redis hash adds TTLs out-of-band). ``detail`` explains a terminal
    ``flagged``/``rejected`` for the public status endpoint.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str = Field(min_length=1)
    status: SubmissionStatus
    detail: str | None = None
    result_id: str | None = None


class JobStore(Protocol):
    """Persistence for submission-job lifecycle records (a Redis hash in the deployment)."""

    def put_job(self, job: JobRecord) -> None:
        """Persist ``job`` (replaces any prior record for its ``job_id``)."""
        ...

    def get_job(self, job_id: str) -> JobRecord | None:
        """Return the job for ``job_id``, or ``None`` if unknown."""
        ...


class InMemoryJobStore:
    """A process-local :class:`JobStore` — the dependency-clean default backend."""

    def __init__(self) -> None:
        self._by_id: dict[str, JobRecord] = {}

    def put_job(self, job: JobRecord) -> None:
        self._by_id[job.job_id] = job

    def get_job(self, job_id: str) -> JobRecord | None:
        return self._by_id.get(job_id)


class SubmissionEnvelope(BaseModel):
    """The message a submission rides across the pipeline's **async hop** (bench.md §7, §10).

    A hosted submission is accepted by the request handler and evaluated by a worker; between them
    sits a queue (NATS/JetStream in the deployment, :class:`InMemoryJobQueue` here). ``headers``
    carries the **W3C trace context** across that hop — the request handler injects it, the worker
    extracts it — so ``submit → evaluate → score → rank`` is one OpenTelemetry trace rather than two
    disconnected ones (bench#32). It is an opaque carrier: Bench writes and reads it only through
    :mod:`astro_mine.bench.telemetry`.

    ``subject`` is the **authenticated** principal (bench#29) — the key quotas and the audit trail
    are bound to. It is set from the verified bearer token, never from the request body.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str = Field(min_length=1)
    scenario_id: str = Field(min_length=1)
    #: The submission's handle: a Hub reference, or a local ``module:attribute`` policy reference.
    reference: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    headers: dict[str, str] = Field(default_factory=dict)


class JobQueue(Protocol):
    """The submit→evaluate hop (a NATS/JetStream subject in the deployment).

    :meth:`depth` is the **queue depth** bench.md §10 puts on the dashboard — the primary
    back-pressure signal for the evaluation fleet (bench.md §8).
    """

    def publish(self, envelope: SubmissionEnvelope) -> None:
        """Enqueue an accepted submission for evaluation."""
        ...

    def consume(self) -> SubmissionEnvelope | None:
        """Take the next submission to evaluate, or ``None`` when the queue is empty."""
        ...

    def depth(self) -> int:
        """How many accepted submissions are still awaiting a worker."""
        ...


class InMemoryJobQueue:
    """A process-local FIFO :class:`JobQueue` — the dependency-clean default backend."""

    def __init__(self) -> None:
        self._pending: deque[SubmissionEnvelope] = deque()

    def publish(self, envelope: SubmissionEnvelope) -> None:
        self._pending.append(envelope)

    def consume(self) -> SubmissionEnvelope | None:
        return self._pending.popleft() if self._pending else None

    def depth(self) -> int:
        return len(self._pending)


class RateLimitError(Exception):
    """Raised when a submitter exceeds its allowed submissions in the current window."""


class RateLimiter(Protocol):
    """A per-identity submission-rate gate (bench.md §9).

    Keyed on the **authenticated** principal (``Principal.identity``) since bench#29 — the pre-auth
    leaderboard keyed it on a client-supplied ``identity`` field, which a submitter could simply
    change to reset their own counter.
    """

    def check(self, identity: str) -> None:
        """Count one submission by ``identity``; raise :class:`RateLimitError` over the limit."""
        ...

    def observed(self, identity: str) -> int:
        """How many submissions ``identity`` has made in the current window.

        Feeds the policy engine's per-role **submission quota** (bench.md §9): the flat rate limit
        bounds brute-force seed search, while the quota is the role-aware cap layered on top.
        """
        ...


class InMemoryRateLimiter:
    """A process-local fixed-window rate limiter (a Redis ``INCR``/``EXPIRE`` in the deployment).

    ``limit`` submissions are allowed per identity before the window is manually
    :meth:`reset`. Kept clock-free — deterministic for tests — so the window is advanced by the
    caller/scheduler rather than a wall-clock TTL; the deployment's Redis backend applies a real
    ``EXPIRE``.
    """

    def __init__(self, *, limit: int = 60) -> None:
        if limit < 1:
            raise ValueError("rate-limit must allow at least one submission per window")
        self._limit = limit
        self._counts: dict[str, int] = {}

    def check(self, identity: str) -> None:
        count = self._counts.get(identity, 0) + 1
        if count > self._limit:
            raise RateLimitError(
                f"{identity!r} exceeded {self._limit} submissions in the current window"
            )
        self._counts[identity] = count

    def observed(self, identity: str) -> int:
        return self._counts.get(identity, 0)

    def reset(self, identity: str | None = None) -> None:
        """Advance the window — clear one identity's count, or all when ``identity`` is ``None``."""
        if identity is None:
            self._counts.clear()
        else:
            self._counts.pop(identity, None)
