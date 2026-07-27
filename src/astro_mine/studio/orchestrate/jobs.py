"""Durable, cancelable, resumable design-loop jobs (studio.md §2 principle 5, §3
``orchestrate/jobs``; STUDIO-03 acceptance).

A trade study is minutes-to-hours of distributed work, so design exploration is durable
background work, never a blocking request. This module ships the **local tier** of that
model (conventions.md §7 — the local tier MUST work): an in-process, asyncio fan-out over
a pluggable :class:`JobStore`. The three guarantees are realized as:

- **durable** — every job's state lives in the ``JobStore``; reusing the same store across
  runs preserves progress. The production Redis(status)/Postgres(metadata) backends
  implement the same Protocol.
- **cancelable** — ``request_cancel`` flags a job; a job canceled before it starts running
  is skipped cooperatively.
- **resumable** — a job already ``SUCCEEDED`` in the store, or whose content-addressed
  result is already in the cache, is not re-evaluated ("resumes from content-addressed
  checkpoints without re-running completed candidates").

The **transport is a seam**: :class:`JobDispatcher` runs the loop in-process here
(:class:`LocalDispatcher`), and :class:`~.cloud.CloudDispatcher` runs the same loop over
``cloud.submit()`` — same Protocol, same three guarantees, which is why they live above the
dispatcher rather than inside it. The NATS+JetStream :class:`EventSink` (RM-P1-CLOUD-06)
implements its Protocol the same way. The :class:`EventSink`/:class:`JobEvent` shape mirrors
Cloud's ``EventPublisher``/``CompletionEvent``, and
:class:`~.cloud.JobEventPublisher` is the adapter between them.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol, runtime_checkable

from astro_mine.core.objective import ObjectiveDocument

from .._base import FrozenStudioModel, StudioModel
from ..models import DesignCandidate, EvaluatedCandidate
from .cache import ResultCache, cache_key
from .clients import SiblingClients
from .loop import evaluate_candidate


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELED = "canceled"


class JobEvent(FrozenStudioModel):
    """A job-lifecycle event (mirrors Cloud's ``CompletionEvent``)."""

    job_id: str
    status: JobStatus


class JobRecord(StudioModel):
    """The durable state of one candidate/seed evaluation."""

    job_id: str
    candidate_id: str
    seed: int
    cache_key: str
    status: JobStatus = JobStatus.PENDING
    result_digest: str | None = None
    error: str | None = None
    cancel_requested: bool = False


def job_id_for(candidate: DesignCandidate, seed: int) -> str:
    """The stable job id for a candidate/seed (stable across runs -> resumable)."""
    return f"{candidate.id}@{seed}"


@runtime_checkable
class JobStore(Protocol):
    """The durable job-state seam (in-memory here; Redis/Postgres in deployment)."""

    def create(self, record: JobRecord) -> JobRecord: ...

    def get(self, job_id: str) -> JobRecord: ...

    def has(self, job_id: str) -> bool: ...

    def update(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        result_digest: str | None = None,
        error: str | None = None,
    ) -> None: ...

    def request_cancel(self, job_id: str) -> None: ...

    def records(self) -> Sequence[JobRecord]: ...


class InMemoryJobStore:
    """Tier-1 in-process :class:`JobStore`."""

    def __init__(self) -> None:
        self._records: dict[str, JobRecord] = {}

    def create(self, record: JobRecord) -> JobRecord:
        # An existing record wins, so a re-run resumes rather than resetting progress.
        return self._records.setdefault(record.job_id, record)

    def get(self, job_id: str) -> JobRecord:
        return self._records[job_id]

    def has(self, job_id: str) -> bool:
        return job_id in self._records

    def update(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        result_digest: str | None = None,
        error: str | None = None,
    ) -> None:
        record = self._records[job_id]
        if status is not None:
            record.status = status
        if result_digest is not None:
            record.result_digest = result_digest
        if error is not None:
            record.error = error

    def request_cancel(self, job_id: str) -> None:
        self._records[job_id].cancel_requested = True

    def records(self) -> Sequence[JobRecord]:
        return tuple(self._records.values())


@runtime_checkable
class EventSink(Protocol):
    def emit(self, event: JobEvent) -> None: ...


class NullEventSink:
    def emit(self, event: JobEvent) -> None:
        return None


class CollectingEventSink:
    """Captures emitted events (tests, and a stand-in for the durable NATS stream)."""

    def __init__(self) -> None:
        self.events: list[JobEvent] = []

    def emit(self, event: JobEvent) -> None:
        self.events.append(event)


@runtime_checkable
class JobDispatcher(Protocol):
    async def evaluate(
        self,
        candidate: DesignCandidate,
        objective: ObjectiveDocument,
        *,
        seed: int,
        max_steps: int,
    ) -> EvaluatedCandidate: ...


class LocalDispatcher:
    """Runs the design loop in-process (off the event loop via a worker thread).
    :class:`~.cloud.CloudDispatcher` implements the same Protocol via ``cloud.submit()``."""

    def __init__(self, clients: SiblingClients, *, cache: ResultCache | None = None) -> None:
        self._clients = clients
        self._cache = cache

    async def evaluate(
        self,
        candidate: DesignCandidate,
        objective: ObjectiveDocument,
        *,
        seed: int,
        max_steps: int,
    ) -> EvaluatedCandidate:
        return await asyncio.to_thread(
            evaluate_candidate,
            candidate,
            objective,
            clients=self._clients,
            seed=seed,
            max_steps=max_steps,
            cache=self._cache,
        )


async def run_batch(
    candidates: Sequence[DesignCandidate],
    objective: ObjectiveDocument,
    *,
    dispatcher: JobDispatcher,
    seeds: Sequence[int] = (0,),
    store: JobStore | None = None,
    cache: ResultCache | None = None,
    events: EventSink | None = None,
    max_steps: int = 8,
    concurrency: int = 4,
) -> list[JobRecord]:
    """Fan a batch of candidates x seeds out as durable/cancelable/resumable jobs.

    Returns the final ``JobRecord`` for each candidate/seed in submission order."""
    store = store if store is not None else InMemoryJobStore()
    sink = events if events is not None else NullEventSink()
    semaphore = asyncio.Semaphore(concurrency)

    plan: list[tuple[str, DesignCandidate, int]] = []
    for candidate in candidates:
        for seed in seeds:
            job_id = job_id_for(candidate, seed)
            store.create(
                JobRecord(
                    job_id=job_id,
                    candidate_id=candidate.id,
                    seed=seed,
                    cache_key=cache_key(candidate, objective, seed),
                )
            )
            plan.append((job_id, candidate, seed))

    async def run_one(job_id: str, candidate: DesignCandidate, seed: int) -> None:
        record = store.get(job_id)
        # Resume (cheap, no slot consumed): already done in a prior run, or cached.
        if record.status is JobStatus.SUCCEEDED:
            sink.emit(JobEvent(job_id=job_id, status=JobStatus.SUCCEEDED))
            return
        if cache is not None and cache.has(record.cache_key):
            cached = cache.get(record.cache_key)
            store.update(job_id, status=JobStatus.SUCCEEDED, result_digest=cached.digest())
            sink.emit(JobEvent(job_id=job_id, status=JobStatus.SUCCEEDED))
            return

        async with semaphore:
            # Single cancel checkpoint — covers a job canceled before the batch and one
            # canceled while it waited for a slot.
            if store.get(job_id).cancel_requested:
                store.update(job_id, status=JobStatus.CANCELED)
                sink.emit(JobEvent(job_id=job_id, status=JobStatus.CANCELED))
                return
            store.update(job_id, status=JobStatus.RUNNING)
            sink.emit(JobEvent(job_id=job_id, status=JobStatus.RUNNING))
            try:
                evaluated = await dispatcher.evaluate(
                    candidate, objective, seed=seed, max_steps=max_steps
                )
            except Exception as exc:  # any sibling failure → a recorded FAILED job
                store.update(job_id, status=JobStatus.FAILED, error=str(exc))
                sink.emit(JobEvent(job_id=job_id, status=JobStatus.FAILED))
                return
            if cache is not None:
                cache.put(record.cache_key, evaluated)
            store.update(job_id, status=JobStatus.SUCCEEDED, result_digest=evaluated.digest())
            sink.emit(JobEvent(job_id=job_id, status=JobStatus.SUCCEEDED))

    await asyncio.gather(*(run_one(job_id, candidate, seed) for job_id, candidate, seed in plan))
    return [store.get(job_id) for job_id, _, _ in plan]
