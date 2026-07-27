"""STUDIO-03 — durable / cancelable / resumable async job model."""

from __future__ import annotations

import asyncio

from astro_mine.core.objective import ObjectiveDocument
from astro_mine.studio.models import AssetSelection, DesignCandidate
from astro_mine.studio.orchestrate import (
    CollectingEventSink,
    InMemoryJobStore,
    InMemoryResultCache,
    JobDispatcher,
    JobRecord,
    JobStatus,
    JobStore,
    LocalDispatcher,
    NullEventSink,
    SiblingClients,
    cache_key,
    job_id_for,
    run_batch,
)
from astro_mine.studio.orchestrate.jobs import EvaluatedCandidate


class _CountingDispatcher:
    """Wraps a real dispatcher and counts actual evaluations (to prove resume)."""

    def __init__(self, inner: JobDispatcher) -> None:
        self._inner = inner
        self.calls = 0

    async def evaluate(
        self, candidate: DesignCandidate, objective: ObjectiveDocument, *, seed: int, max_steps: int
    ) -> EvaluatedCandidate:
        self.calls += 1
        return await self._inner.evaluate(candidate, objective, seed=seed, max_steps=max_steps)


def _candidates(n: int) -> list[DesignCandidate]:
    return [
        DesignCandidate(id=f"c{i}", swarm=[AssetSelection(sadf_ref="rover", count=i + 1)])
        for i in range(n)
    ]


# ---- store + sinks -------------------------------------------------------- #


def test_in_memory_job_store() -> None:
    store = InMemoryJobStore()
    assert isinstance(store, JobStore)
    rec = JobRecord(job_id="j", candidate_id="c", seed=0, cache_key="sha256:k")
    assert store.create(rec) is rec
    # create is idempotent — an existing record wins (resume, not reset)
    assert (
        store.create(JobRecord(job_id="j", candidate_id="c", seed=0, cache_key="sha256:k")) is rec
    )
    assert store.has("j") and store.get("j").status is JobStatus.PENDING
    store.update("j", status=JobStatus.SUCCEEDED, result_digest="sha256:r", error="none")
    assert store.get("j").status is JobStatus.SUCCEEDED
    assert store.get("j").result_digest == "sha256:r" and store.get("j").error == "none"
    store.request_cancel("j")
    assert store.get("j").cancel_requested is True
    assert len(store.records()) == 1


def test_event_sinks() -> None:
    from astro_mine.studio.orchestrate import JobEvent

    event = JobEvent(job_id="j", status=JobStatus.RUNNING)
    assert NullEventSink().emit(event) is None
    sink = CollectingEventSink()
    sink.emit(event)
    assert sink.events == [event]


# ---- run_batch ------------------------------------------------------------ #


def test_run_batch_all_succeed_with_events(
    objective_doc: ObjectiveDocument, clients: SiblingClients
) -> None:
    candidates = _candidates(3)
    events = CollectingEventSink()
    records = asyncio.run(
        run_batch(
            candidates,
            objective_doc,
            dispatcher=LocalDispatcher(clients),
            seeds=(1, 2),
            events=events,
            concurrency=2,
        )
    )
    assert len(records) == 6
    assert all(r.status is JobStatus.SUCCEEDED for r in records)
    assert all(r.result_digest is not None for r in records)
    emitted = {(e.job_id, e.status) for e in events.events}
    assert (job_id_for(candidates[0], 1), JobStatus.RUNNING) in emitted
    assert (job_id_for(candidates[0], 1), JobStatus.SUCCEEDED) in emitted


def test_run_batch_is_deterministic(
    objective_doc: ObjectiveDocument, clients: SiblingClients
) -> None:
    candidates = _candidates(3)

    def run() -> list[str | None]:
        records = asyncio.run(
            run_batch(candidates, objective_doc, dispatcher=LocalDispatcher(clients), seeds=(1, 2))
        )
        return sorted(r.result_digest for r in records)

    assert run() == run()  # a re-run reproduces the identical result set


def test_run_batch_resumes_from_store_without_reevaluating(
    objective_doc: ObjectiveDocument, clients: SiblingClients
) -> None:
    candidates = _candidates(3)
    store = InMemoryJobStore()
    dispatcher = _CountingDispatcher(LocalDispatcher(clients))
    asyncio.run(
        run_batch(candidates, objective_doc, dispatcher=dispatcher, seeds=(1,), store=store)
    )
    first = dispatcher.calls
    asyncio.run(
        run_batch(candidates, objective_doc, dispatcher=dispatcher, seeds=(1,), store=store)
    )
    assert first == 3
    assert dispatcher.calls == first  # resumed: no candidate re-evaluated


def test_run_batch_resumes_from_cache(
    objective_doc: ObjectiveDocument, clients: SiblingClients
) -> None:
    candidates = _candidates(2)
    cache = InMemoryResultCache()
    dispatcher = _CountingDispatcher(LocalDispatcher(clients))
    asyncio.run(
        run_batch(candidates, objective_doc, dispatcher=dispatcher, seeds=(1,), cache=cache)
    )
    # fresh store, warm cache -> jobs resolve from the content-addressed checkpoint
    records = asyncio.run(
        run_batch(
            candidates,
            objective_doc,
            dispatcher=dispatcher,
            seeds=(1,),
            store=InMemoryJobStore(),
            cache=cache,
        )
    )
    assert dispatcher.calls == 2  # only the first run evaluated
    assert all(r.status is JobStatus.SUCCEEDED for r in records)


def test_run_batch_cancels_flagged_job(
    objective_doc: ObjectiveDocument, clients: SiblingClients
) -> None:
    candidates = _candidates(1)
    store = InMemoryJobStore()
    job_id = job_id_for(candidates[0], 9)
    store.create(
        JobRecord(
            job_id=job_id,
            candidate_id=candidates[0].id,
            seed=9,
            cache_key=cache_key(candidates[0], objective_doc, 9),
            cancel_requested=True,
        )
    )
    records = asyncio.run(
        run_batch(
            candidates, objective_doc, dispatcher=LocalDispatcher(clients), seeds=(9,), store=store
        )
    )
    assert records[0].status is JobStatus.CANCELED
    assert records[0].result_digest is None


def test_run_batch_records_guard_failure(
    objective_doc: ObjectiveDocument, clients: SiblingClients
) -> None:
    unsafe = DesignCandidate(
        id="unsafe",
        swarm=[AssetSelection(sadf_ref="rover", count=1)],
        decision_vector={"unsafe": 1.0},
    )
    records = asyncio.run(run_batch([unsafe], objective_doc, dispatcher=LocalDispatcher(clients)))
    assert records[0].status is JobStatus.FAILED
    assert records[0].error is not None and "certification" in records[0].error
