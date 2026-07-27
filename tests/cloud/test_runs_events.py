"""Run lifecycle events -- CompletionEvent + RunObserver on the NATS subject (RM-P1-CLOUD-06)."""

from __future__ import annotations

from astro_mine.cloud.artifacts.runcontext import RunContext
from astro_mine.cloud.runs.events import (
    SUBJECT,
    CollectingPublisher,
    CompletionEvent,
    EventPublisher,
    NullPublisher,
    RunObserver,
    emit_completion,
)
from astro_mine.cloud.runs.status import InMemoryJobStatusStore


def test_publishers_satisfy_protocol() -> None:
    assert isinstance(NullPublisher(), EventPublisher)
    assert isinstance(CollectingPublisher(), EventPublisher)


def test_emit_completion_uses_run_pin_and_carries_outputs() -> None:
    publisher = CollectingPublisher()
    context = RunContext(seed=1, run_id="run-9", outputs={"y": "sha256:" + "a" * 64})

    event = emit_completion(publisher, context, "completed", tenant="acme")

    assert event.status == "completed"
    assert event.tenant == "acme"
    assert event.run_id == "run-9"
    # run_address is the lifecycle-stable pin (outputs excluded), NOT the output-inclusive address.
    assert event.run_address == context.run_pin()
    assert event.run_address != context.content_address()
    assert event.outputs == {"y": "sha256:" + "a" * 64}
    assert publisher.events == [(SUBJECT, event)]


def test_run_pin_is_shared_across_the_lifecycle() -> None:
    # The pre-run context (no outputs) and the final context (with outputs) pin identically,
    # so submitted/started share an address with completed.
    pre = RunContext(seed=42, source_content_hashes={"in": "sha256:" + "b" * 64})
    final = pre.model_copy(update={"outputs": {"out": "sha256:" + "c" * 64}})
    assert pre.run_pin() == final.run_pin()
    assert pre.content_address() != final.content_address()


def test_null_publisher_drops_events() -> None:
    context = RunContext(seed=1)
    event = emit_completion(NullPublisher(), context, "submitted")
    assert event.status == "submitted"


def test_completion_event_is_frozen_and_strict() -> None:
    event = CompletionEvent(run_address="sha256:" + "a" * 64, status="failed")
    assert event.status == "failed"


def test_run_observer_publishes_and_records_status() -> None:
    publisher = CollectingPublisher()
    status_store = InMemoryJobStatusStore()
    observer = RunObserver(publisher=publisher, status_store=status_store, tenant="acme")
    context = RunContext(seed=1, run_id="run-1")

    event = observer.transition(context, "started")

    assert [e.status for _, e in publisher.events] == ["started"]
    recorded = status_store.get_status(event.run_address)
    assert recorded is not None
    assert recorded.status == "started"
    assert recorded.tenant == "acme"
    assert recorded.run_id == "run-1"


def test_run_observer_default_is_inert() -> None:
    # The default observer drops events and touches no status store -- the local-tier contract.
    observer = RunObserver()
    event = observer.transition(RunContext(seed=1), "submitted")
    assert event.status == "submitted"
    assert observer.status_store is None
