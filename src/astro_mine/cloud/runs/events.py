"""Lifecycle events -- run completion on NATS/JetStream for Bench/Studio/Hub.

Cloud emits a :class:`CompletionEvent` as a run moves through submitted -> started ->
completed/failed, on **NATS/JetStream**, so [Bench](bench.md)/[Studio]/[Hub] can ingest
results without polling (``cloud.md`` §4, §6). The transport is the injectable
:class:`EventPublisher` seam: :class:`NullPublisher` (default, no-op) and
:class:`CollectingPublisher` (tests) keep the local tier transport-free; a real
NATS-JetStream publisher (``[nats]`` extra) is the production sink.

Backlog: RM-P1-CLOUD-05 -- astro-mine-cloud#16
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from astro_mine.cloud.artifacts.runcontext import RunContext
    from astro_mine.cloud.runs.status import JobStatusStore

__all__ = [
    "SUBJECT",
    "CollectingPublisher",
    "CompletionEvent",
    "EventPublisher",
    "NullPublisher",
    "RunObserver",
    "RunStatus",
    "emit_completion",
]

#: The NATS subject run lifecycle events are published on.
SUBJECT = "astro-mine.cloud.runs"

RunStatus = Literal["submitted", "started", "completed", "failed"]


class CompletionEvent(BaseModel):
    """A run lifecycle event: its status, tenant, reproducibility pin, and outputs."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    run_address: str
    status: RunStatus
    tenant: str | None = None
    run_id: str | None = None
    outputs: dict[str, str] = Field(default_factory=dict)


@runtime_checkable
class EventPublisher(Protocol):
    """Publishes a :class:`CompletionEvent` to a subject."""

    def publish(self, subject: str, event: CompletionEvent) -> None: ...


class NullPublisher:
    """The default publisher: drops events, so the local tier needs no message bus."""

    def publish(self, subject: str, event: CompletionEvent) -> None:
        return None


class CollectingPublisher:
    """Records published events in memory -- for tests and local inspection."""

    def __init__(self) -> None:
        self.events: list[tuple[str, CompletionEvent]] = []

    def publish(self, subject: str, event: CompletionEvent) -> None:
        self.events.append((subject, event))


def emit_completion(
    publisher: EventPublisher,
    context: RunContext,
    status: RunStatus,
    *,
    tenant: str | None = None,
) -> CompletionEvent:
    """Build and publish a :class:`CompletionEvent` for *context*; return the event.

    ``run_address`` is the context's ``run_pin`` -- the lifecycle-stable identity (outputs
    excluded) -- so every event of one run (``submitted`` → ``completed``) shares an address a
    consumer correlates on, while the terminal event still carries the produced ``outputs``.
    """
    event = CompletionEvent(
        run_address=context.run_pin(),
        status=status,
        tenant=tenant,
        run_id=context.run_id,
        outputs=dict(context.outputs),
    )
    publisher.publish(SUBJECT, event)
    return event


@dataclass(frozen=True)
class RunObserver:
    """Bundles the eventing side effects threaded through the run path (RM-P1-CLOUD-06).

    One value carries the injected :class:`EventPublisher` (default :class:`NullPublisher`, so the
    local tier stays broker-free), the optional ephemeral :class:`JobStatusStore`, and the run's
    ``tenant``. :meth:`transition` publishes a :class:`CompletionEvent` **and** updates the status
    store, so a caller emits a lifecycle transition with one call. The default instance is inert --
    passing it changes nothing, keeping every existing call site working.
    """

    publisher: EventPublisher = field(default_factory=NullPublisher)
    status_store: JobStatusStore | None = None
    tenant: str | None = None

    def transition(self, context: RunContext, status: RunStatus) -> CompletionEvent:
        """Publish the ``status`` event for ``context``; record it in the status store, if any."""
        event = emit_completion(self.publisher, context, status, tenant=self.tenant)
        if self.status_store is not None:
            # Imported lazily: status.py imports RunStatus from this module, so a top-level import
            # here would be a cycle.
            from astro_mine.cloud.runs.status import JobStatus

            self.status_store.set_status(
                JobStatus(
                    run_address=event.run_address,
                    status=status,
                    tenant=self.tenant,
                    run_id=event.run_id,
                )
            )
        return event
