"""Runs -- MLflow tracking and lifecycle eventing.

Every job is an MLflow run keyed by its reproducibility pin (``cloud.md`` §5, §6):
:mod:`.tracking` logs the :class:`~astro_mine.cloud.artifacts.runcontext.RunContext` envelope
(input hashes, image digest, Core interface version, seed, lockfile) and content-addressed
artifact pointers, and stamps the MLflow ``run_id`` back onto the context, so a scaled run is
as reproducible as a laptop run and a re-run reproduces from the recorded envelope.
:mod:`.events` emits **NATS/JetStream** completion events that Bench / Studio / Hub consume
(``cloud.md`` §4, §6). Both heavy backends (MLflow, NATS) sit behind injectable clients so the
local tier needs neither.

Backlog: RM-P1-CLOUD-05 -- https://github.com/astro-mine/astro-mine-cloud/issues/16
"""

from __future__ import annotations

from astro_mine.cloud.runs.eventlog import (
    DurableSubscriber,
    EventLog,
    InMemoryEventLog,
)
from astro_mine.cloud.runs.events import (
    SUBJECT,
    CollectingPublisher,
    CompletionEvent,
    EventPublisher,
    NullPublisher,
    RunObserver,
    RunStatus,
    emit_completion,
)
from astro_mine.cloud.runs.nats import STREAM, JetStreamConsumer, NatsEventPublisher
from astro_mine.cloud.runs.status import (
    InMemoryJobStatusStore,
    JobStatus,
    JobStatusStore,
    RedisJobStatusStore,
)
from astro_mine.cloud.runs.tracking import RunTracker, TrackingClient

__all__ = [
    "STREAM",
    "SUBJECT",
    "CollectingPublisher",
    "CompletionEvent",
    "DurableSubscriber",
    "EventLog",
    "EventPublisher",
    "InMemoryEventLog",
    "InMemoryJobStatusStore",
    "JetStreamConsumer",
    "JobStatus",
    "JobStatusStore",
    "NatsEventPublisher",
    "NullPublisher",
    "RedisJobStatusStore",
    "RunObserver",
    "RunStatus",
    "RunTracker",
    "TrackingClient",
    "emit_completion",
]
