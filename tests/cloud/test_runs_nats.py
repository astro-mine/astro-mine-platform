"""Opt-in integration test against a real NATS + JetStream server (RM-P1-CLOUD-06).

Exercises :class:`NatsEventPublisher` -> a durable JetStream stream -> :class:`JetStreamConsumer`
end to end, including durable-consumer resume across a fresh consumer. Requires the ``[nats]`` extra
and a running server; self-skips otherwise, so CI (which runs no broker) never depends on it -- the
same opt-in shape as the MinIO S3 integration test. Run it with:

    docker compose up -d nats
    NATS_URL=nats://localhost:4222 uv run pytest -m nats
"""

from __future__ import annotations

import os
import uuid

import pytest

pytest.importorskip("nats")

from astro_mine.cloud.runs.events import CompletionEvent
from astro_mine.cloud.runs.nats import JetStreamConsumer, NatsEventPublisher

pytestmark = pytest.mark.nats

_NATS_URL = os.environ.get("NATS_URL")
_skip = pytest.mark.skipif(
    not _NATS_URL, reason="set NATS_URL (and `docker compose up -d nats`) to exercise real NATS"
)


@_skip
def test_publish_then_durable_consume_roundtrips() -> None:
    servers = _NATS_URL or ""
    durable = f"bench-{uuid.uuid4().hex[:8]}"
    publisher = NatsEventPublisher(servers)
    event = CompletionEvent(run_address="sha256:" + "a" * 64, status="completed", tenant="acme")
    try:
        publisher.publish("astro-mine.cloud.runs", event)
    finally:
        publisher.close()

    consumer = JetStreamConsumer(durable, servers)
    try:
        received = consumer.fetch(batch=10, timeout=2.0)
    finally:
        consumer.close()
    assert any(e.run_address == event.run_address and e.status == "completed" for e in received)


@_skip
def test_durable_consumer_resumes_after_restart() -> None:
    servers = _NATS_URL or ""
    durable = f"studio-{uuid.uuid4().hex[:8]}"
    publisher = NatsEventPublisher(servers)
    first = CompletionEvent(run_address="sha256:" + "b" * 64, status="submitted")
    second = CompletionEvent(run_address="sha256:" + "c" * 64, status="completed")
    try:
        publisher.publish("astro-mine.cloud.runs", first)
        consumer = JetStreamConsumer(durable, servers)
        try:
            assert len(consumer.fetch(batch=10, timeout=2.0)) >= 1  # consumes + acks `first`
        finally:
            consumer.close()

        publisher.publish("astro-mine.cloud.runs", second)
        # A fresh consumer with the same durable name resumes past the acked `first`.
        resumed = JetStreamConsumer(durable, servers)
        try:
            addresses = {e.run_address for e in resumed.fetch(batch=10, timeout=2.0)}
        finally:
            resumed.close()
        assert second.run_address in addresses
        assert first.run_address not in addresses
    finally:
        publisher.close()
