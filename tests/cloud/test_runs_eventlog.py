"""The durable event-log contract subscribers consume (RM-P1-CLOUD-06).

Exercises the JetStream durable-pull-consumer semantics against the in-memory double: replay from a
cursor, explicit ack, at-least-once redelivery, and resume-across-restart -- the consumer-driven
contract test the AC calls for, run in CI with no broker (a stub Bench/Studio subscriber).
"""

from __future__ import annotations

import pytest

from astro_mine.cloud.runs.eventlog import DurableSubscriber, EventLog, InMemoryEventLog
from astro_mine.cloud.runs.events import CompletionEvent


def _event(n: int, status: str = "completed") -> CompletionEvent:
    return CompletionEvent(run_address=f"sha256:{n:064d}", status=status)  # type: ignore[arg-type]


def test_log_satisfies_protocol() -> None:
    assert isinstance(InMemoryEventLog(), EventLog)


def test_append_assigns_increasing_sequences() -> None:
    log = InMemoryEventLog()
    assert log.append(_event(1)) == 1
    assert log.append(_event(2)) == 2


def test_fetch_replays_from_the_cursor_then_ack_advances() -> None:
    log = InMemoryEventLog()
    log.append(_event(1))
    log.append(_event(2))

    pending = log.fetch("bench")
    assert [seq for seq, _ in pending] == [1, 2]

    log.ack("bench", 1)
    assert [seq for seq, _ in log.fetch("bench")] == [2]  # only the unacked remainder


def test_at_least_once_redelivers_unacked() -> None:
    log = InMemoryEventLog()
    log.append(_event(1))
    # Fetch without acking -> the message is redelivered on the next fetch.
    assert [seq for seq, _ in log.fetch("bench")] == [1]
    assert [seq for seq, _ in log.fetch("bench")] == [1]


def test_durable_consumers_are_independent() -> None:
    log = InMemoryEventLog()
    log.append(_event(1))
    log.ack("bench", 1)
    # Studio's durable consumer has its own floor -> it still sees the event.
    assert [seq for seq, _ in log.fetch("studio")] == [1]


def test_subscriber_drains_and_acks() -> None:
    log = InMemoryEventLog()
    for n in range(1, 4):
        log.append(_event(n))
    seen: list[str] = []
    subscriber = DurableSubscriber(log, "bench", lambda e: seen.append(e.run_address))

    assert subscriber.drain() == 3
    assert len(seen) == 3
    assert subscriber.drain() == 0  # all acked; nothing pending


def test_subscriber_resumes_after_restart() -> None:
    log = InMemoryEventLog()
    log.append(_event(1))
    seen: list[str] = []
    DurableSubscriber(log, "bench", lambda e: seen.append(e.run_address)).drain()

    log.append(_event(2))
    # A fresh subscriber with the SAME durable name resumes from the persisted ack floor.
    resumed = DurableSubscriber(log, "bench", lambda e: seen.append(e.run_address))
    assert resumed.drain() == 1  # only the new event, not a replay of #1
    assert len(seen) == 2


def test_handler_failure_leaves_message_unacked() -> None:
    log = InMemoryEventLog()
    log.append(_event(1))

    def boom(_: CompletionEvent) -> None:
        raise RuntimeError("subscriber failed")

    with pytest.raises(RuntimeError, match="subscriber failed"):
        DurableSubscriber(log, "bench", boom).drain()

    # Unacked -> redelivered, so a healthy subscriber still processes it.
    seen: list[str] = []
    assert DurableSubscriber(log, "bench", lambda e: seen.append(e.run_address)).drain() == 1
    assert len(seen) == 1
