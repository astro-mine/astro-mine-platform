"""The durable event-log contract subscribers consume (RM-P1-CLOUD-06; cloud.md §4, §5).

Lifecycle events flow on a **durable, replayable** JetStream stream: a subscriber (Bench result
ingestion, Studio job tracking, Hub events) reads with **at-least-once** delivery, **explicit ack**,
and can **replay from a cursor** and **resume across a restart**. Those semantics are the
contract -- independent of NATS -- so this module states them as an :class:`EventLog` Protocol + an
:class:`InMemoryEventLog` double that models a JetStream **durable pull consumer**:

- the stream is an append-only log of :class:`CompletionEvent`s, each with a 1-based sequence;
- a *durable consumer* is named; the log persists its **ack floor**, so ``fetch`` returns only
  messages above the floor (unacked), ``ack`` advances the floor, and an unacked message is
  redelivered on the next ``fetch`` (at-least-once);
- a fresh durable name replays from sequence 0; reusing the name after a "restart" resumes from the
  persisted floor.

The real NATS/JetStream realization lives in :mod:`astro_mine.cloud.runs.nats` behind the ``[nats]``
extra; this double lets the **consumer-driven contract test** run in CI with no broker (as ``moto``
covers the S3 path). :class:`DurableSubscriber` is the stub subscriber that test drives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Callable

    from astro_mine.cloud.runs.events import CompletionEvent

__all__ = [
    "DurableSubscriber",
    "EventLog",
    "InMemoryEventLog",
]


@runtime_checkable
class EventLog(Protocol):
    """A durable, replayable event log with named durable consumers (JetStream semantics)."""

    def append(self, event: CompletionEvent) -> int:
        """Append ``event`` to the stream; return its 1-based sequence number."""
        ...

    def fetch(self, durable: str, *, batch: int = 64) -> list[tuple[int, CompletionEvent]]:
        """Return up to ``batch`` unacked ``(seq, event)`` for ``durable``, above its ack floor."""
        ...

    def ack(self, durable: str, seq: int) -> None:
        """Advance ``durable``'s ack floor to ``seq`` (idempotent; never rewinds)."""
        ...


class InMemoryEventLog:
    """An in-memory :class:`EventLog` double modeling a JetStream durable pull consumer.

    The append-only ``_events`` list is the persisted stream; ``_ack_floor`` maps a durable
    consumer name to the highest sequence it has acked. Because the floor lives in the log (not the
    subscriber), constructing a new :class:`DurableSubscriber` with the same durable name after a
    crash resumes exactly where it left off -- the "survives a restart" guarantee -- and any message
    fetched but not acked is redelivered (at-least-once).
    """

    def __init__(self) -> None:
        self._events: list[CompletionEvent] = []
        self._ack_floor: dict[str, int] = {}

    def append(self, event: CompletionEvent) -> int:
        self._events.append(event)
        return len(self._events)

    def fetch(self, durable: str, *, batch: int = 64) -> list[tuple[int, CompletionEvent]]:
        floor = self._ack_floor.get(durable, 0)
        pending = [(seq, event) for seq, event in enumerate(self._events, start=1) if seq > floor]
        return pending[:batch]

    def ack(self, durable: str, seq: int) -> None:
        self._ack_floor[durable] = max(self._ack_floor.get(durable, 0), seq)


class DurableSubscriber:
    """A stub durable subscriber -- fetch a batch, hand each event to a handler, then ack.

    Models a real consumer's loop: only after the handler processes an event is it acked, so a
    handler that raises leaves the message unacked for redelivery. ``drain`` processes every pending
    event; a subscriber created afresh with the same ``durable`` name resumes from the log's floor.
    """

    def __init__(
        self, log: EventLog, durable: str, handler: Callable[[CompletionEvent], None]
    ) -> None:
        self._log = log
        self._durable = durable
        self._handler = handler

    def drain(self) -> int:
        """Process and ack every currently-pending event; return how many were handled."""
        handled = 0
        for seq, event in self._log.fetch(self._durable):
            self._handler(event)  # a raise here leaves seq unacked -> redelivered next drain
            self._log.ack(self._durable, seq)
            handled += 1
        return handled
