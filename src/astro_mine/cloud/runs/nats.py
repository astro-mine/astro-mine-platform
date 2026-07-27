"""The real NATS + JetStream eventing backend (RM-P1-CLOUD-06; conventions.md §4; cloud.md §4, §6).

The production sink behind the :class:`~astro_mine.cloud.runs.events.EventPublisher` seam:
:class:`NatsEventPublisher` publishes :class:`~astro_mine.cloud.runs.events.CompletionEvent`s to a
durable **JetStream** stream, and :class:`JetStreamConsumer` is a durable **pull** consumer with the
semantics the :class:`~astro_mine.cloud.runs.eventlog.EventLog` contract states -- at-least-once,
explicit ack, replay from a cursor, resume across a restart (a durable consumer's ack floor lives in
the server). This is the cluster tier; the local tier stays broker-free on the ``NullPublisher``
default (``cloud.md`` §2 principle 2).

``nats-py`` is async; the seam is synchronous, so the publisher drives a private event loop and
holds one persistent connection. Both classes need the ``[nats]`` extra + a running server, so they
lazy-import ``nats`` and are covered only by the opt-in ``nats``-marked integration test -- exactly
how ``KubectlClusterClient`` is gated on a live cluster. Importing this module needs no broker.
"""

from __future__ import annotations

import asyncio
import contextlib
from typing import TYPE_CHECKING, Any, cast

from astro_mine.cloud.runs.events import SUBJECT, CompletionEvent

if TYPE_CHECKING:
    from collections.abc import Coroutine

__all__ = ["STREAM", "JetStreamConsumer", "NatsEventPublisher"]

#: The JetStream stream backing the run-lifecycle subject (durable + replayable).
STREAM = "astro-mine-cloud-runs"

#: Default NATS endpoint (the docker-compose service / a laptop broker).
DEFAULT_SERVERS = "nats://localhost:4222"


def _require_nats() -> Any:  # pragma: no cover - requires the [nats] extra
    try:
        import nats
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "the NATS eventing backend needs the 'nats' extra: pip install 'astro-mine-cloud[nats]'"
        ) from exc
    return nats


class NatsEventPublisher:  # pragma: no cover - requires the [nats] extra + a running server
    """A JetStream-backed :class:`~astro_mine.cloud.runs.events.EventPublisher`.

    Publishes each event's canonical JSON to ``SUBJECT`` on the durable :data:`STREAM`, so Bench /
    Studio / Hub ingest results without polling. Synchronous ``publish`` over an async client: a
    private event loop is created lazily, the connection + JetStream context persist on the
    instance, and the stream is created on first use (idempotent). :meth:`close` releases it.
    """

    def __init__(self, servers: str = DEFAULT_SERVERS, *, subject: str = SUBJECT) -> None:
        self._servers = servers
        self._subject = subject
        self._loop = asyncio.new_event_loop()
        self._nc: Any = None
        self._js: Any = None

    def _run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        return self._loop.run_until_complete(coro)

    async def _ensure(self) -> Any:
        if self._js is None:
            nats = _require_nats()
            self._nc = await nats.connect(self._servers)
            self._js = self._nc.jetstream()
            # Idempotent create: reuse the stream if it already exists.
            with contextlib.suppress(Exception):
                await self._js.add_stream(name=STREAM, subjects=[self._subject])
        return self._js

    async def _publish(self, subject: str, event: CompletionEvent) -> None:
        js = await self._ensure()
        await js.publish(subject, event.model_dump_json().encode())

    def publish(self, subject: str, event: CompletionEvent) -> None:
        """Publish ``event`` to ``subject`` on JetStream (the :class:`EventPublisher` contract)."""
        self._run(self._publish(subject, event))

    def close(self) -> None:
        """Drain and close the underlying connection and event loop."""
        if self._nc is not None:
            self._run(self._nc.drain())
            self._nc = None
            self._js = None
        self._loop.close()


class JetStreamConsumer:  # pragma: no cover - requires the [nats] extra + a running server
    """A durable JetStream **pull** consumer over the run-lifecycle stream.

    ``fetch`` pulls up to ``batch`` messages for the named durable consumer, parses each into a
    :class:`CompletionEvent`, and acks it (a handler that raises before this leaves it unacked, so
    it is redelivered -- at-least-once). Because the durable's ack floor is server-side, a consumer
    created afresh with the same ``durable`` name resumes exactly where it stopped -- replay from a
    cursor and resume-across-restart, the AC's durable-consumer guarantees.
    """

    def __init__(
        self, durable: str, servers: str = DEFAULT_SERVERS, *, subject: str = SUBJECT
    ) -> None:
        self._durable = durable
        self._servers = servers
        self._subject = subject
        self._loop = asyncio.new_event_loop()
        self._nc: Any = None
        self._sub: Any = None

    def _run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        return self._loop.run_until_complete(coro)

    async def _ensure(self) -> Any:
        if self._sub is None:
            nats = _require_nats()
            self._nc = await nats.connect(self._servers)
            js = self._nc.jetstream()
            self._sub = await js.pull_subscribe(self._subject, durable=self._durable, stream=STREAM)
        return self._sub

    async def _fetch(self, batch: int, timeout: float) -> list[CompletionEvent]:
        sub = await self._ensure()
        messages = await sub.fetch(batch, timeout=timeout)
        events: list[CompletionEvent] = []
        for message in messages:
            events.append(CompletionEvent.model_validate_json(message.data))
            await message.ack()
        return events

    def fetch(self, *, batch: int = 64, timeout: float = 1.0) -> list[CompletionEvent]:
        """Pull up to ``batch`` events (acking each); empty if none arrive within ``timeout``."""
        return cast("list[CompletionEvent]", self._run(self._fetch(batch, timeout)))

    def close(self) -> None:
        """Close the underlying connection and event loop."""
        if self._nc is not None:
            self._run(self._nc.drain())
            self._nc = None
            self._sub = None
        self._loop.close()
