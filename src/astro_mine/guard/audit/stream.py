"""The MCAP SafetyVerdict stream — the auditable, replayable output surface (RM-P1-GUARD-06).

Streams per-tick :class:`~astro_mine.guard.audit.model.SafetyVerdict` records to an **MCAP**
container (timestamped, jsonschema-tagged) on the topic :data:`VERDICTS_TOPIC`, written alongside
Sim/Ops telemetry so a shielded run's safety behaviour is replayable channel-by-channel and Bench
scores it through the same MCAP artifact boundary it decodes Sim across (guard.md §5, §6;
conventions.md §4/§5). Mirrors ``astro_mine.sim.recording``.

**Best-effort, never on the safety path (safety-critical MUST).** A ``PolicyShield`` calls
:meth:`VerdictStream.write_verdict` *after* the certified action is fixed, so this module swallows
every write fault (encoding, IO, back-pressure) — a dropped log never changes the certified action
(guard.md §8, §9.1). ``mcap`` is an **optional extra** (``astro-mine-guard[recording]``), imported
lazily, so the base package stays core + pydantic and the safety path carries zero
telemetry dependencies — importing this module never imports ``mcap``.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterator
from contextlib import contextmanager
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from astro_mine.guard.audit.model import SafetyVerdict, load_schema

if TYPE_CHECKING:
    from mcap.writer import Writer

__all__ = [
    "VERDICTS_TOPIC",
    "VERDICT_SCHEMA_NAME",
    "VerdictStream",
    "open_verdict_stream",
    "read_verdicts",
]

_LOG = logging.getLogger(__name__)

#: The MCAP topic the per-tick verdicts are written to (the View overlay / Bench score feed).
VERDICTS_TOPIC = "/guard/verdicts"
#: The registered MCAP schema name for the verdict channel.
VERDICT_SCHEMA_NAME = "astro_mine.guard.audit.SafetyVerdict"
_NS_PER_S = 1_000_000_000


def _guard_version() -> str:
    """The installed ``astro-mine-guard`` version (the MCAP library stamp)."""
    try:
        return metadata.version("astro-mine-platform")
    except metadata.PackageNotFoundError:  # pragma: no cover - source tree without metadata
        return "0.0.0"


def _log_time_ns(sim_time_s: float) -> int:
    """MCAP log time in nanoseconds from elapsed sim seconds (0 for a non-finite/negative time)."""
    if math.isfinite(sim_time_s) and sim_time_s >= 0.0:
        return int(sim_time_s * _NS_PER_S)
    return 0


class VerdictStream:
    """A best-effort :class:`~astro_mine.guard.audit.sink.VerdictSink` backed by an open MCAP.

    Obtained from :func:`open_verdict_stream`. :meth:`write_verdict` is the per-tick sink the shield
    drives after certifying an action; it **never raises** — any encoding/IO/back-pressure fault is
    logged and swallowed (telemetry is best-effort, the shield is not)."""

    def __init__(self, writer: Writer) -> None:
        self._writer = writer
        self._schema_id = writer.register_schema(
            name=VERDICT_SCHEMA_NAME,
            encoding="jsonschema",
            data=json.dumps(load_schema(), sort_keys=True).encode(),
        )
        self._channel_id = writer.register_channel(
            topic=VERDICTS_TOPIC, message_encoding="json", schema_id=self._schema_id
        )
        self._sequence = 0

    def write_verdict(self, verdict: SafetyVerdict) -> None:
        """Write one verdict as a timestamped JSON message. Best-effort — never raises."""
        try:
            payload: dict[str, Any] = verdict.model_dump(mode="json")
            data = json.dumps(payload, sort_keys=True).encode()
            log_time = _log_time_ns(verdict.sim_time_s)
            self._writer.add_message(
                channel_id=self._channel_id,
                log_time=log_time,
                data=data,
                publish_time=log_time,
                sequence=self._sequence,
            )
            self._sequence += 1
        except Exception:
            _LOG.warning(
                "dropped a SafetyVerdict on the audit stream (best-effort telemetry); "
                "the certified action is unaffected",
                exc_info=True,
            )


@contextmanager
def open_verdict_stream(path: str | Path) -> Iterator[VerdictStream]:
    """Open an MCAP verdict stream at ``path`` for streaming per-tick verdicts.

    Yields a :class:`VerdictStream` whose :meth:`~VerdictStream.write_verdict` is the shield's
    best-effort audit sink. The MCAP header and footer are written on entry and exit, so the file
    is well-formed even if nothing is recorded. Requires the ``[recording]`` extra (``mcap``),
    imported here lazily."""
    from mcap.writer import Writer  # lazily: the [recording] extra supplies mcap

    with Path(path).open("wb") as handle:
        writer = Writer(handle)
        writer.start(profile="astro-mine-guard", library=f"astro-mine-guard/{_guard_version()}")
        try:
            yield VerdictStream(writer)
        finally:
            writer.finish()


def read_verdicts(path: str | Path) -> list[SafetyVerdict]:
    """Read a verdict MCAP at ``path`` back into typed :class:`SafetyVerdict` records (log order).

    The inverse of the stream: recovers every ``/guard/verdicts`` message so Bench / View / a
    determinism gate replay a shielded run's safety behaviour from the file alone. Requires the
    ``[recording]`` extra (``mcap``), imported here lazily."""
    from mcap.reader import make_reader  # lazily: the [recording] extra supplies mcap

    out: list[SafetyVerdict] = []
    with Path(path).open("rb") as handle:
        reader = make_reader(handle)
        for _schema, channel, message in reader.iter_messages():
            if channel.topic == VERDICTS_TOPIC:
                out.append(SafetyVerdict.model_validate(json.loads(message.data)))
    return out
