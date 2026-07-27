"""The verdict-sink seam — where a ``PolicyShield`` hands off its per-tick SafetyVerdict.

The shield (:mod:`astro_mine.guard.wrap`) writes one
:class:`~astro_mine.guard.audit.model.SafetyVerdict` per certified action to a :class:`VerdictSink`
**after** it has returned the certified action, so
verdict emission is **best-effort and off the safety path** (guard.md §5, §6; §8 "telemetry is
best-effort, the shield is not"). A sink MUST NOT raise into the shield — a dropped log never
changes the certified action.

Two dependency-light sinks live here (no ``mcap`` / ``pyarrow``):

- :class:`CollectingSink` — accumulates verdicts in memory (tests, in-process metric extraction);
- :class:`NullSink` — discards them (the explicit "shielding with no audit" default).

The MCAP stream sink is :class:`~astro_mine.guard.audit.stream.VerdictStream` (behind the
``[recording]`` extra), so opening a file-backed sink is the only place a telemetry dependency
is touched.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from astro_mine.guard.audit.model import SafetyVerdict

__all__ = ["CollectingSink", "NullSink", "VerdictSink"]


@runtime_checkable
class VerdictSink(Protocol):
    """A best-effort consumer of per-tick :class:`SafetyVerdict` records.

    Implementations MUST swallow their own faults (back-pressure, IO, encoding) and never raise
    into the caller — the shield calls :meth:`write_verdict` after the certified action is
    fixed, and a logging fault must never change it (guard.md §9.1)."""

    def write_verdict(self, verdict: SafetyVerdict) -> None:
        """Record one verdict. Never raises."""
        ...


class CollectingSink:
    """A :class:`VerdictSink` that appends verdicts to an in-memory list.

    The in-process audit surface for tests and for metric extraction without an MCAP file
    (:mod:`astro_mine.guard.audit.metrics` scores ``sink.verdicts`` directly)."""

    def __init__(self) -> None:
        self.verdicts: list[SafetyVerdict] = []

    def write_verdict(self, verdict: SafetyVerdict) -> None:
        """Append ``verdict`` to :attr:`verdicts`."""
        self.verdicts.append(verdict)


class NullSink:
    """A :class:`VerdictSink` that discards every verdict — shielding with auditing off."""

    def write_verdict(self, verdict: SafetyVerdict) -> None:
        """Discard ``verdict``."""
