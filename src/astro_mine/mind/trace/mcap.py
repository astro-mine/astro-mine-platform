# SPDX-License-Identifier: Apache-2.0
"""MCAP decision-trace serialization (RM-P1-MIND-07; conventions.md §4).

The replayable-in-View / object-store form of a
:class:`~astro_mine.mind.trace.model.DecisionTrace`: a single MCAP container carrying
heterogeneous, timestamped, schema-tagged channels (conventions.md §4) — one per decision-record
kind the executive emits (mind.md §10): tier decisions, plan revisions (replans, with their
trigger), Guard interventions (with clause/certificate provenance, RM-P1-MIND-05), and fallback /
degradation activations (RM-P1-MIND-06). The trace's :class:`DecisionProvenance` (plugin set,
Core interface versions, seed, input content hashes) rides as MCAP metadata, so the container is
self-describing and auditable.

A sibling serializer to :mod:`astro_mine.mind.trace.canonical` over the **same** neutral records —
the canonical JSON stays the byte-for-byte determinism gate; MCAP is the streamable form. Message
payloads are deterministic given a deterministic trace (seed + pinned plugins + fixed inputs), so
a replay reproduces. Requires the optional ``[recording]`` extra (``mcap``).
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

from mcap.reader import make_reader
from mcap.writer import Writer

from astro_mine.mind.trace.model import DecisionTrace, TickRecord

__all__ = ["MCAP_CHANNELS", "read_mcap_messages", "to_mcap_bytes", "write_mcap"]

#: The schema-tagged channels the trace is fanned out to (topic → JSON-schema title).
MCAP_CHANNELS: dict[str, str] = {
    "mind/tier_decision": "mind.TierDecision",
    "mind/plan_revision": "mind.PlanRevision",
    "mind/guard_intervention": "mind.GuardIntervention",
    "mind/fallback_activation": "mind.FallbackActivation",
}

_OBJECT_SCHEMA = json.dumps({"type": "object"}).encode("utf-8")


def to_mcap_bytes(trace: DecisionTrace) -> bytes:
    """Serialize ``trace`` to an in-memory MCAP container."""
    buffer = io.BytesIO()
    writer = Writer(buffer)
    writer.start()
    channels = {
        topic: writer.register_channel(
            topic=topic,
            message_encoding="json",
            schema_id=writer.register_schema(
                name=title, encoding="jsonschema", data=_OBJECT_SCHEMA
            ),
        )
        for topic, title in MCAP_CHANNELS.items()
    }
    writer.add_metadata("mind.provenance", _provenance_metadata(trace))

    sequence = 0
    for tick in trace.ticks:
        for message in _tick_messages(tick):
            topic, payload = message
            writer.add_message(
                channel_id=channels[topic],
                log_time=_nanos(tick.sim_time_s),
                publish_time=_nanos(tick.sim_time_s),
                sequence=sequence,
                data=json.dumps(payload, sort_keys=True).encode("utf-8"),
            )
            sequence += 1
    writer.finish()
    return buffer.getvalue()


def write_mcap(trace: DecisionTrace, path: str | Path) -> None:
    """Write ``trace`` to ``path`` as an MCAP file."""
    Path(path).write_bytes(to_mcap_bytes(trace))


def read_mcap_messages(source: bytes | str | Path) -> list[tuple[str, dict[str, Any]]]:
    """Decode an MCAP container to ``(topic, payload)`` messages, in log order."""
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    reader = make_reader(io.BytesIO(data))
    out: list[tuple[str, dict[str, Any]]] = []
    for _schema, channel, message in reader.iter_messages():
        out.append((channel.topic, json.loads(message.data)))
    return out


def _tick_messages(tick: TickRecord) -> list[tuple[str, dict[str, Any]]]:
    messages: list[tuple[str, dict[str, Any]]] = []
    for tier in tick.tiers:
        base = {"tick": tick.tick, "sim_time_s": tick.sim_time_s, **tier.to_dict()}
        messages.append(("mind/tier_decision", base))
        if tier.replanned:
            messages.append(("mind/plan_revision", base))
        if tier.fallback_used or tier.note is not None:
            messages.append(("mind/fallback_activation", base))
    if tick.shield.intervened:
        messages.append(
            (
                "mind/guard_intervention",
                {"tick": tick.tick, "sim_time_s": tick.sim_time_s, **tick.shield.to_dict()},
            )
        )
    return messages


def _provenance_metadata(trace: DecisionTrace) -> dict[str, str]:
    provenance = trace.provenance
    return {
        "stack_id": trace.stack_id,
        "seed": str(provenance.seed),
        "plugin_versions": json.dumps(dict(provenance.plugin_versions), sort_keys=True),
        "core_interface_versions": json.dumps(
            dict(provenance.core_interface_versions), sort_keys=True
        ),
        "input_hashes": json.dumps(dict(provenance.input_hashes), sort_keys=True),
    }


def _nanos(sim_time_s: float) -> int:
    return round(sim_time_s * 1_000_000_000)
