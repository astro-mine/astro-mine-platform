"""Content hashing for decision-trace provenance (RM-P1-MIND-07; conventions.md §5).

Every emitted plan records the content-addressed identities of its inputs — the stack spec,
SADF, belief snapshot, comms model, ONNX policy artifacts — so a Bench result or an Ops replan
reproduces exactly (principle 8). :func:`content_hash` is the one canonical digest the trace's
:class:`~astro_mine.mind.trace.model.DecisionProvenance` records; the composer threads the
resulting map into the graph's provenance.
"""

from __future__ import annotations

import hashlib

__all__ = ["content_hash"]


def content_hash(data: str | bytes) -> str:
    """The canonical content hash of ``data`` (``sha256:<hex>``) — an artifact's identity."""
    payload = data.encode("utf-8") if isinstance(data, str) else data
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
