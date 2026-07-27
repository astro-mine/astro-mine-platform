"""Decision traces — Mind's reproducibility and explanation substrate.

The neutral record model (:mod:`astro_mine.mind.trace.model`) plus the canonical-JSON serializer
(:mod:`astro_mine.mind.trace.canonical`) that backs the byte-for-byte determinism gate.
RM-P1-MIND-07 adds content-hash provenance (:mod:`astro_mine.mind.trace.provenance`), the plan
explanation (:mod:`astro_mine.mind.trace.explain`), and — behind the optional ``[recording]``
extra — the MCAP serializer (:mod:`astro_mine.mind.trace.mcap`, imported explicitly).
"""

from __future__ import annotations

from astro_mine.mind.trace.canonical import to_canonical_json
from astro_mine.mind.trace.explain import PlanExplanation, TickExplanation, explain
from astro_mine.mind.trace.model import (
    DecisionProvenance,
    DecisionTrace,
    ShieldRecord,
    TickRecord,
    TierDecisionRecord,
)
from astro_mine.mind.trace.provenance import content_hash

__all__ = [
    "DecisionProvenance",
    "DecisionTrace",
    "PlanExplanation",
    "ShieldRecord",
    "TickExplanation",
    "TickRecord",
    "TierDecisionRecord",
    "content_hash",
    "explain",
    "to_canonical_json",
]
