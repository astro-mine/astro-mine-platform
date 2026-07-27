"""Canonical-JSON serialization of a decision trace (RM-P1-MIND-01).

The determinism gate's wire form: a byte-stable JSON rendering of a
:class:`~astro_mine.mind.trace.model.DecisionTrace` — sorted keys, fixed indentation,
trailing newline — so a seeded run compares byte-for-byte against a stored golden and CI
fails on any drift (conventions.md §11; mind.md §10). This is the JSON serializer over the
neutral trace records; the MCAP serializer (RM-P1-MIND-07) is a sibling over the same
records.
"""

from __future__ import annotations

import json

from astro_mine.mind.trace.model import DecisionTrace

__all__ = ["to_canonical_json"]


def to_canonical_json(trace: DecisionTrace) -> str:
    """Render ``trace`` to canonical JSON (sorted keys, 2-space indent, trailing newline).

    Deterministic for a deterministic trace: identical decisions ⇒ identical bytes.
    """
    return (
        json.dumps(
            trace.to_dict(),
            sort_keys=True,
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
