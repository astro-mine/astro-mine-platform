"""Content-addressed result cache (studio.md §8; STUDIO-03 acceptance).

Identical ``(design, world, seed)`` tuples are never re-evaluated: the cache key is the
content hash of the candidate digest + the objective wire hash + the seed. This is also
the checkpoint that makes a batch **resumable** — a killed run resumes by skipping keys
already present. Phase-1 ships the in-memory tier; the Hub-backed cache (``HubClient``
pull/verify, deferred) implements the same :class:`ResultCache` Protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from astro_mine.core.objective import ObjectiveDocument, to_wire

from ..hashing import content_hash, content_hash_json
from ..models import DesignCandidate, EvaluatedCandidate


def cache_key(candidate: DesignCandidate, objective: ObjectiveDocument, seed: int) -> str:
    """The content-addressed identity of one candidate evaluation."""
    return content_hash_json(
        {
            "candidate": candidate.digest(),
            "objective": content_hash(to_wire(objective)),
            "seed": seed,
        }
    )


@runtime_checkable
class ResultCache(Protocol):
    def has(self, key: str) -> bool: ...

    def get(self, key: str) -> EvaluatedCandidate: ...

    def put(self, key: str, value: EvaluatedCandidate) -> None: ...


class InMemoryResultCache:
    """Tier-1 in-process :class:`ResultCache`."""

    def __init__(self) -> None:
        self._entries: dict[str, EvaluatedCandidate] = {}

    def has(self, key: str) -> bool:
        return key in self._entries

    def get(self, key: str) -> EvaluatedCandidate:
        return self._entries[key]

    def put(self, key: str, value: EvaluatedCandidate) -> None:
        self._entries[key] = value
