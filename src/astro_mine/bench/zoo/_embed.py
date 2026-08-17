# SPDX-License-Identifier: Apache-2.0
"""Deterministic scenario embeddings for pgvector similarity search (bench#33; bench.md §5).

bench.md §5 puts the zoo catalog in PostgreSQL with "**pgvector** for similarity/search". A vector
index needs vectors — and Bench cannot get them from a text-embedding model, because that would drag
a neural runtime and a model download into a package whose whole discipline is ``core + pydantic``
and whose tier-1 promise is *offline, no account, no cloud*.

So the default embedder is a **feature-hashed bag of tokens**: the scenario's identity, name,
description, pinned content ids, and metric names are tokenized, each token is hashed to a bucket
and a sign (BLAKE2b — stable across processes and releases, unlike Python's salted ``hash``), and
the resulting sparse vector is L2-normalized. It is:

- **deterministic** — the same spec always embeds to the same vector, on any machine, so a catalog
  seeded twice is byte-identical and a leaderboard's lineage stays reproducible (CX-REPRO);
- **offline** — no model, no download, no network;
- **useful** — scenarios sharing a world, a fleet, a resource species, or a metric set land near
  each other, which is what "find me scenarios like this one" actually means in a benchmark zoo.

It is a *lexical* embedding, not a semantic one: it will not match "ice" to "volatiles". The
:class:`Embedder` seam is where a deployment swaps in a real text-embedding model, without changing
the catalog, the schema, or the query — only the vectors.

Backlog: bench#33 — astro-mine-bench#33
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Iterable, Sequence
from typing import Protocol

from astro_mine.bench.scenario import ScenarioSpec

__all__ = [
    "EMBEDDING_DIM",
    "Embedder",
    "cosine_distance",
    "embed_scenario",
    "embed_text",
    "scenario_tokens",
]

#: The vector width — the ``vector(N)`` column pgvector indexes. Small enough to store cheaply for
#: a community-scale zoo, wide enough that hash collisions do not dominate the signal.
EMBEDDING_DIM = 256

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder(Protocol):
    """Maps a scenario to the vector the catalog indexes — the seam a real model swaps into."""

    def __call__(self, spec: ScenarioSpec) -> tuple[float, ...]:
        """The scenario's embedding, of width :data:`EMBEDDING_DIM`."""
        ...


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens — the unit the hashing embedder buckets."""
    return _TOKEN.findall(text.lower())


def scenario_tokens(spec: ScenarioSpec) -> tuple[str, ...]:
    """The tokens that characterize a scenario for similarity search.

    Deliberately *not* the whole spec: content **hashes** and seeds carry no similarity signal (two
    revisions of the same world differ in every hash bit), so what is embedded is the scenario's
    meaning — its identity, its prose, the *ids* of the world/fleet/prospect/link content it pins,
    and the metrics it is scored on.
    """
    parts: list[str] = [spec.scenario_id, spec.name, spec.description or ""]
    parts += [ref.id for ref in spec.content_refs()]
    parts += [metric.name for metric in spec.metrics]
    parts += list(spec.core_interface)
    tokens: list[str] = []
    for part in parts:
        tokens.extend(_tokenize(part))
    return tuple(tokens)


def _bucket(token: str) -> tuple[int, float]:
    """Hash ``token`` to a (bucket, sign). BLAKE2b: stable across processes and releases."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    # The low bit picks the sign — the signed-hashing trick that keeps collisions from only ever
    # reinforcing each other.
    return value % EMBEDDING_DIM, 1.0 if value & (1 << 63) else -1.0


def _hash_embed(tokens: Iterable[str]) -> tuple[float, ...]:
    """Feature-hash ``tokens`` into an L2-normalized vector of width :data:`EMBEDDING_DIM`."""
    vector = [0.0] * EMBEDDING_DIM
    for token in tokens:
        index, sign = _bucket(token)
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return tuple(vector)
    return tuple(value / norm for value in vector)


def embed_text(text: str) -> tuple[float, ...]:
    """Embed a free-text search query into the catalog's vector space."""
    return _hash_embed(_tokenize(text))


def embed_scenario(spec: ScenarioSpec) -> tuple[float, ...]:
    """Embed a :class:`ScenarioSpec` — the default :class:`Embedder` the catalog indexes with."""
    return _hash_embed(scenario_tokens(spec))


def cosine_distance(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine distance in ``[0, 2]`` — what pgvector's ``<=>`` operator computes.

    The SQLite backend ranks with this in Python; Postgres ranks with ``<=>`` in the index. Both
    answer the same query, so the search path is verified locally and deploys on pgvector.
    """
    if len(left) != len(right):
        raise ValueError(f"cannot compare vectors of width {len(left)} and {len(right)}")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0
    return 1.0 - (dot / (left_norm * right_norm))
