# SPDX-License-Identifier: Apache-2.0
"""Deterministic local text embedding for semantic search (RM-P1-HUB-02).

The default search provider's embedding: a **feature-hashing bag-of-tokens** vector — no model, no
network, deterministic — so tier-1 local semantic search works **offline** (conventions §7 tier-1).
It is a real vector-space model (cosine similarity ranks "find something like this"), just a simple
one; the hosted tier swaps in a learned embedding + pgvector behind the same provider seam
(hub.md §3, §11). Search **degrades** to faceted/keyword when embeddings are unavailable
(hub.md §9 principle 9) — the ranker falls back to lexical overlap when a vector is empty.
"""

from __future__ import annotations

import hashlib
import math
import re

__all__ = ["EMBED_DIM", "cosine", "embed", "tokenize"]

#: Embedding dimensionality (small — this is a lexical stand-in, not a learned model).
EMBED_DIM = 64

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens of ``text``."""
    return _TOKEN.findall(text.lower())


def embed(text: str, *, dim: int = EMBED_DIM) -> tuple[float, ...]:
    """A deterministic L2-normalized feature-hashing vector of ``text`` (empty text → zero vector).

    Each token is hashed to a bucket and a sign (the signed-hashing trick, so collisions partly
    cancel), then the vector is L2-normalized so :func:`cosine` is a true cosine. Same text always
    yields the same vector — reproducible content addressing of the semantic index.
    """
    vec = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big") % dim
        sign = 1.0 if digest[4] & 1 == 0 else -1.0
        vec[bucket] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm == 0.0:
        return tuple(vec)
    return tuple(v / norm for v in vec)


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Cosine similarity of two embeddings (0.0 if either is empty or shapes differ)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b, strict=True))
