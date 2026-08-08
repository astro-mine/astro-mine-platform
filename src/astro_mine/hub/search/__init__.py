"""Discovery — faceted + full-text + capability + semantic search over the catalog (RM-P1-HUB-02).

The query surface hub.md §3 exposes over :mod:`astro_mine.hub.index`: precision facets and full
text, **capability negotiation** against Core manifests, and **semantic** "find something like
this" ranking — degrading to keyword/faceted when embeddings are unavailable (hub.md §9).

- :func:`search` — the query planner; pushes facets/text/tags (and, on PostgreSQL, the **pgvector**
  top-k) into SQL when the catalog is a :class:`SqlSearchable`, else filters in Python.
- :class:`EmbeddingProvider` — the swappable semantic backend (hub.md §3 extension points):
  :class:`HashingEmbedding` offline by default, :class:`HttpEmbedding` for a served learned model.

Backlog: RM-P1-HUB-02 — astro-mine-hub#2
"""

from __future__ import annotations

from astro_mine.hub._embed import EMBED_DIM, cosine, embed, tokenize
from astro_mine.hub.search._provider import (
    EmbeddingError,
    EmbeddingProvider,
    HashingEmbedding,
    HttpEmbedding,
    default_provider,
)
from astro_mine.hub.search._search import SearchQuery, SearchResult, SqlSearchable, search

__all__ = [
    "EMBED_DIM",
    "EmbeddingError",
    "EmbeddingProvider",
    "HashingEmbedding",
    "HttpEmbedding",
    "SearchQuery",
    "SearchResult",
    "SqlSearchable",
    "cosine",
    "default_provider",
    "embed",
    "search",
    "tokenize",
]
