# SPDX-License-Identifier: Apache-2.0
"""Discovery over the catalog — faceted + full-text + capability + semantic (RM-P1-HUB-02).

The query planner hub.md §3 describes: **faceted** browse (kind / license / namespace, excluding
deprecated/yanked by default), **full-text** over the manifest document, **capability negotiation**
(``PluginManifest.satisfies`` — Core's rule, so a query returns only artifacts that satisfy the
requested Core interface versions + capability tags), and **semantic** "find something like this"
ranking through the swappable :class:`~astro_mine.hub.search.EmbeddingProvider`.

**Two execution paths, one ranking.** A catalog that can answer in its own query language — a
:class:`SqlSearchable`, i.e. :class:`~astro_mine.hub.index._sql.SqlCatalog` — has the facets,
capability tags, and full-text pushed **into SQL**, and on PostgreSQL the semantic top-k runs as a
**pgvector** ``ORDER BY embedding <=> :query`` against an HNSW index (hub.md §8 "sub-second …
semantic queries at catalog sizes of 10^5-10^6"), so nothing is scanned in Python. An in-memory
catalog filters the same predicates in Python. Both hand the *same candidate set* to the *same*
ranker below, so ranking and tie-breaks are identical on either backend and the SQL path stays a
pruning optimization, never a second definition of relevance.

Semantic **degrades** to lexical/faceted when embeddings are unavailable (hub.md §9 principle 9) —
if no candidate has a non-zero vector, the ranker falls back to token-overlap.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from astro_mine.hub._embed import cosine, tokenize
from astro_mine.hub.index import Catalog, CatalogEntry
from astro_mine.hub.index._ingest import index_document
from astro_mine.hub.search._provider import EmbeddingProvider, default_provider

__all__ = ["SearchQuery", "SearchResult", "SqlSearchable", "search"]

#: How many rows the SQL path fetches per requested result when a Python-side capability filter
#: still has to run after it — the vector top-k is pruned, so it must be oversampled to stay exact.
_OVERSAMPLE = 10


@dataclass(frozen=True)
class SearchQuery:
    """A discovery request: facets + full-text + capability constraints + semantic query."""

    text: str | None = None
    semantic: str | None = None
    kind: str | None = None
    #: Hub's *container* kind — the payload-shape axis, independent of :attr:`kind` (the Core
    #: interface). Filtering on one never implies the other.
    artifact_kind: str | None = None
    license: str | None = None
    namespace: str | None = None
    interfaces: Mapping[str, str] | None = None
    capability_tags: Sequence[str] | None = None
    include_deprecated: bool = False
    include_yanked: bool = False
    use_semantic: bool = True
    limit: int = 20


@dataclass(frozen=True)
class SearchResult:
    """A matched catalog entry and its relevance score."""

    entry: CatalogEntry
    score: float


@runtime_checkable
class SqlSearchable(Protocol):
    """A catalog that can execute a :class:`SearchQuery` itself (facets/text/tags — and vectors).

    Implemented by :class:`~astro_mine.hub.index._sql.SqlCatalog`. :func:`search` detects it
    structurally, so any future backend (OpenSearch — hub.md §11) becomes the query engine by
    implementing this one method; no caller changes.
    """

    def search_entries(
        self,
        query: SearchQuery,
        *,
        query_vector: tuple[float, ...] | None = None,
        limit: int | None = None,
    ) -> list[CatalogEntry]:
        """Entries matching ``query``'s facets/full-text/tags, vector-ordered when possible."""
        ...


def _facet_match(entry: CatalogEntry, query: SearchQuery) -> bool:
    if entry.yanked and not query.include_yanked:
        return False
    if entry.deprecated and not query.include_deprecated:
        return False
    if query.kind is not None and entry.kind != query.kind:
        return False
    if query.artifact_kind is not None and entry.artifact_kind != query.artifact_kind:
        return False
    if query.license is not None and entry.license != query.license:
        return False
    return not (query.namespace is not None and entry.namespace != query.namespace)


def _capability_match(entry: CatalogEntry, query: SearchQuery) -> bool:
    if not query.interfaces and not query.capability_tags:
        return True
    return entry.satisfies(interfaces=query.interfaces, capability_tags=query.capability_tags)


def _full_text_match(entry: CatalogEntry, query: SearchQuery) -> bool:
    if not query.text:
        return True
    haystack = set(tokenize(index_document(entry.manifest)))
    return all(token in haystack for token in tokenize(query.text))


def _lexical_score(entry: CatalogEntry, terms: Sequence[str]) -> float:
    if not terms:
        return 0.0
    haystack = set(tokenize(index_document(entry.manifest)))
    return sum(1 for term in terms if term in haystack) / len(terms)


def _candidates(
    catalog: Catalog, query: SearchQuery, query_vector: tuple[float, ...] | None
) -> list[CatalogEntry]:
    """The filtered candidate set — pushed into SQL when the catalog can run the query itself."""
    if isinstance(catalog, SqlSearchable):
        # The SQL backend applies facets, full-text, and capability *tags*; interface-version
        # satisfaction is Core's SemVer rule (`PluginManifest.satisfies`) and stays in Python, so
        # the limit is oversampled when a pruned top-k must still survive that second filter.
        #
        # The limit is a *hint*: the backend applies it only when its own order is already the
        # final order (a vector top-k, or a pure facet browse), and ignores it when this ranker
        # still has to run — truncating before ranking could drop a better match. Only the backend
        # knows which it did, so the decision lives there.
        limit = query.limit * _OVERSAMPLE if query.interfaces else query.limit
        entries = catalog.search_entries(query, query_vector=query_vector, limit=limit)
        return [entry for entry in entries if _capability_match(entry, query)]

    return [
        entry
        for entry in catalog.all()
        if _facet_match(entry, query)
        and _capability_match(entry, query)
        and _full_text_match(entry, query)
    ]


def search(
    catalog: Catalog, query: SearchQuery, *, provider: EmbeddingProvider | None = None
) -> list[SearchResult]:
    """Return the ranked entries matching ``query`` (top ``limit``).

    Filters by facets → capability negotiation → full-text (in SQL where the catalog supports it),
    then ranks: semantic cosine when a ``semantic`` query is given and any candidate has an
    embedding, else lexical token-overlap (the faceted/keyword degradation). ``provider`` embeds the
    query — it MUST be the provider the catalog was ingested with, since vectors from different
    providers are not comparable. Ties break by ``reference`` for a stable order.
    """
    backend = provider if provider is not None else default_provider()
    query_vector = (
        backend.embed(query.semantic) if (query.semantic and query.use_semantic) else None
    )
    candidates = _candidates(catalog, query, query_vector)

    if query_vector is not None:
        scored = [(entry, cosine(query_vector, entry.embedding)) for entry in candidates]
        if any(score > 0.0 for _, score in scored):
            scored.sort(key=lambda pair: (-pair[1], pair[0].reference))
            return [SearchResult(entry, score) for entry, score in scored[: query.limit]]

    terms = tokenize(f"{query.text or ''} {query.semantic or ''}")
    ranked = sorted(candidates, key=lambda entry: (-_lexical_score(entry, terms), entry.reference))
    return [SearchResult(entry, _lexical_score(entry, terms)) for entry in ranked[: query.limit]]
