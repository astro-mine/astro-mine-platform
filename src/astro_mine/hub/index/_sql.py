# SPDX-License-Identifier: Apache-2.0
"""A SQLAlchemy-backed :class:`~astro_mine.hub.index.Catalog` (RM-P1-HUB-02; hub.md §5, §8, §11).

One store, two databases: the same SQLAlchemy Core code runs on **SQLite** (tests + a laptop — the
tier-1 fallback that needs no server and no pgvector) and on **PostgreSQL + pgvector** (the
``docker compose`` deployment / the integration job), selected only by the connection URL.

**The row is a queryable projection, not an opaque blob.** hub.md §5 specifies "one row per artifact
version with the indexed manifest projection (kind, Core interface versions, capability tags,
license, provenance, publisher, namespace), download/usage counters, deprecation/yank status, and
the **pgvector** embedding for semantic search". So each entry is stored as:

- **facet columns** — ``name``/``version``/``kind``/``namespace``/``publisher``/``license``/
  ``deprecated``/``yanked``/``downloads`` — indexed, so faceted browse is a SQL ``WHERE``;
- **a tag table** (``catalog_tags``) — one row per capability tag, so tag filters are a SQL
  ``EXISTS`` on an index rather than a Python scan of every manifest;
- **a token document** — the full-text index (space-delimited tokens, matched with ``LIKE '% t %'``
  so SQL token semantics are *identical* to the in-memory path's set membership);
- **an embedding column** — a real **pgvector** ``vector(dim)`` with an **HNSW** cosine index on
  PostgreSQL (hub.md §8: "sub-second … top-k semantic queries at catalog sizes of 10^5-10^6"),
  degrading to a JSON column + Python ranking on SQLite (hub.md §9 principle 9);
- **the JSON ``body``** — still the source of truth a :class:`CatalogEntry` is rehydrated from, so
  the row remains the Core-manifest projection and never a Hub-private schema (hub.md §2
  principle 2). The columns are *derived* facets.

Almost all of those facets are projections of the Core manifest. The exception is
``artifact_kind`` — Hub's **container** vocabulary (payload shape), which Core does not describe
and deliberately should not (hub#33): container shape is a packaging concern, and widening Core to
absorb it would put a packaging problem in the narrow waist. It is a *second* axis alongside
``kind`` (the Core interface), never a replacement for it.

**Schema note.** This module creates tables with ``create_all``, which does not alter an existing
one. A database created before ``artifact_kind`` existed needs the column added by hand::

    ALTER TABLE catalog ADD COLUMN artifact_kind VARCHAR(64);
    CREATE INDEX ix_catalog_artifact_kind ON catalog (artifact_kind);

Existing rows read back with ``artifact_kind = NULL``, which is a valid value (an artifact may
genuinely carry no container kind), so nothing breaks before the backfill — entries simply do not
answer container-kind queries until they are re-indexed.

Requires the ``[service]`` extra (SQLAlchemy; ``psycopg`` + ``pgvector`` for PostgreSQL). Imported
lazily, so the base package + :class:`~astro_mine.hub.index.InMemoryCatalog` stay dependency-clean
and the offline tier needs none of it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    and_,
    create_engine,
    delete,
    insert,
    select,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.sql import ColumnElement, Select
from sqlalchemy.types import TypeEngine

from astro_mine.hub._embed import EMBED_DIM, tokenize
from astro_mine.hub.index._catalog import CatalogEntry
from astro_mine.hub.index._ingest import index_document

if TYPE_CHECKING:  # the query type lives in search/; importing it at runtime would be a cycle
    from astro_mine.hub.search import SearchQuery

__all__ = ["SqlCatalog"]

#: ``hnsw.ef_search`` for vector-ordered queries — pgvector's maximum, i.e. fully recall-biased.
#: The server default (40) is speed-biased and measurably under-recalls on tie-heavy catalogs: at
#: 10^4 hashing-embedded entries it drops an *exact-match* neighbour from the top-10 in ~1/3 of
#: graph builds, because the search frontier fills with equidistant ties before it reaches the true
#: nearest node — and intermediate values (200) still missed occasionally, since the tie plateau
#: around a query can hold hundreds of nodes. Catalog search is interactive discovery and the cap
#: measured ~4 ms at that scale, far inside the hub.md §8 sub-second budget, so there is nothing to
#: buy with a lower value.
_HNSW_EF_SEARCH = 1000


def vector_type(dim: int) -> TypeEngine[Any] | None:
    """The ``vector(dim)`` column type, or ``None`` when **pgvector** is not installed.

    pgvector is a PostgreSQL-only concern (it ships in the ``[service]`` extra): the SQLite/offline
    tier-1 path must never require it, and falls back to a JSON column + Python ranking.
    """
    try:
        from pgvector.sqlalchemy import Vector
    except ImportError:  # pragma: no cover - the pgvector-less environment
        return None
    column: TypeEngine[Any] = Vector(dim)
    return column


def _document(entry: CatalogEntry) -> str:
    """The full-text column: space-delimited tokens, padded so ``LIKE '% t %'`` matches tokens.

    Padding is what makes SQL full-text agree *exactly* with the in-memory path (set membership of
    tokens) instead of degenerating into substring matching — "dig" must not match "digging".
    """
    return f" {' '.join(tokenize(index_document(entry.manifest)))} "


class SqlCatalog:
    """A durable :class:`Catalog` over any SQLAlchemy engine (SQLite / PostgreSQL + pgvector)."""

    def __init__(
        self,
        url: str | None = None,
        *,
        engine: Engine | None = None,
        embedding_dim: int = EMBED_DIM,
    ) -> None:
        """Open (or reuse) an engine for ``url``, create the schema, and index the vector column.

        ``embedding_dim`` MUST match the :class:`~astro_mine.hub.search.EmbeddingProvider` the
        catalog is ingested with — it is the width of the ``vector(dim)`` column.
        """
        if engine is None:
            if url is None:
                raise ValueError("SqlCatalog needs a database url or an engine")
            engine = create_engine(url)
        self._engine = engine
        self._dim = embedding_dim

        #: pgvector is used only where it exists: PostgreSQL with the package installed.
        vector = vector_type(embedding_dim) if engine.dialect.name == "postgresql" else None
        self._vector = vector is not None
        if self._vector:
            with self._engine.begin() as connection:
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))

        embedding: TypeEngine[Any] = vector if vector is not None else JSON()
        self._metadata = MetaData()
        self._catalog = Table(
            "catalog",
            self._metadata,
            Column("reference", String(512), primary_key=True),
            Column("name", String(256), nullable=False, index=True),
            Column("version", String(64), nullable=False),
            Column("kind", String(64), nullable=False, index=True),
            # Hub's *container* kind — a second, independent axis from `kind` (the Core interface
            # the manifest declares). Nullable: artifacts indexed before this facet existed, and
            # any published by another OCI tool, legitimately carry none.
            Column("artifact_kind", String(64), nullable=True, index=True),
            Column("namespace", String(64), nullable=False, index=True),
            Column("publisher", String(256), nullable=False, index=True),
            Column("license", String(128), nullable=True, index=True),
            Column("deprecated", Boolean, nullable=False, default=False, index=True),
            Column("yanked", Boolean, nullable=False, default=False, index=True),
            Column("downloads", Integer, nullable=False, default=0),
            Column("document", Text, nullable=False),
            Column("embedding_provider", String(128), nullable=False, default=""),
            Column("embedding", embedding, nullable=True),
            Column("body", JSON, nullable=False),
        )
        self._tags = Table(
            "catalog_tags",
            self._metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column(
                "reference",
                String(512),
                ForeignKey("catalog.reference", ondelete="CASCADE"),
                nullable=False,
                index=True,
            ),
            Column("tag", String(128), nullable=False, index=True),
            UniqueConstraint("reference", "tag", name="uq_catalog_tag"),
        )
        self._metadata.create_all(engine)

        if self._vector:
            # HNSW over cosine distance — the index hub.md §8 names for top-k semantic search.
            with self._engine.begin() as connection:
                connection.execute(
                    text(
                        "CREATE INDEX IF NOT EXISTS catalog_embedding_hnsw ON catalog "
                        "USING hnsw (embedding vector_cosine_ops)"
                    )
                )

    # -- Catalog contract -----------------------------------------------------------------------

    def add(self, entry: CatalogEntry) -> None:
        body: dict[str, Any] = entry.model_dump(mode="json")
        embedding = list(entry.embedding) if entry.embedding else None
        if embedding is not None and len(embedding) != self._dim:
            raise ValueError(
                f"{entry.reference} carries a {len(embedding)}-d embedding but this catalog's "
                f"vector column is {self._dim}-d (provider {entry.embedding_provider!r})"
            )
        with self._engine.begin() as connection:
            connection.execute(delete(self._tags).where(self._tags.c.reference == entry.reference))
            connection.execute(
                delete(self._catalog).where(self._catalog.c.reference == entry.reference)
            )
            connection.execute(
                insert(self._catalog).values(
                    reference=entry.reference,
                    name=entry.name,
                    version=entry.version,
                    kind=entry.kind,
                    artifact_kind=entry.artifact_kind,
                    namespace=entry.namespace,
                    publisher=entry.publisher,
                    license=entry.license,
                    deprecated=entry.deprecated,
                    yanked=entry.yanked,
                    downloads=entry.downloads,
                    document=_document(entry),
                    embedding_provider=entry.embedding_provider,
                    embedding=embedding,
                    body=body,
                )
            )
            if tags := entry.capability_tags:
                connection.execute(
                    insert(self._tags),
                    [{"reference": entry.reference, "tag": tag} for tag in tags],
                )

    def get(self, reference: str) -> CatalogEntry | None:
        statement = select(self._catalog.c.body).where(self._catalog.c.reference == reference)
        with self._engine.connect() as connection:
            row = connection.execute(statement).fetchone()
        return None if row is None else CatalogEntry.model_validate(row[0])

    def all(self) -> list[CatalogEntry]:
        statement = select(self._catalog.c.body).order_by(self._catalog.c.reference)
        with self._engine.connect() as connection:
            rows = connection.execute(statement).fetchall()
        return [CatalogEntry.model_validate(row[0]) for row in rows]

    # -- SqlSearchable: the query runs in the database ------------------------------------------

    def _predicates(self, query: SearchQuery) -> list[ColumnElement[bool]]:
        """``query``'s facets, full-text, and capability tags as SQL — no Python scan."""
        catalog, tags = self._catalog, self._tags
        clauses: list[ColumnElement[bool]] = []
        if not query.include_yanked:
            clauses.append(catalog.c.yanked.is_(False))
        if not query.include_deprecated:
            clauses.append(catalog.c.deprecated.is_(False))
        if query.kind is not None:
            clauses.append(catalog.c.kind == query.kind)
        if query.license is not None:
            clauses.append(catalog.c.license == query.license)
        if query.namespace is not None:
            clauses.append(catalog.c.namespace == query.namespace)
        for token in tokenize(query.text or ""):
            clauses.append(catalog.c.document.like(f"% {token} %"))
        for tag in query.capability_tags or ():
            clauses.append(
                select(tags.c.id)
                .where(and_(tags.c.reference == catalog.c.reference, tags.c.tag == tag))
                .exists()
            )
        return clauses

    def search_entries(
        self,
        query: SearchQuery,
        *,
        query_vector: tuple[float, ...] | None = None,
        limit: int | None = None,
    ) -> list[CatalogEntry]:
        """Entries matching ``query`` — facets/full-text/tags in SQL, top-k by **pgvector** on PG.

        Rows are ordered by **pgvector cosine distance** when a ``query_vector`` is given and this
        backend has the extension (the HNSW top-k), else by ``reference``.

        ``limit`` is a **hint**, applied only when the database's order is already the *final* order
        — i.e. when the rows were vector-ordered here, or when the query has nothing to rank by at
        all (a pure facet browse, which both backends order by ``reference``). It is deliberately
        **ignored** otherwise: a lexical query is ranked by the caller *after* this returns, and on
        SQLite a semantic query is too (no vector index), so truncating here would silently drop
        better matches. Only the backend knows which of those it just did, so only the backend can
        decide. Capability *interface-version* satisfaction is Core's SemVer rule and stays with the
        caller (:mod:`astro_mine.hub.search`).
        """
        statement: Select[Any] = select(self._catalog.c.body).where(*self._predicates(query))

        vector_ordered = self._vector and query_vector is not None and any(query_vector)
        if vector_ordered and query_vector is not None:
            statement = statement.order_by(
                self._catalog.c.embedding.cosine_distance(list(query_vector))
            )
        else:
            statement = statement.order_by(self._catalog.c.reference)

        ranks_in_caller = bool(query.text) or query_vector is not None
        if limit is not None and (vector_ordered or not ranks_in_caller):
            statement = statement.limit(limit)

        with self._engine.connect() as connection:
            if vector_ordered:
                # Transaction-local, so it never leaks to other work on the pooled connection.
                connection.execute(
                    text("SELECT set_config('hnsw.ef_search', :ef, true)"),
                    {"ef": str(_HNSW_EF_SEARCH)},
                )
            rows = connection.execute(statement).fetchall()
        return [CatalogEntry.model_validate(row[0]) for row in rows]

    def dispose(self) -> None:
        """Close pooled connections (production keeps one long-lived engine)."""
        self._engine.dispose()
