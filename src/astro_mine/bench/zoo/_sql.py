"""The Postgres/pgvector scenario-zoo catalog (bench#33; bench.md §5, §11).

bench.md §5's data architecture puts the **scenario zoo catalog in PostgreSQL** — *"Indexes specs,
versions, lineage; **pgvector** for similarity/search"*. This is that catalog, and it follows the
house pattern already set by :class:`~astro_mine.bench.leaderboard._sql.SqlStore`: **one store, two
databases**. The same SQLAlchemy Core code runs on

- **SQLite** — the tests and a laptop. Embeddings are stored as JSON and ranked with
  :func:`~astro_mine.bench.zoo._embed.cosine_distance` in Python, so the *whole* catalog path
  (schema, upsert, lineage, similarity ranking, seeding) is exercised with no database server; and
- **PostgreSQL + pgvector** — the deployment. ``CREATE EXTENSION vector``, a real ``vector(256)``
  column, and ranking pushed down to pgvector's ``<=>`` cosine operator (an **exact** scan — see
  :meth:`SqlCatalog._create_schema` for why an IVFFlat index is deliberately *not* created).

Selected only by the connection URL. Which means the similarity search you test offline is the
similarity search that ships — the dialect changes where the cosine is computed, not what it means.

The filesystem scan remains the tier-1 default (:class:`~astro_mine.bench.zoo.FilesystemCatalog`);
this catalog is opt-in via ``ASTRO_MINE_BENCH_CATALOG_DSN`` and never required to score a scenario
offline (CX-LOCAL; bench#33 AC3). :meth:`SqlCatalog.seed_from` is the migration path that populates
it from the packaged zoo's ``scenario.json`` documents (bench#33 AC4), and is what the
``astro-mine bench zoo-sync`` command runs.

Requires the ``[leaderboard]`` extra (SQLAlchemy; ``psycopg`` for a Postgres URL). Imported lazily,
so the base package stays ``core + pydantic``.

Backlog: bench#33 — astro-mine-bench#33
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from sqlalchemy import (
    JSON,
    Column,
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    insert,
    select,
    text,
)
from sqlalchemy.engine import Engine

from astro_mine.bench.scenario import ScenarioSpec
from astro_mine.bench.zoo._catalog import CatalogEntry, ScenarioCatalog, SearchHit, catalog_entry
from astro_mine.bench.zoo._embed import EMBEDDING_DIM, cosine_distance, embed_text

__all__ = ["SqlCatalog", "pgvector_literal"]

_METADATA = MetaData()

#: The catalog table. ``embedding`` is JSON here and a real ``vector(N)`` column on Postgres — the
#: dialect-specific column is added by :meth:`SqlCatalog._create_schema`, because SQLAlchemy Core
#: cannot express a type that only one dialect has without the pgvector SQLAlchemy adapter.
_SCENARIOS = Table(
    "zoo_scenarios",
    _METADATA,
    Column("scenario_id", String, primary_key=True),
    Column("family", String, nullable=False, index=True),
    Column("version", Integer, nullable=False),
    Column("parent_id", String, nullable=True),
    Column("name", String, nullable=False),
    Column("spec_version", String, nullable=False),
    Column("description", String, nullable=True),
    Column("spec_hash", String, nullable=False, index=True),
    Column("spec", JSON, nullable=False),
    Column("embedding_json", JSON, nullable=False),
)


def pgvector_literal(vector: Sequence[float]) -> str:
    """Render an embedding as a pgvector literal (``[0.1,0.2,…]``) for the ``<=>`` operator."""
    return "[" + ",".join(f"{value:.9g}" for value in vector) + "]"


class SqlCatalog:
    """The hosted scenario catalog: spec + version + lineage index, with vector similarity search.

    A :class:`~astro_mine.bench.zoo.WritableCatalog` — it can be authored into, which the packaged
    filesystem zoo deliberately cannot be.
    """

    def __init__(self, url: str | None = None, *, engine: Engine | None = None) -> None:
        """Open (or reuse) an engine and create the schema — including pgvector, on Postgres."""
        if engine is None:
            if url is None:
                raise ValueError("SqlCatalog needs a database url or an engine")
            engine = create_engine(url)
        self._engine = engine
        self._is_postgres = engine.dialect.name == "postgresql"
        self._create_schema()

    @property
    def uses_pgvector(self) -> bool:
        """Whether similarity search runs on a real pgvector index (Postgres) or in Python (SQLite).

        The distinction is an *implementation* one: both backends answer the same query with the
        same cosine distance, so a result computed either way ranks the catalog identically.
        """
        return self._is_postgres

    def _create_schema(self) -> None:
        """Create the table, plus the ``vector`` extension and column on Postgres.

        **Deliberately no ANN index.** The obvious move is an ``ivfflat`` index over
        ``vector_cosine_ops``, and it is a trap: IVFFlat derives its centroids from the rows present
        when the index is *built*, so an index created as part of the schema — i.e. on an empty
        table, which is exactly when a catalog's schema is created — has no usable lists and makes
        ``<=>`` queries return **zero rows**, silently. (CI caught this against a real pgvector
        server; it would otherwise have shipped as "search finds nothing" in production.) The
        correct IVFFlat workflow is to load data *first* and index *after*, which a catalog that is
        continuously appended to cannot honour.

        So the search is an **exact** ``<=>`` scan over the ``vector`` column. That is still
        pgvector-backed similarity search — the operator, the type, and the distance are pgvector's
        — it simply does not approximate. For a curated scenario zoo (bench.md §5: immutable,
        append-only, community-scale) an exact scan is sub-millisecond and, unlike an ANN index, it
        is guaranteed to rank identically to the SQLite backend, which is what makes the offline
        tier a faithful rehearsal of the deployed one. Add an **HNSW** index (which, unlike IVFFlat,
        builds incrementally from empty) if and when the zoo outgrows an exact scan.
        """
        _METADATA.create_all(self._engine)
        if not self._is_postgres:
            return
        with self._engine.begin() as connection:
            connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            connection.execute(
                text(
                    "ALTER TABLE zoo_scenarios "
                    f"ADD COLUMN IF NOT EXISTS embedding vector({EMBEDDING_DIM})"
                )
            )
            # Drop the IVFFlat index this class used to create: on an empty table it produces no
            # centroids, and queries through it return nothing. Dropping it makes an existing
            # deployment self-heal on the next open rather than keep answering with zero rows.
            connection.execute(text("DROP INDEX IF EXISTS zoo_scenarios_embedding_idx"))

    # -- discovery (the ScenarioCatalog interface) ------------------------------------------------

    def list_scenarios(self) -> tuple[str, ...]:
        statement = select(_SCENARIOS.c.scenario_id).order_by(_SCENARIOS.c.scenario_id)
        with self._engine.connect() as connection:
            return tuple(str(row[0]) for row in connection.execute(statement).fetchall())

    def load_scenario(self, scenario_id: str) -> ScenarioSpec:
        statement = select(_SCENARIOS.c.spec).where(_SCENARIOS.c.scenario_id == scenario_id)
        with self._engine.connect() as connection:
            row = connection.execute(statement).fetchone()
        if row is None:
            available = ", ".join(self.list_scenarios()) or "(none)"
            raise KeyError(f"no zoo scenario {scenario_id!r}; available: {available}")
        return ScenarioSpec.model_validate(row[0])

    def entries(self) -> tuple[CatalogEntry, ...]:
        statement = select(_SCENARIOS).order_by(_SCENARIOS.c.scenario_id)
        with self._engine.connect() as connection:
            rows = connection.execute(statement).mappings().fetchall()
        return tuple(self._to_entry(row) for row in rows)

    @staticmethod
    def _to_entry(row: Any) -> CatalogEntry:
        """Rehydrate a catalog row (the embedding rides the JSON column on both dialects)."""
        embedding = row["embedding_json"]
        if isinstance(embedding, str):  # SQLite may hand JSON back as text
            embedding = json.loads(embedding)
        spec = row["spec"]
        if isinstance(spec, str):
            spec = json.loads(spec)
        return CatalogEntry(
            scenario_id=row["scenario_id"],
            name=row["name"],
            family=row["family"],
            version=row["version"],
            parent_id=row["parent_id"],
            spec_version=row["spec_version"],
            description=row["description"],
            spec_hash=row["spec_hash"],
            spec=spec,
            embedding=tuple(float(value) for value in embedding),
        )

    # -- authoring (the WritableCatalog interface) -------------------------------------------------

    def upsert(self, spec: ScenarioSpec) -> CatalogEntry:
        """Index ``spec``, deriving its family/version and linking it to the previous version.

        Idempotent on ``scenario_id``. Re-indexing a scenario whose document changed is *allowed
        here* but is not how the zoo evolves: bench.md §5 says a fix is a **new version**, and the
        lineage this catalog derives is what makes that visible.
        """
        entry = catalog_entry(spec, parent_id=self._previous_version(spec.scenario_id))
        with self._engine.begin() as connection:
            connection.execute(
                delete(_SCENARIOS).where(_SCENARIOS.c.scenario_id == entry.scenario_id)
            )
            connection.execute(
                insert(_SCENARIOS).values(
                    scenario_id=entry.scenario_id,
                    family=entry.family,
                    version=entry.version,
                    parent_id=entry.parent_id,
                    name=entry.name,
                    spec_version=entry.spec_version,
                    description=entry.description,
                    spec_hash=entry.spec_hash,
                    spec=entry.spec,
                    embedding_json=list(entry.embedding),
                )
            )
            if self._is_postgres:
                connection.execute(
                    text(
                        "UPDATE zoo_scenarios SET embedding = CAST(:vector AS vector) "
                        "WHERE scenario_id = :scenario_id"
                    ),
                    {
                        "vector": pgvector_literal(entry.embedding),
                        "scenario_id": entry.scenario_id,
                    },
                )
            # A newly-published version becomes the parent of any version that already followed it.
            connection.execute(
                _SCENARIOS.update()
                .where(_SCENARIOS.c.family == entry.family)
                .where(_SCENARIOS.c.version == entry.version + 1)
                .values(parent_id=entry.scenario_id)
            )
        return entry

    def _previous_version(self, scenario_id: str) -> str | None:
        """The newest version of the scenario's family below it — the lineage edge (bench.md §5)."""
        from astro_mine.bench.zoo._catalog import _parse_id

        family, version = _parse_id(scenario_id)
        statement = (
            select(_SCENARIOS.c.scenario_id)
            .where(_SCENARIOS.c.family == family)
            .where(_SCENARIOS.c.version < version)
            .order_by(_SCENARIOS.c.version.desc())
            .limit(1)
        )
        with self._engine.connect() as connection:
            row = connection.execute(statement).fetchone()
        return None if row is None else str(row[0])

    def seed_from(self, source: ScenarioCatalog) -> tuple[CatalogEntry, ...]:
        """Populate the catalog from ``source`` — the migration off the filesystem scan (AC4).

        Seeds in version order within each family, so every lineage edge is linked as it lands.
        Idempotent: re-seeding an already-populated catalog produces the identical rows, because the
        specs are immutable and the embedding is deterministic.
        """
        specs = [source.load_scenario(scenario_id) for scenario_id in source.list_scenarios()]

        from astro_mine.bench.zoo._catalog import _parse_id

        specs.sort(key=lambda spec: _parse_id(spec.scenario_id))
        return tuple(self.upsert(spec) for spec in specs)

    def lineage(self, scenario_id: str) -> tuple[CatalogEntry, ...]:
        """Every version of the scenario's family up to and including it, oldest first."""
        from astro_mine.bench.zoo._catalog import _parse_id

        family, version = _parse_id(scenario_id)
        statement = (
            select(_SCENARIOS)
            .where(_SCENARIOS.c.family == family)
            .where(_SCENARIOS.c.version <= version)
            .order_by(_SCENARIOS.c.version)
        )
        with self._engine.connect() as connection:
            rows = connection.execute(statement).mappings().fetchall()
        return tuple(self._to_entry(row) for row in rows)

    # -- similarity search (bench#33 AC2) ----------------------------------------------------------

    def search(self, query: str | Sequence[float], *, limit: int = 5) -> list[SearchHit]:
        """Rank the catalog by cosine similarity to ``query`` (free text, or an embedding).

        On **Postgres** the ranking is pushed down to pgvector's ``<=>`` cosine operator over the
        ``vector`` column — the deployment path bench.md §5 specifies. On SQLite the same cosine is
        computed in Python over the JSON embeddings. Both are exact, so they answer identically.
        """
        vector = embed_text(query) if isinstance(query, str) else tuple(float(v) for v in query)
        if len(vector) != EMBEDDING_DIM:
            raise ValueError(f"query embedding must have width {EMBEDDING_DIM}, got {len(vector)}")
        if self._is_postgres:
            return self._search_pgvector(vector, limit=limit)
        return self._search_python(vector, limit=limit)

    def _search_pgvector(self, vector: Sequence[float], *, limit: int) -> list[SearchHit]:
        """pgvector cosine search — an exact ``ORDER BY embedding <=> :vector`` over the column.

        Exact, not approximate: see :meth:`_create_schema`. That is what lets this rank identically
        to the SQLite backend for any catalog, which the tests assert.
        """
        statement = text(
            "SELECT scenario_id, family, version, parent_id, name, spec_version, description, "
            "       spec_hash, spec, embedding_json, "
            "       (embedding <=> CAST(:vector AS vector)) AS distance "
            "FROM zoo_scenarios "
            "WHERE embedding IS NOT NULL "
            "ORDER BY embedding <=> CAST(:vector AS vector) "
            "LIMIT :limit"
        )
        with self._engine.connect() as connection:
            rows = (
                connection.execute(
                    statement, {"vector": pgvector_literal(vector), "limit": max(0, limit)}
                )
                .mappings()
                .fetchall()
            )
        return [
            SearchHit(entry=self._to_entry(row), distance=float(row["distance"])) for row in rows
        ]

    def _search_python(self, vector: Sequence[float], *, limit: int) -> list[SearchHit]:
        """The SQLite path: the same cosine distance, computed over the JSON embeddings."""
        hits = [
            SearchHit(entry=entry, distance=cosine_distance(entry.embedding, vector))
            for entry in self.entries()
            if entry.embedding
        ]
        hits.sort(key=lambda hit: (hit.distance, hit.entry.scenario_id))
        return hits[: max(0, limit)]

    def dispose(self) -> None:
        """Close the engine's pooled connections (production keeps one long-lived engine)."""
        self._engine.dispose()
