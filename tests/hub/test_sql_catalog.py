"""The SqlCatalog's queryable projection + SQL-side search (RM-P1-HUB-02; hub.md §5, §8, §11).

The catalog row is no longer an opaque JSON blob: facets are **columns**, capability tags are a
**table**, full text is a **token document**, and the embedding is a vector column — so a filtered
or semantic query runs in the database rather than ``catalog.all()`` + a Python scan. These tests
run on SQLite (the tier-1 fallback, which must keep working **without** pgvector); the PostgreSQL +
pgvector path — the ``vector(dim)`` column, its HNSW index, and the ``ORDER BY embedding <=> q``
top-k at 10^4 scale — is proven in ``tests/integration/test_postgres.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, select

from astro_mine.core.registry import PluginKind
from astro_mine.hub.index import CatalogEntry, ingest, sql_catalog
from astro_mine.hub.index._sql import SqlCatalog, vector_type
from astro_mine.hub.search import HashingEmbedding, SearchQuery, SqlSearchable, search

from .conftest import make_manifest


@pytest.fixture
def catalog(tmp_path: Path) -> SqlCatalog:
    return SqlCatalog(f"sqlite:///{tmp_path / 'hub.db'}")


def _seed(catalog: SqlCatalog) -> None:
    ingest(
        catalog,
        make_manifest(
            "excavator",
            "1.0.0",
            kind=PluginKind.POLICY,
            description="lunar excavation digging policy",
            tags=["mobility.wheeled"],
        ),
        digest="sha256:" + "a" * 64,
        publisher="alice",
    )
    ingest(
        catalog,
        make_manifest(
            "orbiter",
            "1.0.0",
            kind=PluginKind.WORLD_PROVIDER,
            description="orbital imaging world",
            tags=["mobility.orbiter", "operational_targeting"],
            interfaces={"world_provider": "0.1.0"},
        ),
        digest="sha256:" + "b" * 64,
        publisher="bob",
        namespace="curated",
    )


def _names(catalog: SqlCatalog, query: SearchQuery) -> list[str]:
    return [result.entry.name for result in search(catalog, query)]


def test_sql_catalog_is_the_query_engine(catalog: SqlCatalog) -> None:
    """`search()` must delegate to SQL — the whole point is not scanning the catalog in Python."""
    assert isinstance(catalog, SqlSearchable)


def test_facets_are_queryable_columns(catalog: SqlCatalog) -> None:
    _seed(catalog)
    columns = catalog._catalog.c
    with catalog._engine.connect() as connection:
        row = connection.execute(
            select(
                columns.name,
                columns.version,
                columns.kind,
                columns.namespace,
                columns.publisher,
                columns.license,
                columns.deprecated,
                columns.yanked,
                columns.downloads,
                columns.embedding_provider,
            ).where(columns.reference == "orbiter:1.0.0")
        ).one()
    assert row.name == "orbiter"
    assert row.version == "1.0.0"
    assert row.kind == "world_provider"
    assert row.namespace == "curated"
    assert row.publisher == "bob"
    assert row.license == "Apache-2.0"
    assert (row.deprecated, row.yanked, row.downloads) == (False, False, 0)
    assert row.embedding_provider == "hashing-64"


def test_capability_tags_are_a_queryable_table(catalog: SqlCatalog) -> None:
    _seed(catalog)
    with catalog._engine.connect() as connection:
        rows = connection.execute(
            select(catalog._tags.c.reference, catalog._tags.c.tag).order_by(catalog._tags.c.tag)
        ).all()
    assert [(r.reference, r.tag) for r in rows] == [
        ("orbiter:1.0.0", "mobility.orbiter"),
        ("excavator:1.0.0", "mobility.wheeled"),
        ("orbiter:1.0.0", "operational_targeting"),
    ]


def test_faceted_full_text_and_tag_search_in_sql(catalog: SqlCatalog) -> None:
    _seed(catalog)
    assert _names(catalog, SearchQuery(kind="policy")) == ["excavator"]
    assert _names(catalog, SearchQuery(namespace="curated")) == ["orbiter"]
    assert _names(catalog, SearchQuery(license="MIT")) == []
    assert _names(catalog, SearchQuery(text="digging")) == ["excavator"]
    assert _names(catalog, SearchQuery(text="nonexistentterm")) == []
    assert _names(catalog, SearchQuery(capability_tags=["mobility.orbiter"])) == ["orbiter"]
    assert _names(catalog, SearchQuery(capability_tags=["mobility.orbiter", "nope"])) == []


def test_full_text_matches_whole_tokens_not_substrings(catalog: SqlCatalog) -> None:
    """The padded token document keeps SQL `LIKE` from degenerating into substring matching."""
    _seed(catalog)
    assert _names(catalog, SearchQuery(text="digging")) == ["excavator"]
    assert _names(catalog, SearchQuery(text="dig")) == []  # 'dig' is not a token of 'digging'


def test_capability_negotiation_still_applies_after_the_sql_filter(catalog: SqlCatalog) -> None:
    _seed(catalog)
    assert _names(catalog, SearchQuery(interfaces={"policy": "0.1.0"})) == ["excavator"]
    assert _names(catalog, SearchQuery(interfaces={"policy": "0.2.0"})) == []  # incompatible minor


def test_deprecated_and_yanked_are_excluded_in_sql(catalog: SqlCatalog) -> None:
    _seed(catalog)
    entry = catalog.get("excavator:1.0.0")
    assert entry is not None
    entry.yanked = True
    catalog.add(entry)  # re-add replaces the row (and its tags)

    assert _names(catalog, SearchQuery()) == ["orbiter"]
    assert sorted(_names(catalog, SearchQuery(include_yanked=True))) == ["excavator", "orbiter"]


def test_semantic_ranking_over_the_sql_path(catalog: SqlCatalog) -> None:
    _seed(catalog)
    results = search(catalog, SearchQuery(semantic="excavation digging"))
    assert results[0].entry.name == "excavator"
    assert results[0].score > 0.0


def test_re_add_replaces_the_row_and_its_tags(catalog: SqlCatalog) -> None:
    _seed(catalog)
    entry = catalog.get("orbiter:1.0.0")
    assert entry is not None
    entry.namespace = "open"
    catalog.add(entry)

    with catalog._engine.connect() as connection:
        rows = connection.execute(select(func.count()).select_from(catalog._catalog)).scalar_one()
        tags = connection.execute(select(func.count()).select_from(catalog._tags)).scalar_one()
    assert rows == 2  # not 3 — the re-add replaced, it did not duplicate
    assert tags == 3
    assert _names(catalog, SearchQuery(namespace="curated")) == []


def test_a_lexical_query_is_ranked_before_it_is_truncated(tmp_path: Path) -> None:
    """The SQL LIMIT must not be pushed down ahead of a ranking the caller has not run yet.

    Pushing `LIMIT k` under a *lexical* query would truncate by `reference` order and hand the
    ranker an arbitrary slice — the best match could be dropped before it was scored. Here the only
    strong match sorts LAST by reference, so a premature LIMIT would lose it.
    """
    catalog = SqlCatalog(f"sqlite:///{tmp_path / 'rank.db'}")
    for index in range(20):
        ingest(
            catalog,
            make_manifest(
                f"pol{index:02d}",
                "1.0.0",
                # only the last entry (in reference order) is a strong lexical match
                description="lunar excavation digging" if index == 19 else "generic policy",
            ),
            digest=f"sha256:{index:064x}",
            publisher="p",
        )

    hits = search(catalog, SearchQuery(text="excavation", limit=3))
    assert [hit.entry.name for hit in hits] == ["pol19"]  # ranked across ALL matches, then cut


def test_a_facet_browse_is_limited_in_sql(tmp_path: Path) -> None:
    """With nothing to rank by, SQL's `reference` order IS the final order — so the LIMIT lands."""
    catalog = SqlCatalog(f"sqlite:///{tmp_path / 'browse.db'}")
    for index in range(20):
        ingest(
            catalog,
            make_manifest(f"pol{index:02d}", "1.0.0"),
            digest=f"sha256:{index:064x}",
            publisher="p",
        )

    # the backend applies the limit itself — only `limit` rows are materialized, not all 20
    entries = catalog.search_entries(SearchQuery(limit=5), limit=5)
    assert [entry.name for entry in entries] == ["pol00", "pol01", "pol02", "pol03", "pol04"]
    assert [hit.entry.name for hit in search(catalog, SearchQuery(limit=5))] == [
        entry.name for entry in entries
    ]


def test_embedding_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    """A vector of the wrong width would silently corrupt a pgvector column — refuse it up front."""
    catalog = SqlCatalog(f"sqlite:///{tmp_path / 'hub.db'}", embedding_dim=8)
    entry = CatalogEntry(
        manifest=make_manifest(),
        digest="sha256:" + "a" * 64,
        publisher="p",
        embedding=HashingEmbedding(64).embed("x"),
        embedding_provider="hashing-64",
    )
    with pytest.raises(ValueError, match="64-d embedding"):
        catalog.add(entry)


def test_sqlite_needs_no_pgvector(catalog: SqlCatalog) -> None:
    """The tier-1 fallback: no vector column, no extension, and search still works (principle 9)."""
    assert catalog._vector is False
    _seed(catalog)
    assert search(catalog, SearchQuery(semantic="excavation"))[0].entry.name == "excavator"


def test_vector_type_is_available_for_postgres() -> None:
    """pgvector ships in the [service] extra: the vector(dim) column type must be constructible."""
    column = vector_type(64)
    assert column is not None
    assert column.dim == 64  # type: ignore[attr-defined]


def test_sql_catalog_needs_a_url_or_engine() -> None:
    with pytest.raises(ValueError, match="needs a database url or an engine"):
        SqlCatalog()


def test_sql_catalog_accepts_an_injected_engine(tmp_path: Path) -> None:
    engine = create_engine(f"sqlite:///{tmp_path / 'x.db'}")
    catalog = SqlCatalog(engine=engine)
    _seed(catalog)
    assert len(catalog.all()) == 2
    catalog.dispose()


def test_sql_catalog_factory(tmp_path: Path) -> None:
    catalog = sql_catalog(f"sqlite:///{tmp_path / 'f.db'}")
    ingest(catalog, make_manifest(), digest="sha256:" + "a" * 64, publisher="p")
    assert catalog.get("pol:1.0.0") is not None


def test_search_at_scale_runs_in_sql(tmp_path: Path) -> None:
    """A synthetic catalog large enough that a Python scan would be the wrong implementation.

    The offline gate keeps this at 10^3 to stay fast; the 10^4 pgvector run is the Postgres
    integration test (hub.md §8 targets 10^5-10^6).
    """
    catalog = SqlCatalog(f"sqlite:///{tmp_path / 'scale.db'}")
    for index in range(1000):
        ingest(
            catalog,
            make_manifest(
                f"pol{index:04d}",
                "1.0.0",
                description="excavation digging" if index == 500 else "generic filler policy",
                tags=["mobility.wheeled"] if index % 2 == 0 else [],
            ),
            digest=f"sha256:{index:064x}",
            publisher="alice",
        )

    hits = search(catalog, SearchQuery(text="excavation", limit=5))
    assert [hit.entry.name for hit in hits] == ["pol0500"]

    tagged = catalog.search_entries(SearchQuery(capability_tags=["mobility.wheeled"]))
    assert len(tagged) == 500  # the tag filter ran as a SQL EXISTS, not a 1000-entry Python scan

    ranked = search(catalog, SearchQuery(semantic="excavation digging", limit=3))
    assert ranked[0].entry.name == "pol0500"
    assert len(ranked) == 3
