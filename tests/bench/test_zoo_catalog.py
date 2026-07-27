"""The Postgres/pgvector scenario-zoo catalog (bench#33; bench.md §5).

bench.md §5's data architecture puts the scenario-zoo catalog in **PostgreSQL** — *"Indexes specs,
versions, lineage; **pgvector** for similarity/search"* — while the implementation scanned the
installed package for ``scenario.json`` files. That is fine at three scenarios and a real deviation
at a community zoo's worth. These tests cover the fix, against the acceptance criteria:

1. a **Postgres-backed catalog behind the existing discovery interface**, indexing spec, version,
   and lineage;
2. **pgvector similarity search** when the Postgres backend is active;
3. the **filesystem scan stays the default** when no Postgres is configured — the local, offline,
   account-free tier is untouched (CX-LOCAL);
4. a **migration/seed utility** populates the catalog from the existing zoo's ``scenario.json``.

The catalog is one SQLAlchemy code path over two dialects (the house pattern set by ``SqlStore``):
every test here runs on **SQLite**, so the whole path — schema, upsert, lineage, seeding, and cosine
ranking — is verified with no database server. The ``postgres``-marked tests additionally drive the
**real pgvector** path (``CREATE EXTENSION vector``, a ``vector(256)`` column, an IVFFlat index, and
the ``<=>`` operator); they run in CI against the ``pgvector/pgvector`` service and skip on a laptop
where ``$ASTRO_MINE_BENCH_TEST_DSN`` is unset.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from astro_mine.bench.zoo import (
    ANCHOR_SCENARIO_ID,
    CATALOG_DSN_ENV,
    EMBEDDING_DIM,
    CatalogEntry,
    FilesystemCatalog,
    ScenarioCatalog,
    WritableCatalog,
    catalog_entry,
    default_catalog,
    embed_scenario,
    embed_text,
    list_scenarios,
    load_scenario,
    open_sql_catalog,
)
from astro_mine.bench.zoo._embed import cosine_distance, scenario_tokens
from astro_mine.bench.zoo._sql import SqlCatalog, pgvector_literal
from tests.bench._factories import make_scenario_spec

#: A live Postgres+pgvector DSN. Set by CI (the pgvector service container); unset on a laptop,
#: the pgvector-specific tests skip and the identical logic is covered on SQLite.
TEST_DSN = os.environ.get("ASTRO_MINE_BENCH_TEST_DSN")

needs_pgvector = pytest.mark.skipif(
    not TEST_DSN, reason="needs a live PostgreSQL with pgvector ($ASTRO_MINE_BENCH_TEST_DSN)"
)


@pytest.fixture
def catalog(tmp_path: Path) -> Iterator[SqlCatalog]:
    """The catalog on SQLite — the same code the deployment runs on Postgres."""
    built = SqlCatalog(f"sqlite:///{tmp_path / 'zoo.db'}")
    yield built
    built.dispose()


@pytest.fixture
def pg_catalog() -> Iterator[SqlCatalog]:
    """The catalog on a **real** PostgreSQL + pgvector — the deployment path (bench.md §5)."""
    assert TEST_DSN
    built = SqlCatalog(TEST_DSN)
    with built._engine.begin() as connection:
        from sqlalchemy import text

        connection.execute(text("TRUNCATE TABLE zoo_scenarios"))
    yield built
    built.dispose()


# =================================================================================================
# AC3 — the filesystem scan stays the tier-1 default (checked FIRST: it is the one that must not
# regress; the local tier is sacred)
# =================================================================================================


def test_the_default_catalog_is_the_filesystem_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """No DSN configured ⇒ no database anywhere near the local tier (CX-LOCAL)."""
    monkeypatch.delenv(CATALOG_DSN_ENV, raising=False)
    assert isinstance(default_catalog(), FilesystemCatalog)


def test_the_local_scoring_path_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """`load_scenario` / `list_scenarios` still work offline, with no Postgres and no account."""
    monkeypatch.delenv(CATALOG_DSN_ENV, raising=False)
    assert ANCHOR_SCENARIO_ID in list_scenarios()
    assert load_scenario(ANCHOR_SCENARIO_ID).scenario_id == ANCHOR_SCENARIO_ID


def test_the_dsn_selects_the_sql_catalog(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(CATALOG_DSN_ENV, f"sqlite:///{tmp_path / 'selected.db'}")
    selected = default_catalog()
    assert isinstance(selected, SqlCatalog)


def test_the_filesystem_catalog_is_not_writable() -> None:
    """The packaged zoo ships in the wheel; mutating it at runtime would break immutability."""
    assert isinstance(FilesystemCatalog(), ScenarioCatalog)
    assert not isinstance(FilesystemCatalog(), WritableCatalog)


def test_both_backends_satisfy_the_discovery_interface(catalog: SqlCatalog) -> None:
    """AC1: the Postgres catalog sits *behind the existing discovery interface*."""
    assert isinstance(FilesystemCatalog(), ScenarioCatalog)
    assert isinstance(catalog, ScenarioCatalog)
    assert isinstance(catalog, WritableCatalog)  # ...and adds the authoring surface


# =================================================================================================
# AC1 — the index: spec, version, lineage
# =================================================================================================


def test_an_entry_indexes_the_spec_its_version_and_its_hash() -> None:
    entry = catalog_entry(load_scenario(ANCHOR_SCENARIO_ID))
    assert entry.scenario_id == ANCHOR_SCENARIO_ID
    assert entry.family == "lunar-polar-ice-prospecting"
    assert entry.version == 1
    assert entry.spec_hash.startswith("sha256:")
    assert entry.to_spec().spec_hash == entry.spec_hash  # the catalog *serves*, not just indexes
    assert len(entry.embedding) == EMBEDDING_DIM


def test_an_unversioned_id_is_version_one() -> None:
    entry = catalog_entry(make_scenario_spec(scenario_id="no-version-suffix"))
    assert entry.family == "no-version-suffix"
    assert entry.version == 1


def test_lineage_is_derived_from_the_immutable_versioning_rule(catalog: SqlCatalog) -> None:
    """bench.md §5: 'a fix is a new version'. So v2 descends from v1 — and the catalog knows it."""
    for version in (1, 2, 3):
        catalog.upsert(make_scenario_spec(scenario_id=f"ice-v{version}", name=f"Ice v{version}"))

    lineage = catalog.lineage("ice-v3")
    assert [entry.scenario_id for entry in lineage] == ["ice-v1", "ice-v2", "ice-v3"]
    assert [entry.parent_id for entry in lineage] == [None, "ice-v1", "ice-v2"]
    # A middle version's lineage stops at itself — history, not the future.
    assert [e.scenario_id for e in catalog.lineage("ice-v2")] == ["ice-v1", "ice-v2"]


def test_a_late_arriving_earlier_version_relinks_its_successor(catalog: SqlCatalog) -> None:
    """Seeding order must not corrupt the lineage: publishing v1 after v2 still links them."""
    catalog.upsert(make_scenario_spec(scenario_id="ice-v2"))
    assert catalog.lineage("ice-v2")[-1].parent_id is None

    catalog.upsert(make_scenario_spec(scenario_id="ice-v1"))
    relinked = {e.scenario_id: e.parent_id for e in catalog.entries()}
    assert relinked == {"ice-v1": None, "ice-v2": "ice-v1"}


def test_upsert_is_idempotent(catalog: SqlCatalog) -> None:
    spec = load_scenario(ANCHOR_SCENARIO_ID)
    first = catalog.upsert(spec)
    again = catalog.upsert(spec)
    assert first == again
    assert catalog.list_scenarios() == (ANCHOR_SCENARIO_ID,)  # not duplicated


def test_the_catalog_serves_a_scenario_by_id(catalog: SqlCatalog) -> None:
    catalog.upsert(load_scenario(ANCHOR_SCENARIO_ID))
    served = catalog.load_scenario(ANCHOR_SCENARIO_ID)
    assert served.spec_hash == load_scenario(ANCHOR_SCENARIO_ID).spec_hash  # byte-for-byte the same


def test_an_unknown_scenario_raises_keyerror(catalog: SqlCatalog) -> None:
    with pytest.raises(KeyError, match="no zoo scenario"):
        catalog.load_scenario("not-in-the-zoo-v1")


def test_sql_catalog_requires_a_url_or_engine() -> None:
    with pytest.raises(ValueError, match="url or an engine"):
        SqlCatalog()


# =================================================================================================
# AC4 — the migration/seed utility
# =================================================================================================


def test_seed_from_populates_the_catalog_from_the_packaged_zoo(catalog: SqlCatalog) -> None:
    """AC4: the migration path off the filesystem scan."""
    source = FilesystemCatalog()
    seeded = catalog.seed_from(source)

    assert {entry.scenario_id for entry in seeded} == set(source.list_scenarios())
    assert catalog.list_scenarios() == source.list_scenarios()
    # Every seeded spec round-trips byte-for-byte.
    for scenario_id in source.list_scenarios():
        assert (
            catalog.load_scenario(scenario_id).spec_hash
            == source.load_scenario(scenario_id).spec_hash
        )


def test_seeding_is_idempotent_and_deterministic(catalog: SqlCatalog, tmp_path: Path) -> None:
    """Re-seeding gives identical rows — the specs are immutable and the embedding is deterministic.

    This matters more than it looks: a catalog that drifted between seedings would make a hosted
    leaderboard's scenario lineage irreproducible (CX-REPRO).
    """
    first = catalog.seed_from(FilesystemCatalog())
    second = catalog.seed_from(FilesystemCatalog())
    assert first == second

    other = SqlCatalog(f"sqlite:///{tmp_path / 'other.db'}")
    assert other.seed_from(FilesystemCatalog()) == first  # ...and on a different database, too
    other.dispose()


def test_filesystem_entries_carry_their_lineage() -> None:
    entries = FilesystemCatalog().entries()
    assert {entry.scenario_id for entry in entries} == set(FilesystemCatalog().list_scenarios())
    assert all(isinstance(entry, CatalogEntry) for entry in entries)


# =================================================================================================
# AC2 — pgvector similarity search
# =================================================================================================


def test_the_embedding_is_deterministic_and_normalized() -> None:
    spec = load_scenario(ANCHOR_SCENARIO_ID)
    first, second = embed_scenario(spec), embed_scenario(spec)
    assert first == second  # same spec ⇒ same vector, on any machine, in any process
    assert len(first) == EMBEDDING_DIM
    assert abs(sum(value * value for value in first) ** 0.5 - 1.0) < 1e-9  # L2-normalized


def test_the_embedding_captures_meaning_not_hashes() -> None:
    """Content *hashes* carry no similarity signal: two world revisions differ in every bit."""
    tokens = scenario_tokens(load_scenario(ANCHOR_SCENARIO_ID))
    assert "ice" in tokens and "prospecting" in tokens
    assert "shackleton" in tokens  # the pinned world's *id*
    assert "water" in tokens and "mass" in tokens  # the metric names
    assert not any(len(token) == 64 for token in tokens)  # ...but never a sha256 hex digest


def test_cosine_distance_is_a_metric() -> None:
    a, b = embed_text("lunar polar ice"), embed_text("lunar polar ice")
    assert cosine_distance(a, b) == pytest.approx(0.0, abs=1e-9)  # identical ⇒ distance 0
    far = embed_text("asteroid regolith excavation throughput")
    assert cosine_distance(a, far) > cosine_distance(a, b)
    with pytest.raises(ValueError, match="width"):
        cosine_distance((1.0, 2.0), (1.0, 2.0, 3.0))


def test_cosine_distance_of_an_empty_vector() -> None:
    assert cosine_distance((0.0,) * EMBEDDING_DIM, embed_text("anything")) == 1.0
    assert embed_text("") == (0.0,) * EMBEDDING_DIM


def test_similarity_search_finds_the_relevant_scenario(catalog: SqlCatalog) -> None:
    """AC2: 'find me scenarios like this one' — what a community zoo actually needs."""
    catalog.seed_from(FilesystemCatalog())
    hits = catalog.search("lunar polar water ice prospecting", limit=3)

    assert hits, "the seeded catalog must return hits"
    assert hits[0].entry.scenario_id.startswith("lunar-polar-ice")
    # Ranked: distances ascend, and the best hit is genuinely close.
    assert [hit.distance for hit in hits] == sorted(hit.distance for hit in hits)
    assert hits[0].distance < 1.0


def test_searching_by_a_scenarios_own_embedding_finds_itself(catalog: SqlCatalog) -> None:
    catalog.seed_from(FilesystemCatalog())
    anchor = catalog.load_scenario(ANCHOR_SCENARIO_ID)
    hits = catalog.search(embed_scenario(anchor), limit=1)
    assert hits[0].entry.scenario_id == ANCHOR_SCENARIO_ID
    assert hits[0].distance == pytest.approx(0.0, abs=1e-6)  # a scenario is closest to itself


def test_search_limit_is_respected(catalog: SqlCatalog) -> None:
    catalog.seed_from(FilesystemCatalog())
    assert len(catalog.search("ice", limit=1)) == 1
    assert catalog.search("ice", limit=0) == []


def test_search_rejects_a_mis_shaped_query_vector(catalog: SqlCatalog) -> None:
    with pytest.raises(ValueError, match="width"):
        catalog.search([0.1, 0.2, 0.3])


def test_sqlite_ranks_in_python_postgres_ranks_in_pgvector(catalog: SqlCatalog) -> None:
    assert catalog.uses_pgvector is False  # SQLite: the same cosine, computed in Python


def test_pgvector_literal_renders_the_operator_argument() -> None:
    assert pgvector_literal([0.5, -0.25]) == "[0.5,-0.25]"


# --- the real pgvector path (runs in CI against the pgvector service; skipped on a laptop) --------


@pytest.mark.postgres
@needs_pgvector
def test_postgres_catalog_uses_real_pgvector(pg_catalog: SqlCatalog) -> None:
    """AC2, for real: `CREATE EXTENSION vector`, a vector(256) column, and the `<=>` operator."""
    assert pg_catalog.uses_pgvector is True
    pg_catalog.seed_from(FilesystemCatalog())

    hits = pg_catalog.search("lunar polar water ice prospecting", limit=3)
    assert hits and hits[0].entry.scenario_id.startswith("lunar-polar-ice")
    assert [hit.distance for hit in hits] == sorted(hit.distance for hit in hits)


@pytest.mark.postgres
@needs_pgvector
def test_pgvector_search_finds_rows_seeded_after_the_schema_was_created(
    pg_catalog: SqlCatalog,
) -> None:
    """A catalog is *always* schema-first, rows-second. Search must still find them.

    Regression: this class used to create an **IVFFlat** index in `_create_schema`. IVFFlat derives
    its centroids from the rows present when the index is built — and a schema is created on an
    *empty* table — so every `<=>` query through it returned **zero rows**, silently. Caught by CI
    against a real pgvector server; it would otherwise have shipped as "search finds nothing".
    """
    # The fixture has already created the schema on an empty table. Now seed, then search.
    assert pg_catalog.search("anything at all", limit=5) == []  # genuinely empty, not broken
    pg_catalog.seed_from(FilesystemCatalog())

    hits = pg_catalog.search("ice", limit=10)
    assert len(hits) == len(FilesystemCatalog().list_scenarios())  # every seeded row is reachable


@pytest.mark.postgres
@needs_pgvector
def test_postgres_and_sqlite_rank_identically(pg_catalog: SqlCatalog, catalog: SqlCatalog) -> None:
    """The dialect changes *where* the cosine is computed, never *what it means*.

    This is the whole justification for the dual-backend design: the similarity search verified
    offline on SQLite is the similarity search that ships on pgvector.
    """
    pg_catalog.seed_from(FilesystemCatalog())
    catalog.seed_from(FilesystemCatalog())

    query = "ice prospecting endurance through the lunar night"
    postgres = [hit.entry.scenario_id for hit in pg_catalog.search(query, limit=5)]
    sqlite = [hit.entry.scenario_id for hit in catalog.search(query, limit=5)]
    assert postgres == sqlite


@pytest.mark.postgres
@needs_pgvector
def test_postgres_catalog_indexes_spec_version_and_lineage(pg_catalog: SqlCatalog) -> None:
    """AC1, for real: the index bench.md §5 specifies, on the database it specifies."""
    for version in (1, 2):
        pg_catalog.upsert(make_scenario_spec(scenario_id=f"pg-ice-v{version}"))

    lineage = pg_catalog.lineage("pg-ice-v2")
    assert [entry.scenario_id for entry in lineage] == ["pg-ice-v1", "pg-ice-v2"]
    assert lineage[-1].parent_id == "pg-ice-v1"
    served = pg_catalog.load_scenario("pg-ice-v1")
    assert served.scenario_id == "pg-ice-v1"


@pytest.mark.postgres
@needs_pgvector
def test_postgres_seeding_is_idempotent(pg_catalog: SqlCatalog) -> None:
    first = pg_catalog.seed_from(FilesystemCatalog())
    second = pg_catalog.seed_from(FilesystemCatalog())
    assert first == second
    assert len(pg_catalog.list_scenarios()) == len(FilesystemCatalog().list_scenarios())


def test_open_sql_catalog_is_a_lazy_wrapper(tmp_path: Path) -> None:
    opened = open_sql_catalog(f"sqlite:///{tmp_path / 'lazy.db'}")
    assert isinstance(opened, SqlCatalog)
    opened.dispose()
