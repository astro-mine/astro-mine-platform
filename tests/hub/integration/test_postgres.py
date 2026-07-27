"""Integration: the SqlCatalog on real PostgreSQL + pgvector (RM-P1-HUB-02; hub.md §5, §8, §11).

The offline suite runs the same SQLAlchemy code path on SQLite; what only a real PostgreSQL can
prove is the part hub.md §11 actually specifies — a **pgvector** ``vector(dim)`` column with an
**HNSW** index, and semantic search executing as a **SQL vector query** (``ORDER BY embedding <=> q
LIMIT k``) rather than ``catalog.all()`` + a Python scan. §8 sets the bar at "sub-second faceted +
full-text + top-k semantic queries at catalog sizes of 10^5-10^6", so the scale check seeds a
synthetic catalog in the 10^4 band and asserts both correctness and a latency budget.

Runs in the `integration` CI job (a `pgvector/pgvector:pg16` service) or locally when
``HUB_POSTGRES_URL`` is set (`docker compose up -d`); skipped otherwise.
"""

from __future__ import annotations

import os
import random
import time
from collections.abc import Iterator

import pytest
from sqlalchemy import func, select, text

from astro_mine.core.registry import PluginManifest
from astro_mine.hub.index import index_document, ingest
from astro_mine.hub.index._sql import _HNSW_EF_SEARCH, SqlCatalog
from astro_mine.hub.search import HashingEmbedding, SearchQuery, cosine, search

from ..conftest import make_manifest

pytestmark = pytest.mark.integration

_POSTGRES_URL = os.environ.get("HUB_POSTGRES_URL")

#: Big enough that a Python scan is the wrong implementation, small enough to seed in CI.
SCALE = int(os.environ.get("HUB_SCALE_ENTRIES", "10000"))

#: The corpus is generated from a fixed seed — conventions.md §11 ("seeded runs … CI fails on
#: nondeterminism"). Its predecessor drew a fresh ``uuid4()`` per run, so a failure described a
#: corpus that no longer existed and could not be reproduced.
SEED = 20260712

#: Recall is a *statistical* property of an approximate index, so one probe per run measures it to
#: ±1 sample. Probing many documents makes a single run a real measurement (see the scale test).
PROBES = 20

requires_postgres = pytest.mark.skipif(not _POSTGRES_URL, reason="HUB_POSTGRES_URL not set")

#: The synthetic catalog's descriptions: one word from each axis, so every artifact is *about*
#: something and no two say the same thing. The axes are disjoint (a token belongs to exactly one),
#: so a document's token set — and hence its vector — identifies it.
#:
#: Vocabulary breadth is load-bearing, not decoration. Three words drawn from a single 20-word list
#: (this corpus's predecessor) collapses 10^4 documents onto a handful of exactly-tied vectors and
#: quantizes every cosine onto a few discrete levels — thousands of documents at *identical*
#: distance from a query. An HNSW walk has no gradient to descend on a landscape like that, and
#: whether it ever reaches the true nearest neighbour is a coin flip on the graph's shape. That is
#: the whole story of issue #27. A real commons does not look like that, and neither does this.
_AXES: tuple[list[str], ...] = tuple(
    axis.split()
    for axis in (
        "excavation hauling drilling survey mapping prospecting sampling grading trenching coring",
        "regolith ice basalt ilmenite volatiles dust brine permafrost ore tailings",
        "crater rille plain ridge slope pit rim dune canyon plateau",
        "rover lander orbiter hopper excavator hauler auger smelter cryoplant beacon",
        "thermal power comms navigation autonomy avionics guidance storage cooling telemetry",
        "night dawn dusk daylight eclipse traverse descent ascent standby recharge",
    )
)

#: A token no filler description contains — the full-text needle (a SQL ``LIKE``, exact by
#: construction, and the one assertion here that owes nothing to the vector index).
NEEDLE_TOKEN = "cryotrap"

NEEDLE_INDEX = SCALE // 2

NEEDLE_DESCRIPTION = f"lunar polar {NEEDLE_TOKEN} volatiles inventory prospecting ice drilling"


def _manifest(index: int, description: str) -> PluginManifest:
    return make_manifest(
        f"pol{index:06d}",
        "1.0.0",
        description=description,
        tags=["mobility.wheeled"] if index % 2 == 0 else [],
    )


def _corpus() -> list[PluginManifest]:
    """The synthetic 10^4 catalog: one planted needle, half of the fillers tagged.

    Every artifact gets a **distinct embedding**: a description whose vector collides with one
    already drawn (feature hashing has finitely many buckets, so a few of 10^4 will) is re-drawn.
    Distinctness is not fastidiousness — it is the precondition the recall probes rest on. These are
    unit vectors, so two are equal *iff* their cosine is 1.0; keeping them distinct is what makes a
    document the **unique** exact nearest neighbour of its own text, and therefore what makes "the
    index ranked something else first" mean the index failed to recall it — rather than the index
    returning an equally-good twin that the assertion had no right to reject.
    """
    provider = HashingEmbedding()
    rng = random.Random(SEED)
    manifests: list[PluginManifest] = []
    vectors: set[tuple[float, ...]] = set()

    for index in range(SCALE):
        if index == NEEDLE_INDEX:
            manifest = _manifest(index, NEEDLE_DESCRIPTION)
            vector = provider.embed(index_document(manifest))
            assert vector not in vectors, "the needle collided with a filler — pick another needle"
        else:
            while True:
                description = " ".join(rng.choice(axis) for axis in _AXES) + " policy"
                manifest = _manifest(index, description)
                vector = provider.embed(index_document(manifest))
                if vector not in vectors:
                    break
        vectors.add(vector)
        manifests.append(manifest)

    return manifests


@pytest.fixture
def catalog() -> Iterator[SqlCatalog]:
    """A catalog in its own schema, so a failed run never poisons the next one."""
    assert _POSTGRES_URL is not None
    catalog = SqlCatalog(_POSTGRES_URL)
    with catalog._engine.begin() as connection:
        connection.execute(text("TRUNCATE TABLE catalog_tags, catalog"))
    yield catalog
    catalog.dispose()


@requires_postgres
def test_sql_catalog_on_postgres(catalog: SqlCatalog) -> None:
    ingest(
        catalog,
        make_manifest("art", "1.0.0", tags=["mobility.wheeled"], description="excavation"),
        digest="sha256:" + "a" * 64,
        publisher="alice",
    )
    got = catalog.get("art:1.0.0")
    assert got is not None
    assert got.publisher == "alice"
    assert got.capability_tags == ["mobility.wheeled"]
    assert got.manifest.name == "art"  # the Core manifest round-trips through Postgres JSON
    assert [entry.reference for entry in catalog.all()] == ["art:1.0.0"]


@requires_postgres
def test_pgvector_column_and_hnsw_index_exist(catalog: SqlCatalog) -> None:
    """hub.md §11: "Postgres + pgvector … one store for catalog + facets + semantic vectors"."""
    assert catalog._vector is True

    with catalog._engine.connect() as connection:
        column_type = connection.execute(
            text(
                "SELECT udt_name FROM information_schema.columns "
                "WHERE table_name = 'catalog' AND column_name = 'embedding'"
            )
        ).scalar_one()
        index = connection.execute(
            text("SELECT indexdef FROM pg_indexes WHERE indexname = 'catalog_embedding_hnsw'")
        ).scalar_one()

    assert column_type == "vector"  # a real pgvector column, not a JSON blob
    assert "USING hnsw" in index and "vector_cosine_ops" in index


@requires_postgres
def test_facet_and_tag_columns_are_queryable_in_sql(catalog: SqlCatalog) -> None:
    ingest(
        catalog,
        make_manifest("excavator", "1.0.0", tags=["mobility.wheeled"], description="digging"),
        digest="sha256:" + "a" * 64,
        publisher="alice",
    )
    ingest(
        catalog,
        make_manifest("orbiter", "1.0.0", tags=["mobility.orbiter"], description="imaging"),
        digest="sha256:" + "b" * 64,
        publisher="bob",
        namespace="curated",
    )

    with catalog._engine.connect() as connection:
        curated = connection.execute(
            select(func.count())
            .select_from(catalog._catalog)
            .where(catalog._catalog.c.namespace == "curated")
        ).scalar_one()
    assert curated == 1

    assert [r.entry.name for r in search(catalog, SearchQuery(namespace="curated"))] == ["orbiter"]
    assert [r.entry.name for r in search(catalog, SearchQuery(text="digging"))] == ["excavator"]
    assert [
        r.entry.name for r in search(catalog, SearchQuery(capability_tags=["mobility.orbiter"]))
    ] == ["orbiter"]


@requires_postgres
def test_semantic_search_runs_as_a_sql_vector_query(catalog: SqlCatalog) -> None:
    """The top-k is pruned by pgvector in the database — `catalog.all()` is never called."""
    for index, description in enumerate(["lunar excavation digging", "orbital imaging", "comms"]):
        ingest(
            catalog,
            make_manifest(f"art{index}", "1.0.0", description=description),
            digest=f"sha256:{index:064x}",
            publisher="p",
        )

    provider = HashingEmbedding()
    entries = catalog.search_entries(
        SearchQuery(semantic="excavation digging"),
        query_vector=provider.embed("excavation digging"),
        limit=1,
    )
    assert [entry.name for entry in entries] == ["art0"]  # SQL LIMIT 1 — ordered by the vector

    results = search(catalog, SearchQuery(semantic="excavation digging", limit=2))
    assert results[0].entry.name == "art0"
    assert results[0].score > 0.0


@requires_postgres
def test_search_at_representative_scale(catalog: SqlCatalog) -> None:
    """A synthetic catalog in the 10^4 band: correctness *and* the §8 sub-second latency target.

    What is under test is the **SQL path** — that facets, full text, tags, and the pgvector top-k
    return the right rows, fast, at scale. Each layer is asserted for what it actually owes:

    - facets, tags and full text are **exact** — they are a SQL ``WHERE``, and an index that dropped
      a row would be a bug, not a tuning parameter;
    - the vector top-k is **approximate**, so its recall is measured over :data:`PROBES` documents
      rather than inferred from one. A single probe is a one-sample estimate of a statistical
      property, which is precisely how the predecessor of this test managed to fail one run in three
      while looking, each time, like a fresh mystery (#27).

    Relevance *quality* of the offline hashing embedding is deliberately not under test: at 10^4
    documents a 64-d feature-hashing vector is a lexical stand-in, which is exactly why hub.md §11
    wants a learned model on the hosted tier. What the index owes is that it finds what it stored.
    """
    manifests = _corpus()
    for index, manifest in enumerate(manifests):
        ingest(catalog, manifest, digest=f"sha256:{index:064x}", publisher="alice")

    with catalog._engine.connect() as connection:
        count = connection.execute(select(func.count()).select_from(catalog._catalog)).scalar_one()
    assert count == SCALE

    entries = catalog.all()
    needle_reference = f"pol{NEEDLE_INDEX:06d}:1.0.0"
    provider = HashingEmbedding()

    # The corpus is in general position: no two artifacts share a vector. This is a property of the
    # *fixture*, not of the catalog — but it is the one the old corpus violated, and asserting it
    # here is what stops a future edit to `_AXES` from quietly reintroducing the degenerate cluster
    # that made this test a coin flip.
    assert len({entry.embedding for entry in entries}) == SCALE

    # 1. Full text — exact, deterministic, and answered by a SQL LIKE on the token document.
    started = time.perf_counter()
    hits = search(catalog, SearchQuery(text=NEEDLE_TOKEN, limit=10))
    full_text_s = time.perf_counter() - started
    assert [hit.entry.reference for hit in hits] == [needle_reference]

    # 2. Semantic — query with the needle's own indexed document. Its cosine with itself is 1.0, the
    #    maximum, so if the index returns it at all the ranker puts it first.
    query_text = index_document(manifests[NEEDLE_INDEX])
    started = time.perf_counter()
    ranked = search(catalog, SearchQuery(semantic=query_text, limit=10), provider=provider)
    semantic_s = time.perf_counter() - started

    assert len(ranked) == 10  # the SQL LIMIT pruned the top-k; nothing was scanned in Python
    assert ranked[0].entry.reference == needle_reference
    assert ranked[0].score == pytest.approx(1.0)

    # …and the approximate top-k is as good as the exhaustive one. Quality is compared by SCORE, not
    # by identity: below the needle the corpus holds many near-equal neighbours, and an identity
    # metric would punish the index for returning a *different but equally good* document.
    query_vector = provider.embed(query_text)
    exact_scores = sorted(
        (cosine(query_vector, entry.embedding) for entry in entries), reverse=True
    )[:10]
    returned = [hit.score for hit in ranked]
    quality = sum(returned) / sum(exact_scores)
    assert quality >= 0.95, (
        f"HNSW top-10 quality was {quality:.3f} of the exhaustive top-10 at {SCALE} entries "
        f"(returned {returned[:3]}…, exact {exact_scores[:3]}…)"
    )

    # 3. Recall, measured — every probe queries with a document's own text, so the document itself
    #    is the exact nearest neighbour (cosine 1.0) and the index owes us its return. Probing many
    #    documents turns one run into a real sample: a 19/20 here is a *finding* (hub.md §8's
    #    "if recall … regresses"), not a re-run.
    stride = SCALE // PROBES
    missed = [
        entry.reference
        for entry in (entries[index * stride] for index in range(PROBES))
        if entry.reference
        not in {
            hit.entry.reference
            for hit in search(
                catalog,
                SearchQuery(semantic=index_document(entry.manifest), limit=10),
                provider=provider,
            )
        }
    ]
    assert not missed, (
        f"pgvector's HNSW top-10 missed the exact match for {len(missed)}/{PROBES} probes at "
        f"{SCALE} entries (hnsw.ef_search={_HNSW_EF_SEARCH}): {missed}"
    )

    # 4. Faceted + capability tag — a SQL EXISTS against the indexed tag table.
    started = time.perf_counter()
    tagged = search(catalog, SearchQuery(capability_tags=["mobility.wheeled"], limit=10))
    faceted_s = time.perf_counter() - started
    assert len(tagged) == 10
    assert all("mobility.wheeled" in hit.entry.capability_tags for hit in tagged)

    # hub.md §8: sub-second faceted + full-text + top-k semantic at catalog scale.
    assert full_text_s < 1.0, f"full-text took {full_text_s:.3f}s at {SCALE} entries"
    assert semantic_s < 1.0, f"semantic top-k took {semantic_s:.3f}s at {SCALE} entries"
    assert faceted_s < 1.0, f"faceted+tag took {faceted_s:.3f}s at {SCALE} entries"
