"""Index tests (RM-P1-HUB-02): manifest projection, facets, InMemory/SQL backend equivalence."""

from __future__ import annotations

from pathlib import Path

import pytest

from astro_mine.hub.index import (
    CatalogEntry,
    InMemoryCatalog,
    index_document,
    ingest,
    sql_catalog,
)
from astro_mine.hub.index._sql import SqlCatalog

from .conftest import make_manifest

DIGEST = "sha256:" + "a" * 64


def test_ingest_builds_entry_with_facets_and_embedding() -> None:
    cat = InMemoryCatalog()
    manifest = make_manifest(
        "exc", "1.2.0", description="excavation policy", inputs=["Observation"], outputs=["Action"]
    )
    entry = ingest(cat, manifest, digest=DIGEST, publisher="alice")
    assert entry.reference == "exc:1.2.0"
    assert entry.publisher == "alice" and entry.namespace == "open"
    assert entry.kind == "policy" and entry.license == "Apache-2.0"
    assert len(entry.embedding) == 64 and any(v != 0.0 for v in entry.embedding)
    assert cat.get("exc:1.2.0") is entry
    assert entry.record.name == "exc"  # the Core CatalogRecord projection


def test_catalog_entry_properties_and_satisfies() -> None:
    manifest = make_manifest(interfaces={"policy": "0.1.0"}, tags=["mobility.wheeled"])
    entry = CatalogEntry(manifest=manifest, digest=DIGEST, publisher="p")
    assert entry.name == "pol" and entry.version == "1.0.0"
    assert entry.capability_tags == ["mobility.wheeled"]
    assert entry.satisfies(interfaces={"policy": "0.1.0"})
    assert not entry.satisfies(interfaces={"policy": "0.2.0"})  # 0.y minor must match exactly
    assert entry.satisfies(capability_tags=["mobility.wheeled"])
    assert not entry.satisfies(capability_tags=["propulsion.chemical"])


def test_inmemory_add_get_all_and_replace() -> None:
    cat = InMemoryCatalog()
    assert cat.all() == [] and cat.get("x:1.0.0") is None
    ingest(cat, make_manifest("x", "1.0.0"), digest=DIGEST, publisher="p")
    ingest(
        cat,
        make_manifest("x", "1.0.0", description="v2"),
        digest="sha256:" + "b" * 64,
        publisher="p",
    )
    assert len(cat.all()) == 1  # same reference replaces
    got = cat.get("x:1.0.0")
    assert got is not None and got.digest == "sha256:" + "b" * 64


def test_index_document_includes_indexed_fields() -> None:
    manifest = make_manifest(
        "nav", description="navigation", tags=["mobility.wheeled"], inputs=["Obs"], outputs=["Act"]
    )
    doc = index_document(manifest)
    for token in ["nav", "navigation", "mobility.wheeled", "Obs", "Act"]:
        assert token in doc


def test_sql_catalog_roundtrip_matches_inmemory(tmp_path: Path) -> None:
    sql = SqlCatalog(f"sqlite+pysqlite:///{tmp_path / 'catalog.db'}")
    mem = InMemoryCatalog()
    manifest = make_manifest("exc", "1.2.0", tags=["mobility.wheeled"], description="excavation")
    entry = ingest(mem, manifest, digest=DIGEST, publisher="alice")

    sql.add(entry)
    got = sql.get("exc:1.2.0")
    assert got is not None
    assert got.manifest == manifest  # the Core manifest round-trips verbatim
    assert got.embedding == entry.embedding
    assert got.publisher == "alice" and got.digest == DIGEST
    assert [e.reference for e in sql.all()] == ["exc:1.2.0"]
    assert sql.get("missing:1.0.0") is None
    sql.dispose()


def test_sql_catalog_factory_and_requires_url() -> None:
    catalog = sql_catalog("sqlite+pysqlite:///:memory:")
    assert isinstance(catalog, SqlCatalog)
    catalog.dispose()  # close the pooled connection (no ResourceWarning)
    with pytest.raises(ValueError):
        SqlCatalog()
