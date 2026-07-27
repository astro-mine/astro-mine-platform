"""Search tests (RM-P1-HUB-02): faceted, full-text, capability, semantic + degradation."""

from __future__ import annotations

import pytest

from astro_mine.core.registry import PluginKind
from astro_mine.hub.index import CatalogEntry, InMemoryCatalog, ingest
from astro_mine.hub.search import SearchQuery, cosine, embed, search, tokenize

from .conftest import make_manifest


def _seed() -> InMemoryCatalog:
    cat = InMemoryCatalog()
    ingest(
        cat,
        make_manifest(
            "excavator",
            "1.0.0",
            kind=PluginKind.POLICY,
            description="lunar excavation digging policy",
            tags=["mobility.wheeled"],
            inputs=["Observation"],
            outputs=["Action"],
        ),
        digest="sha256:" + "a" * 64,
        publisher="alice",
    )
    ingest(
        cat,
        make_manifest(
            "orbiter",
            "1.0.0",
            kind=PluginKind.WORLD_PROVIDER,
            description="orbital imaging world",
            tags=["mobility.orbiter"],
            interfaces={"world_provider": "0.1.0"},
        ),
        digest="sha256:" + "b" * 64,
        publisher="bob",
        namespace="curated",
    )
    return cat


def _names(results: list) -> set[str]:
    return {r.entry.name for r in results}


def test_faceted_kind_license_namespace() -> None:
    cat = _seed()
    assert _names(search(cat, SearchQuery(kind="policy"))) == {"excavator"}
    assert _names(search(cat, SearchQuery(namespace="curated"))) == {"orbiter"}
    assert _names(search(cat, SearchQuery(license="Apache-2.0"))) == {"excavator", "orbiter"}
    assert search(cat, SearchQuery(license="MIT")) == []


def test_empty_query_returns_all() -> None:
    assert len(search(_seed(), SearchQuery())) == 2


def test_full_text_match_and_miss() -> None:
    cat = _seed()
    assert _names(search(cat, SearchQuery(text="digging"))) == {"excavator"}
    assert search(cat, SearchQuery(text="nonexistentterm")) == []


def test_capability_negotiation() -> None:
    cat = _seed()
    assert _names(search(cat, SearchQuery(interfaces={"policy": "0.1.0"}))) == {"excavator"}
    assert search(cat, SearchQuery(interfaces={"policy": "0.2.0"})) == []  # incompatible minor
    assert _names(search(cat, SearchQuery(capability_tags=["mobility.orbiter"]))) == {"orbiter"}


def test_semantic_ranking_prefers_similar() -> None:
    results = search(_seed(), SearchQuery(semantic="excavation digging"))
    assert results[0].entry.name == "excavator"
    assert results[0].score > 0.0


def test_deprecated_and_yanked_excluded_by_default() -> None:
    cat = InMemoryCatalog()
    entry = ingest(cat, make_manifest("old", "1.0.0"), digest="sha256:" + "a" * 64, publisher="p")
    entry.yanked = True
    assert search(cat, SearchQuery()) == []
    assert len(search(cat, SearchQuery(include_yanked=True))) == 1
    entry.yanked = False
    entry.deprecated = True
    assert search(cat, SearchQuery()) == []
    assert len(search(cat, SearchQuery(include_deprecated=True))) == 1


def test_semantic_degrades_when_disabled() -> None:
    results = search(_seed(), SearchQuery(semantic="excavation", use_semantic=False))
    assert results[0].entry.name == "excavator"  # lexical overlap fallback


def test_semantic_falls_back_when_no_embeddings() -> None:
    cat = InMemoryCatalog()
    manifest = make_manifest("excavator", "1.0.0", description="excavation digging")
    cat.add(
        CatalogEntry(manifest=manifest, digest="sha256:" + "a" * 64, publisher="p", embedding=())
    )
    results = search(cat, SearchQuery(semantic="excavation"))
    assert results[0].entry.name == "excavator" and results[0].score > 0.0


def test_limit_truncates() -> None:
    cat = InMemoryCatalog()
    for i in range(5):
        ingest(cat, make_manifest(f"p{i}", "1.0.0"), digest=f"sha256:{i:064x}", publisher="p")
    assert len(search(cat, SearchQuery(limit=3))) == 3


def test_embed_and_cosine_helpers() -> None:
    assert cosine((), (1.0,)) == 0.0
    assert cosine((1.0, 0.0), (2.0,)) == 0.0  # shape mismatch
    assert cosine((1.0, 0.0), (1.0, 0.0)) == pytest.approx(1.0)
    assert embed("") == tuple([0.0] * 64)  # empty → zero vector
    assert tokenize("Hello, World-2") == ["hello", "world", "2"]
