"""Curation tests (RM-P1-HUB-05): namespace tiers + auditable yank/deprecate/promote."""

from __future__ import annotations

from pathlib import Path

import pytest

from astro_mine.hub.client import HubClient
from astro_mine.hub.curation import CurationError, deprecate, promote, yank
from astro_mine.hub.index import InMemoryCatalog, ingest
from astro_mine.hub.index._sql import SqlCatalog
from astro_mine.hub.policy import InMemoryAuditLog
from astro_mine.hub.registry import Registry
from astro_mine.hub.search import SearchQuery, search
from astro_mine.hub.supply_chain import generate_keypair

from .conftest import make_manifest

DIGEST = "sha256:" + "a" * 64


def _catalog() -> InMemoryCatalog:
    cat = InMemoryCatalog()
    ingest(cat, make_manifest("art", "1.0.0"), digest=DIGEST, publisher="p")
    return cat


def _published(tmp_path: Path) -> tuple[Registry, InMemoryCatalog]:
    """A genuinely signed, admitted artifact — what a real promotion candidate looks like."""
    registry = Registry(tmp_path / "reg")
    client = HubClient(registry)
    private_pem, _ = generate_keypair()
    client.publish(
        name="art",
        version="1.0.0",
        kind="policy",
        manifest=make_manifest("art", "1.0.0"),
        private_key_pem=private_pem,
    )
    assert isinstance(client.catalog, InMemoryCatalog)
    return registry, client.catalog


def test_promote_updates_namespace_and_audits(tmp_path: Path) -> None:
    registry, cat = _published(tmp_path)
    audit = InMemoryAuditLog()
    entry = promote(cat, "art:1.0.0", audit=audit, registry=registry)
    assert entry.namespace == "curated"
    got = cat.get("art:1.0.0")
    assert got is not None and got.namespace == "curated"
    assert audit.records[-1].action == "promote"


def test_promotion_into_a_trusted_tier_requires_a_registry() -> None:
    """hub.md §2 principle 3 — promotion cannot grant trust it is unable to check.

    Without a registry there is no evidence to verify against, so the only fail-closed answer is
    to refuse; the denial is audited."""
    cat = _catalog()
    audit = InMemoryAuditLog()
    with pytest.raises(CurationError, match="requires a registry"):
        promote(cat, "art:1.0.0", to="verified", audit=audit)
    got = cat.get("art:1.0.0")
    assert got is not None and got.namespace == "open"  # unchanged
    assert audit.records[-1].reason.startswith("denied:")


def test_unsigned_content_is_never_promoted_to_a_verified_namespace(tmp_path: Path) -> None:
    """The sentence in hub.md §2 principle 3 that described a check which did not exist.

    The artifact is staged straight into the registry with no attestations — the shape admission
    now refuses — and then offered for promotion."""
    registry = Registry(tmp_path / "reg")
    artifact = registry.publish(
        name="art",
        version="1.0.0",
        kind="policy",
        config=make_manifest("art", "1.0.0").model_dump(mode="json"),
    )
    cat = InMemoryCatalog()
    ingest(cat, make_manifest("art", "1.0.0"), digest=artifact.digest, publisher="p")
    audit = InMemoryAuditLog()

    with pytest.raises(CurationError, match="does not verify"):
        promote(cat, "art:1.0.0", to="verified", audit=audit, registry=registry)

    got = cat.get("art:1.0.0")
    assert got is not None and got.namespace == "open"  # the tier was not granted
    assert audit.records[-1].reason.startswith("denied:")  # and the refusal is auditable


def test_demotion_to_open_needs_no_registry() -> None:
    """Lowering trust is always safe, so it must not require evidence to be checkable."""
    cat = _catalog()
    entry = promote(cat, "art:1.0.0", to="open")
    assert entry.namespace == "open"


def test_promote_unknown_namespace_raises() -> None:
    with pytest.raises(CurationError):
        promote(_catalog(), "art:1.0.0", to="nope")


def test_yank_excludes_from_search_and_audits() -> None:
    cat = _catalog()
    audit = InMemoryAuditLog()
    yank(cat, "art:1.0.0", audit=audit)
    got = cat.get("art:1.0.0")
    assert got is not None and got.yanked
    assert search(cat, SearchQuery()) == []  # yanked excluded by default
    assert audit.records[0].action == "yank"


def test_deprecate_flags_entry() -> None:
    cat = _catalog()
    entry = deprecate(cat, "art:1.0.0")  # no audit sink → covers the optional-audit path
    assert entry.deprecated


def test_unknown_reference_raises() -> None:
    with pytest.raises(CurationError):
        yank(_catalog(), "missing:1.0.0")


def test_curation_persists_through_sql_backend(tmp_path: Path) -> None:
    sql = SqlCatalog(f"sqlite+pysqlite:///{tmp_path / 'catalog.db'}")
    ingest(sql, make_manifest("art", "1.0.0"), digest=DIGEST, publisher="p")
    yank(sql, "art:1.0.0")
    got = sql.get("art:1.0.0")
    assert got is not None and got.yanked  # persisted through the backend, not just in memory
    sql.dispose()
