"""Client SDK tests (RM-P1-HUB-06): publish/pull/verify, cache, verify-before-load, fail-closed."""

from __future__ import annotations

from pathlib import Path

import pytest

from astro_mine.core.registry import PluginKind, PluginManifest
from astro_mine.hub.client import HubClient, catalog_from_registry
from astro_mine.hub.registry import Registry
from astro_mine.hub.registry._oci import blob_path
from astro_mine.hub.resolve import ResolutionRequest
from astro_mine.hub.supply_chain import SupplyChainError, generate_keypair

from .conftest import make_manifest


def _client(tmp_path: Path, *, cache: bool = False, trusted: bytes | None = None) -> HubClient:
    return HubClient(
        Registry(tmp_path / "reg"),
        cache_dir=(tmp_path / "cache") if cache else None,
        trusted_public_key_pem=trusted,
    )


def test_publish_verify_pull_roundtrip(tmp_path: Path) -> None:
    private_pem, public_pem = generate_keypair()
    client = _client(tmp_path, trusted=public_pem)
    artifact = client.publish(
        name="pol",
        version="1.0.0",
        kind="policy",
        manifest=make_manifest("pol", "1.0.0"),
        private_key_pem=private_pem,
    )
    assert client.verify("pol:1.0.0") == artifact.digest  # verify-twice, pinned to the trusted key
    config = client.pull("pol:1.0.0")
    assert PluginManifest.model_validate_json(config).name == "pol"


def test_publish_verifies_at_admission(tmp_path: Path) -> None:
    """A signed publish verifies **fail-closed at admission** (hub.md §2.3, §9 "verify twice"): an
    artifact that would not verify — here, signed with a key the client does not trust — is rejected
    at publish, not accepted only to fail at some later pull."""
    signing_pem, _ = generate_keypair()
    _, other_public = generate_keypair()  # a *different* keypair's public half
    client = _client(tmp_path, trusted=other_public)  # the client trusts a different key
    with pytest.raises(SupplyChainError):
        client.publish(
            name="pol",
            version="1.0.0",
            kind="policy",
            manifest=make_manifest("pol", "1.0.0"),
            private_key_pem=signing_pem,  # signed with a key the client does not trust
        )
    # The admission check rejected it before it was indexed into the catalog.
    assert list(client.catalog.all()) == []


def test_pull_tampered_fails_closed(tmp_path: Path) -> None:
    private_pem, _ = generate_keypair()
    client = _client(tmp_path)
    artifact = client.publish(
        name="pol",
        version="1.0.0",
        kind="policy",
        manifest=make_manifest("pol", "1.0.0"),
        private_key_pem=private_pem,
    )
    config_digest = client.registry.read_manifest(artifact.digest)["config"]["digest"]
    blob_path(client.registry.path, config_digest).write_bytes(b'{"tampered":true}')
    with pytest.raises(SupplyChainError):
        client.pull("pol:1.0.0")


def test_cache_serves_repeat_pull(tmp_path: Path) -> None:
    private_pem, _ = generate_keypair()
    client = _client(tmp_path, cache=True)
    artifact = client.publish(
        name="pol",
        version="1.0.0",
        kind="policy",
        manifest=make_manifest("pol", "1.0.0"),
        private_key_pem=private_pem,
    )
    first = client.pull("pol:1.0.0", verify=False)
    assert (tmp_path / "cache" / artifact.digest.split(":", 1)[1]).exists()
    assert client.pull("pol:1.0.0", verify=False) == first  # served from cache


def test_load_registers_through_core(tmp_path: Path) -> None:
    private_pem, _ = generate_keypair()
    client = _client(tmp_path)
    client.publish(
        name="pol",
        version="1.0.0",
        kind="policy",
        manifest=make_manifest("pol", "1.0.0"),
        private_key_pem=private_pem,
    )
    loaded = client.load("pol:1.0.0")
    assert loaded.name == "pol" and loaded.kind == PluginKind.POLICY


def test_resolve_via_client(tmp_path: Path) -> None:
    private_pem, _ = generate_keypair()
    client = _client(tmp_path)
    for version in ("1.0.0", "1.2.0"):
        client.publish(
            name="pol",
            version=version,
            kind="policy",
            manifest=make_manifest("pol", version),
            private_key_pem=private_pem,
        )
    assert client.resolve(ResolutionRequest(name="pol")).primary.version == "1.2.0"


def test_publish_requires_a_signing_key(tmp_path: Path) -> None:
    """Unsigned content cannot be admitted at all — the posture hub#32 decided.

    ``hub.md`` §9 tiers artifacts as open (self-published, *signed*, unreviewed) / curated /
    verified; there is no tier for unsigned content, so indexing it would describe something the
    trust model has no words for. Signing is offline and accountless, so this costs the local
    tier nothing."""
    client = _client(tmp_path)
    with pytest.raises(TypeError, match="private_key_pem"):
        client.publish(  # type: ignore[call-arg]
            name="pol", version="1.0.0", kind="policy", manifest=make_manifest("pol", "1.0.0")
        )
    assert client.catalog.get("pol:1.0.0") is None  # nothing was indexed


def test_unverified_pull_still_reaches_a_signed_artifact(tmp_path: Path) -> None:
    """`verify=False` remains an explicit opt-out for a caller that has already verified."""
    client = _client(tmp_path)
    private_pem, _ = generate_keypair()
    client.publish(
        name="pol",
        version="1.0.0",
        kind="policy",
        manifest=make_manifest("pol", "1.0.0"),
        private_key_pem=private_pem,
    )
    assert client.verify("pol:1.0.0", require=())  # integrity only; no attestations required
    config = client.pull("pol:1.0.0", verify=False)
    assert PluginManifest.model_validate_json(config).name == "pol"


def test_catalog_from_registry_rebuilds(tmp_path: Path) -> None:
    private_pem, _ = generate_keypair()
    registry = Registry(tmp_path / "reg")
    client = HubClient(registry)
    client.publish(
        name="a",
        version="1.0.0",
        kind="policy",
        manifest=make_manifest("a", "1.0.0"),
        private_key_pem=private_pem,
    )
    client.publish(
        name="b",
        version="2.0.0",
        kind="world",
        manifest=make_manifest("b", "2.0.0", kind=PluginKind.WORLD_PROVIDER),
        private_key_pem=private_pem,
    )
    rebuilt = catalog_from_registry(registry)
    assert {entry.reference for entry in rebuilt.all()} == {"a:1.0.0", "b:2.0.0"}
