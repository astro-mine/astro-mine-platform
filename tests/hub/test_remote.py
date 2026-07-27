"""Remote OCI Distribution transport (RM-P1-HUB-06): pull/publish against any registry.

Exercised over **real HTTP** against an in-process OCI registry (``tests/_fake_registry``), because
the deliverable is wire conformance: the ``/v2/…`` endpoints, the blob-upload session, the Referrers
API (and its fallback tag), the bearer-token handshake, and — the load-bearing property —
**verify-twice survives the wire**: a registry that serves bytes not matching their content address
fails closed (hub.md §2.3, §9).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pytest

from astro_mine.hub.client import HubClient
from astro_mine.hub.registry import (
    ArtifactExistsError,
    ArtifactNotFound,
    Blob,
    Credentials,
    IntegrityError,
    Registry,
    RegistryClient,
    RegistryHttpError,
    RemoteRegistry,
    is_remote,
    open_registry,
)
from astro_mine.hub.registry._remote import _Redirects, _split_location
from astro_mine.hub.supply_chain import ARTIFACT_TYPE_SIGNATURE, generate_keypair, verify

from ._fake_registry import FakeRegistry
from .conftest import make_manifest

PAYLOAD = Blob("application/octet-stream", b"onnx-policy-bytes")


def _publish(remote: RemoteRegistry, name: str = "pol", version: str = "1.0.0") -> str:
    artifact = remote.publish(
        name=name,
        version=version,
        kind="policy",
        config=make_manifest(name, version).model_dump(mode="json"),
        layers=[PAYLOAD],
    )
    return artifact.digest


def test_publish_resolve_pull_round_trip() -> None:
    with FakeRegistry() as fake:
        remote = RemoteRegistry(f"{fake.location}/astro-mine")
        digest = _publish(remote)

        assert remote.resolve("pol:1.0.0").digest == digest
        manifest = remote.read_manifest(digest)
        assert manifest["artifactType"] == "application/vnd.astro-mine.policy.v1"
        assert remote.pull_blob(manifest["layers"][0]["digest"]) == PAYLOAD.data
        assert b'"name":"pol"' in remote.read_config(digest)
        remote.verify(digest)  # walks config + layers, re-hashing each

        # the artifact landed in the conventional OCI coordinates: <prefix>/<name>:<version>
        assert "astro-mine/pol" in fake.repos
        assert fake.repos["astro-mine/pol"].tags == {"1.0.0": digest}


def test_registry_client_protocol_is_satisfied_by_both_transports(tmp_path: Path) -> None:
    with FakeRegistry() as fake:
        assert isinstance(RemoteRegistry(fake.location), RegistryClient)
    assert isinstance(Registry(tmp_path), RegistryClient)


def test_verify_twice_on_the_wire_fails_closed() -> None:
    """A registry serving bytes that do not hash to the requested digest must never be trusted."""
    with FakeRegistry() as fake:
        remote = RemoteRegistry(f"{fake.location}/astro-mine")
        digest = _publish(remote)
        layer = remote.read_manifest(digest)["layers"][0]["digest"]

        fake.tamper("astro-mine/pol", layer, b"malicious-bytes")
        with pytest.raises(IntegrityError):
            remote.pull_blob(layer)  # re-hashed before return — no unverified bytes escape
        with pytest.raises(IntegrityError):
            remote.verify(digest)


def test_resolve_by_digest_checks_the_manifest_bytes() -> None:
    with FakeRegistry() as fake:
        remote = RemoteRegistry(f"{fake.location}/astro-mine")
        digest = _publish(remote)

        assert remote.resolve(f"pol@{digest}").digest == digest
        # the registry lies: it serves a different manifest under the requested digest
        fake.repos["astro-mine/pol"].manifests[digest] = b'{"schemaVersion":2}'
        with pytest.raises(IntegrityError):
            remote.resolve(f"pol@{digest}")


def test_immutable_name_version() -> None:
    with FakeRegistry() as fake:
        remote = RemoteRegistry(f"{fake.location}/astro-mine")
        _publish(remote)
        with pytest.raises(ArtifactExistsError):
            _publish(remote)


def test_missing_artifact_and_unknown_digest_repository() -> None:
    with FakeRegistry() as fake:
        remote = RemoteRegistry(f"{fake.location}/astro-mine")
        with pytest.raises(ArtifactNotFound):
            remote.resolve("nope:1.0.0")
        # a *bare* digest cannot name a repository: OCI blobs are repository-scoped
        with pytest.raises(ArtifactNotFound, match="repository-scoped"):
            remote.resolve("sha256:" + "a" * 64)

        digest = _publish(remote)
        # …but once the artifact has been resolved/published through this client, it can
        assert remote.resolve(digest).digest == digest
        with pytest.raises(KeyError):
            remote.pull_blob("sha256:" + "b" * 64)


def test_attestations_round_trip_through_the_referrers_api() -> None:
    with FakeRegistry() as fake:
        remote = RemoteRegistry(f"{fake.location}/astro-mine")
        client = HubClient(remote)
        private_pem, public_pem = generate_keypair()
        artifact = client.publish(
            name="pol",
            version="1.0.0",
            kind="policy",
            manifest=make_manifest("pol", "1.0.0"),
            layers=[PAYLOAD],
            private_key_pem=private_pem,
        )
        # the full supply chain (signature + SLSA + SBOM) re-verifies over HTTP, pinned to the key
        verify(remote, artifact.digest, trusted_public_key_pem=public_pem)
        assert client.verify("pol:1.0.0") == artifact.digest
        assert len(remote.referrers(artifact.digest, artifact_type=ARTIFACT_TYPE_SIGNATURE)) == 1
        assert len(remote.referrers(artifact.digest)) == 3


def test_referrers_fall_back_to_the_tag_scheme() -> None:
    """A registry without the Referrers API still serves attestations (the spec's fallback tag)."""
    with FakeRegistry(supports_referrers=False) as fake:
        remote = RemoteRegistry(f"{fake.location}/astro-mine")
        digest = _publish(remote)
        remote.attach(
            subject=digest,
            artifact_type=ARTIFACT_TYPE_SIGNATURE,
            blob=Blob("application/json", b'{"sig":"x"}'),
        )
        found = remote.referrers(digest, artifact_type=ARTIFACT_TYPE_SIGNATURE)
        assert [descriptor.artifact_type for descriptor in found] == [ARTIFACT_TYPE_SIGNATURE]
        assert remote.referrers(digest, artifact_type="application/vnd.other") == []
        # the fallback index is a real tag: sha256-<hex>
        assert digest.replace(":", "-") in fake.repos["astro-mine/pol"].tags


def test_listing_tags_and_catalog() -> None:
    with FakeRegistry() as fake:
        remote = RemoteRegistry(f"{fake.location}/astro-mine")
        _publish(remote, "pol", "1.0.0")
        _publish(remote, "pol", "1.1.0")
        _publish(remote, "world", "2.0.0")
        remote.attach(  # a fallback tag must never be mistaken for a version
            subject=remote.resolve("pol:1.0.0").digest,
            artifact_type=ARTIFACT_TYPE_SIGNATURE,
            blob=Blob("application/json", b"{}"),
        )
        assert remote.versions("pol") == ["1.0.0", "1.1.0"]
        assert remote.references() == ["pol:1.0.0", "pol:1.1.0", "world:2.0.0"]
        assert remote.versions("absent") == []


def test_registry_without_catalog_api_degrades() -> None:
    """ghcr has no ``_catalog``; discovery degrades to empty rather than failing the pull path."""
    with FakeRegistry(supports_catalog=False) as fake:
        remote = RemoteRegistry(f"{fake.location}/astro-mine")
        _publish(remote)
        assert remote.references() == []
        assert remote.versions("pol") == ["1.0.0"]  # per-repo tag listing still works


def test_bearer_token_handshake_with_credentials() -> None:
    with FakeRegistry(require_auth=True) as fake:
        remote = RemoteRegistry(
            f"{fake.location}/astro-mine", credentials=Credentials("user", "pass")
        )
        digest = _publish(remote)
        assert remote.resolve("pol:1.0.0").digest == digest
        assert fake.token_requests >= 1  # the 401 challenge was answered, then cached
        first = fake.token_requests
        remote.verify(digest)
        assert fake.token_requests == first  # token reused, not re-fetched per request


def test_bad_credentials_fail_closed() -> None:
    with FakeRegistry(require_auth=True) as fake:
        remote = RemoteRegistry(f"{fake.location}/astro-mine", credentials=Credentials("u", "bad"))
        with pytest.raises(RegistryHttpError):
            _publish(remote)


def test_anonymous_against_an_authenticated_registry_fails_closed() -> None:
    with FakeRegistry(require_auth=True) as fake:
        remote = RemoteRegistry(f"{fake.location}/astro-mine", credentials=None)
        with pytest.raises(RegistryHttpError):
            remote.resolve("pol:1.0.0")


def test_unreachable_registry_raises() -> None:
    remote = RemoteRegistry("http://127.0.0.1:1/astro-mine", timeout=1.0)
    with pytest.raises(RegistryHttpError):
        remote.resolve("pol:1.0.0")


def test_authorization_is_dropped_on_a_cross_host_redirect() -> None:
    """Blob egress 307s to object storage — the registry credential must not follow it there."""
    request = urllib.request.Request("https://registry.example.org/v2/x/blobs/sha256:abc")
    request.add_header("Authorization", "Bearer secret")

    same_host = _Redirects().redirect_request(
        request, None, 307, "", {}, "https://registry.example.org/v2/x/blobs/other"
    )
    assert same_host is not None
    assert same_host.get_header("Authorization") == "Bearer secret"

    cross_host = _Redirects().redirect_request(
        request, None, 307, "", {}, "https://blobs.cdn.example.net/xyz"
    )
    assert cross_host is not None
    assert cross_host.get_header("Authorization") is None


@pytest.mark.parametrize(
    ("location", "expected"),
    [
        ("ghcr.io/astro-mine", ("https", "ghcr.io", "astro-mine")),
        ("https://registry.example.org/commons/", ("https", "registry.example.org", "commons")),
        ("http://localhost:5000", ("http", "localhost:5000", "")),
        ("oci://ghcr.io/astro-mine/hub", ("https", "ghcr.io", "astro-mine/hub")),
    ],
)
def test_split_location(location: str, expected: tuple[str, str, str]) -> None:
    assert _split_location(location) == expected


def test_split_location_rejects_a_hostless_location() -> None:
    with pytest.raises(ValueError, match="malformed registry location"):
        _split_location("https://")


@pytest.mark.parametrize(
    ("location", "remote"),
    [
        ("ghcr.io/astro-mine", True),
        ("localhost:5000", True),
        ("http://localhost:5000", True),
        ("oci://ghcr.io/x", True),
        ("./reg", False),
        ("/var/lib/hub", False),
        ("~/reg", False),
        ("reg", False),
    ],
)
def test_is_remote(location: str, remote: bool) -> None:
    assert is_remote(location) is remote


def test_open_registry_picks_the_transport(tmp_path: Path) -> None:
    assert isinstance(open_registry(tmp_path / "reg"), Registry)
    assert isinstance(open_registry("ghcr.io/astro-mine"), RemoteRegistry)
