"""Verified payload-layer retrieval (RM-P1-HUB-06; hub.md §3, §7; conventions.md §5, §9).

An artifact's *config* is its Core manifest; its *layers* are what it exists to carry — the ONNX
policy, the SADF bundle, the Zarr/COG world. Retrieving them MUST stay inside the verify-before-use
contract: the client re-hashes every layer against the digest the **verified** manifest commits to
before it returns or writes a byte, so no consumer has to drop to ``registry.pull_blob()`` for raw,
unverified content. Every path here fails closed on tamper.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astro_mine.hub.client import HubClient
from astro_mine.hub.registry import ArtifactNotFound, Blob, IntegrityError, Registry
from astro_mine.hub.registry._oci import blob_path
from astro_mine.hub.supply_chain import SupplyChainError, generate_keypair

from .conftest import make_manifest

ONNX = Blob("application/vnd.astro-mine.onnx", b"\x08\x01onnx-policy")
SADF = Blob("application/vnd.astro-mine.sadf+json", b'{"asset":"excavator"}')


def _client(tmp_path: Path, *, cache: bool = False) -> tuple[HubClient, str]:
    """A client with a signed, two-layer artifact published — the multi-layer case of the issue."""
    client = HubClient(
        Registry(tmp_path / "reg"), cache_dir=(tmp_path / "cache") if cache else None
    )
    private_pem, _ = generate_keypair()
    artifact = client.publish(
        name="pol",
        version="1.0.0",
        kind="policy",
        manifest=make_manifest("pol", "1.0.0"),
        layers=[ONNX, SADF],
        private_key_pem=private_pem,
    )
    return client, artifact.digest


def test_pull_payload_returns_verified_layers(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    layers = client.pull_payload("pol:1.0.0")

    assert [layer.data for layer in layers] == [ONNX.data, SADF.data]  # manifest order
    assert [layer.media_type for layer in layers] == [ONNX.media_type, SADF.media_type]
    assert [layer.digest for layer in layers] == [ONNX.digest, SADF.digest]
    assert [layer.size for layer in layers] == [ONNX.size, SADF.size]


def test_pull_payload_filters_by_media_type(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    layers = client.pull_payload("pol:1.0.0", media_type=ONNX.media_type)
    assert [layer.data for layer in layers] == [ONNX.data]


def test_payload_descriptors_without_fetching_bytes(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    descriptors = client.payload_descriptors("pol:1.0.0")
    assert [d.digest for d in descriptors] == [ONNX.digest, SADF.digest]


def test_pull_layer_by_digest(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    assert client.pull_layer("pol:1.0.0", SADF.digest) == SADF.data


def test_pull_layer_refuses_a_blob_outside_the_manifest(tmp_path: Path) -> None:
    """A blob that is not a layer of the verified manifest cannot be laundered out of the store."""
    client, _ = _client(tmp_path)
    stray = Blob("application/octet-stream", b"unattested")
    client.registry.publish(  # present in the store, but not a layer of pol:1.0.0
        name="other", version="1.0.0", kind="policy", config=b"{}", layers=[stray]
    )
    with pytest.raises(ArtifactNotFound):
        client.pull_layer("pol:1.0.0", stray.digest)


def test_payload_retrieval_fails_closed_on_a_tampered_layer(tmp_path: Path) -> None:
    """The registry's bytes no longer hash to the digest the signed manifest committed to."""
    client, _ = _client(tmp_path)
    blob_path(Path(client.registry.path), ONNX.digest).write_bytes(b"malicious")

    with pytest.raises((IntegrityError, SupplyChainError)):
        client.pull_payload("pol:1.0.0")
    # even with the supply-chain check skipped, the layer's own content address is still enforced
    with pytest.raises(IntegrityError):
        client.pull_payload("pol:1.0.0", verify=False)
    with pytest.raises(IntegrityError):
        client.materialize("pol:1.0.0", dest=tmp_path / "out", verify=False)


def test_payload_retrieval_fails_closed_on_a_missing_signature(tmp_path: Path) -> None:
    """The pull-side half of verify-twice, on an artifact that never passed admission.

    Admission now refuses unsigned content, so this is staged through the **raw registry** — the
    shape a compromised or non-Hub registry can still serve. The client must not trust it
    (hub.md §2 principle 3: "a compromised registry must not be able to serve an artifact a client
    accepts").
    """
    registry = Registry(tmp_path / "reg")
    registry.publish(  # stored WITHOUT attestations: no signature/SLSA/SBOM referrers
        name="pol",
        version="1.0.0",
        kind="policy",
        config=make_manifest().model_dump(mode="json"),
        layers=[ONNX],
    )
    client = HubClient(registry)
    with pytest.raises(SupplyChainError):
        client.pull_payload("pol:1.0.0")
    assert client.pull_payload("pol:1.0.0", require=())[0].data == ONNX.data


def test_materialize_writes_content_addressed_paths(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    paths = client.materialize("pol:1.0.0", dest=tmp_path / "out")

    assert [p.name for p in paths] == [ONNX.digest.split(":")[1], SADF.digest.split(":")[1]]
    assert [p.read_bytes() for p in paths] == [ONNX.data, SADF.data]
    assert client.materialize("pol:1.0.0", dest=tmp_path / "out") == paths  # idempotent


def test_materialize_defaults_to_the_client_cache(tmp_path: Path) -> None:
    client, _ = _client(tmp_path, cache=True)
    (path,) = client.materialize("pol:1.0.0", media_type=SADF.media_type)
    assert path.parent == tmp_path / "cache"
    assert path.read_bytes() == SADF.data


def test_materialize_without_a_destination_is_an_error(tmp_path: Path) -> None:
    client, _ = _client(tmp_path)
    with pytest.raises(ValueError, match="dest directory or a client cache_dir"):
        client.materialize("pol:1.0.0")


def test_cached_layers_are_re_verified_not_trusted(tmp_path: Path) -> None:
    """A poisoned cache entry is caught: the cache is content-addressed, so it must self-verify."""
    client, _ = _client(tmp_path, cache=True)
    assert client.pull_payload("pol:1.0.0")[0].data == ONNX.data  # populates the cache

    cached = tmp_path / "cache" / ONNX.digest.split(":")[1]
    assert cached.exists()
    cached.write_bytes(b"poisoned")
    with pytest.raises(IntegrityError):
        client.pull_payload("pol:1.0.0")
