"""Signed publish + discovery through Hub, pulled/verified by content hash (FLEET-10).

Exercises the acceptance criteria end-to-end against a temporary Hub OCI-layout registry
(pytest ``tmp_path``): a signed asset bundle publishes to Hub and is pulled/verified **by
content hash**, its catalog metadata is discoverable, a tampered pull fails closed, and the
export-control gate refuses a reserved/gated capability tag before it reaches Hub.
"""

from __future__ import annotations

import pytest

from astro_mine.core.registry import PluginKind, PluginManifest
from astro_mine.core.sadf import SadfDocument, model, to_wire
from astro_mine.core.sadf.enums import CapabilityTag, GeometryFormat, GeometryRole
from astro_mine.fleet.capabilities import CapabilityError
from astro_mine.fleet.library import load_reference
from astro_mine.fleet.packaging import oci
from astro_mine.fleet.packaging.hub import HubError, discover_asset, publish_asset, pull_asset
from astro_mine.fleet.templates import resolve_family
from astro_mine.hub.registry import ArtifactExistsError, Blob, Registry, open_registry
from astro_mine.hub.supply_chain import SupplyChainError, generate_keypair


def test_signed_publish_then_pull_and_verify_by_content_hash(tmp_path):
    doc = load_reference("relay_orbiter")
    private_pem, public_pem = generate_keypair()

    result = publish_asset(doc, open_registry(str(tmp_path / "reg")), sign_key=private_pem)
    assert result.signed and result.namespace == "open"
    assert result.reference == "relay-orbiter:0.1.0"
    assert result.digest.startswith("sha256:")

    # Pull BY CONTENT HASH (the manifest digest); the full supply chain re-verifies before trust.
    restored = pull_asset(open_registry(str(tmp_path / "reg")), result.digest,
        trusted_public_key_pem=public_pem)
    assert to_wire(restored) == to_wire(doc)  # byte-identical rehydration


def test_json_layer_keeps_its_pinned_media_type(tmp_path):
    # Sim rehydrates the Asset from the SADF JSON layer -> its media type must not drift.
    doc = load_reference("prospecting_rover")
    private_pem, _ = generate_keypair()
    result = publish_asset(doc, open_registry(str(tmp_path / "reg")), sign_key=private_pem)

    manifest = Registry(tmp_path / "reg").read_manifest(result.digest)
    media_types = {layer["mediaType"] for layer in manifest["layers"]}
    assert oci.MEDIA_SADF_JSON in media_types
    assert oci.MEDIA_SADF_JSON == "application/vnd.astro-mine.asset.sadf.json.v1+json"


def test_catalog_metadata_is_discoverable(tmp_path):
    doc = load_reference("hauler")
    private_pem, _ = generate_keypair()
    result = publish_asset(doc, open_registry(str(tmp_path / "reg")), sign_key=private_pem)

    reference, digest = discover_asset(open_registry(str(tmp_path / "reg")), doc.asset.identity.id)
    assert reference == result.reference
    assert digest == result.digest


def test_tampered_pull_fails_closed(tmp_path):
    doc = load_reference("excavator")
    private_pem, public_pem = generate_keypair()
    result = publish_asset(doc, open_registry(str(tmp_path / "reg")), sign_key=private_pem)

    # Corrupt the stored SADF wire blob; verify-before-trust must reject it.
    manifest = Registry(tmp_path / "reg").read_manifest(result.digest)
    wire = next(la for la in manifest["layers"] if la["mediaType"] == oci.MEDIA_SADF_WIRE)
    blob = tmp_path / "reg" / "blobs" / "sha256" / wire["digest"].split(":", 1)[1]
    blob.write_bytes(b"tampered")
    with pytest.raises(SupplyChainError):
        pull_asset(open_registry(str(tmp_path / "reg")), result.digest,
            trusted_public_key_pem=public_pem)


def test_publish_requires_a_signing_key(tmp_path):
    """Unsigned publishing is refused — hub.md §9 defines no tier for unsigned content.

    This used to store the artifact with no attestations, leaving a consumer to pull it with an
    empty requirement set (astro-mine-hub#32)."""
    doc = load_reference("isru_plant")
    registry = open_registry(str(tmp_path / "reg"))
    with pytest.raises(TypeError, match="sign_key"):
        publish_asset(doc, registry)  # type: ignore[call-arg]
    # Nothing was published. Asserting on the *contents* rather than on the directory's absence:
    # opening the registry is the caller's act now that Fleet takes one injected, so the layout
    # exists before the refusal. An empty OCI layout is not a stored artifact, and "no artifact"
    # was always the invariant this test meant (astro-mine-hub#32).
    assert registry.references() == []


def test_resolved_family_publishes_and_republish_is_rejected(tmp_path):
    doc = resolve_family("surface-rover", {"chassis_mass_kg": 300.0}, variant="m300")
    private_pem, public_pem = generate_keypair()
    result = publish_asset(doc, open_registry(str(tmp_path / "reg")), sign_key=private_pem)

    # name:version is immutable -- a re-publish is refused (hub.md §2.1).
    with pytest.raises(ArtifactExistsError):
        publish_asset(doc, open_registry(str(tmp_path / "reg")), sign_key=private_pem)

    restored = pull_asset(open_registry(str(tmp_path / "reg")), result.digest,
        trusted_public_key_pem=public_pem)
    assert restored.asset.identity.id == "surface-rover-m300"


def test_publish_refuses_a_gated_capability_tag(tmp_path):
    private_pem, _ = generate_keypair()
    # A gated tag can't survive Core's loader, so build the doc structurally to prove the
    # publish-boundary export-control gate refuses it independently (fleet.md §9).
    asset = model.Asset(
        identity=model.Identity(id="x.gated", name="X", version="0.1.0", kind="rover"),
        capabilities=[CapabilityTag.OPERATIONAL_TARGETING],
        core_interface_versions={"sadf": "0.1.0"},
        root_frame="base",
    )
    doc = SadfDocument(sadf_version="0.1", asset=asset)
    with pytest.raises(CapabilityError, match="reserved/gated"):
        publish_asset(doc, open_registry(str(tmp_path / "reg")), sign_key=private_pem)


def test_geometry_layer_and_provenance_are_published(tmp_path):
    private_pem, _ = generate_keypair()
    (tmp_path / "mesh.usda").write_bytes(b"#usda 1.0\n")
    asset = model.Asset(
        identity=model.Identity(id="geo-asset", name="G", version="0.1.0", kind="rover"),
        core_interface_versions={"sadf": "0.1.0"},
        root_frame="base",
        frames=[model.Frame(name="base")],
        geometry=[
            model.GeometryRef(
                role=GeometryRole.VISUAL, format=GeometryFormat.USD, uri="mesh.usda", frame="base"
            )
        ],
    )
    doc = SadfDocument(sadf_version="0.1", asset=asset)
    result = publish_asset(doc, open_registry(str(tmp_path / "reg")), base_dir=tmp_path,
        sign_key=private_pem)

    registry = Registry(tmp_path / "reg")
    manifest = registry.read_manifest(result.digest)
    assert oci.MEDIA_GEOMETRY_USD in {layer["mediaType"] for layer in manifest["layers"]}
    config = PluginManifest.model_validate_json(registry.read_config(result.digest))
    assert config.kind is PluginKind.ASSET
    assert config.provenance is not None
    assert "mesh.usda" in config.provenance.source_content_hashes


def test_pull_without_a_json_layer_errors(tmp_path):
    manifest = PluginManifest(
        name="no.json", version="0.1.0", kind=PluginKind.ASSET, core_interfaces={"sadf": "0.1.0"}
    )
    registry = Registry(tmp_path / "reg")
    artifact = registry.publish(
        name="no.json",
        version="0.1.0",
        kind="asset",
        config=manifest.model_dump(mode="json"),
        layers=[Blob("application/octet-stream", b"opaque")],
    )
    with pytest.raises(HubError, match="no SADF JSON layer"):
        pull_asset(open_registry(str(tmp_path / "reg")), artifact.digest, verify=False)
