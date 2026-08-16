"""RM-P1-STUDIO-09 — the Fleet/Hub asset catalog as the robot menu + a servable geometry preview.

Studio reads the catalog and a selected asset's layers directly from Hub by content hash — the same
"pure Core consumer, no sibling-package imports" stance as the world materializer. These tests build
asset artifacts the way Fleet publishes them (a SADF-JSON layer + geometry blobs, indexed by a Core
manifest carrying the vehicle kind, capability tags, and `uri → digest` source hashes) so the seams
can be exercised without importing `astro_mine.fleet`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from astro_mine.core.registry import CapabilityTag, PluginKind, PluginManifest, Provenance
from astro_mine.hub.client import HubClient
from astro_mine.hub.registry import Blob, IntegrityError, Registry
from astro_mine.hub.registry._oci import blob_path
from astro_mine.hub.supply_chain import generate_keypair
from astro_mine.studio.hub import (
    AssetCatalog,
    AssetPreviewMaterializer,
    HubAssetCatalog,
    HubAssetPreviewMaterializer,
    PreviewError,
)
from astro_mine.studio.hub.catalog import (
    MEDIA_GEOMETRY_GLTF,
    MEDIA_GEOMETRY_USD,
    MEDIA_SADF_JSON,
)

ORBITER = "relay-orbiter:0.1.0"
HOPPER = "hopper:0.1.0"


def _sha(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _publish_asset(
    registry: Registry,
    private_pem: bytes,
    *,
    asset_id: str,
    kind: str,
    name: str,
    tags: list[CapabilityTag],
    geometry: dict[str, bytes] | None = None,
    with_sadf: bool = True,
) -> str:
    """Publish a signed asset artifact and return its ``name:version`` reference.

    The SADF-JSON layer is opaque to Studio (it copies, never parses it), so any JSON suffices; the
    geometry blobs are mapped ``uri → digest`` through ``provenance.source_content_hashes``, exactly
    as Fleet stamps them.
    """
    geometry = geometry or {}
    sadf = json.dumps(
        {"asset": {"identity": {"id": asset_id, "kind": kind, "name": name}}}
    ).encode()
    layers = [Blob(MEDIA_SADF_JSON, sadf)] if with_sadf else []
    source_hashes: dict[str, str] = {}
    for uri, data in geometry.items():
        layers.append(
            Blob(MEDIA_GEOMETRY_USD if uri.endswith(".usda") else MEDIA_GEOMETRY_GLTF, data)
        )
        source_hashes[uri] = _sha(data)
    manifest = PluginManifest(
        name=asset_id,
        version="0.1.0",
        kind=PluginKind.ASSET,
        capability_tags=list(tags),
        attributes={"asset_kind": kind, "asset_name": name},
        provenance=Provenance(digest=_sha(sadf), source_content_hashes=source_hashes),
    )
    HubClient(registry).publish(
        name=asset_id,
        version="0.1.0",
        kind="asset",
        manifest=manifest,
        layers=layers,
        private_key_pem=private_pem,
    )
    return f"{asset_id}:0.1.0"


@pytest.fixture
def keys() -> tuple[bytes, bytes]:
    return generate_keypair()


@pytest.fixture
def registry(tmp_path: Path, keys: tuple[bytes, bytes]) -> Registry:
    private_pem, _ = keys
    reg = Registry(tmp_path / "registry")
    _publish_asset(
        reg,
        private_pem,
        asset_id="relay-orbiter",
        kind="orbiter",
        name="Relay Orbiter",
        tags=[CapabilityTag("mobility.orbiter"), CapabilityTag("comms.relay")],
    )
    _publish_asset(
        reg,
        private_pem,
        asset_id="hopper",
        kind="hopper",
        name="Hopper Mk1",
        tags=[CapabilityTag("mobility.wheeled")],
        geometry={"geometry/hopper.glb": b"GLB-BYTES-123", "geometry/hopper.usda": b"USDA-BYTES"},
    )
    return reg


@pytest.fixture
def previewer(
    registry: Registry, keys: tuple[bytes, bytes], tmp_path: Path
) -> HubAssetPreviewMaterializer:
    _, public_pem = keys
    client = HubClient(registry, trusted_public_key_pem=public_pem)
    return HubAssetPreviewMaterializer(client, cache_dir=tmp_path / "assets")


# --- menu ------------------------------------------------------------------------


def test_catalog_satisfies_the_seam(registry: Registry) -> None:
    assert isinstance(HubAssetCatalog(registry), AssetCatalog)


def test_menu_lists_assets_by_vehicle_kind_with_tags(registry: Registry) -> None:
    menu = HubAssetCatalog(registry).list_assets()
    assert [m.reference for m in menu] == sorted(m.reference for m in menu)  # deterministic
    by_kind = {m.kind: m for m in menu}
    assert set(by_kind) == {"orbiter", "hopper"}  # the vehicle kind, never the plugin kind "asset"
    assert by_kind["orbiter"].name == "Relay Orbiter"
    assert "comms.relay" in by_kind["orbiter"].capability_tags


def test_menu_requires_filters_by_capability(registry: Registry) -> None:
    menu = HubAssetCatalog(registry).list_assets(requires=[CapabilityTag("mobility.wheeled")])
    assert [m.kind for m in menu] == ["hopper"]  # the orbiter is filtered out


def test_a_new_kind_appears_with_no_studio_edit(
    registry: Registry, keys: tuple[bytes, bytes]
) -> None:
    private_pem, _ = keys
    _publish_asset(
        registry,
        private_pem,
        asset_id="skycrane",
        kind="skycrane",
        name="Sky Crane",
        tags=[CapabilityTag("mobility.rocket")],
    )
    assert "skycrane" in {m.kind for m in HubAssetCatalog(registry).list_assets()}


def test_menu_lists_only_assets_not_other_artifact_kinds(
    registry: Registry, keys: tuple[bytes, bytes]
) -> None:
    # A Hub registry also holds worlds, resource fields, policies, comms models, … — none is a
    # selectable robot, and the preview materializer refuses a non-asset. The menu must filter them
    # out (else it offers rows that error "not an asset" the moment they are clicked).
    private_pem, _ = keys
    world = PluginManifest(name="a-world", version="0.1.0", kind=PluginKind.WORLD_PROVIDER)
    HubClient(registry).publish(
        name="a-world", version="0.1.0", kind="world", manifest=world, private_key_pem=private_pem
    )
    policy = PluginManifest(name="a-policy", version="0.1.0", kind=PluginKind.POLICY)
    HubClient(registry).publish(
        name="a-policy",
        version="0.1.0",
        kind="policy",
        manifest=policy,
        private_key_pem=private_pem,
    )

    menu = HubAssetCatalog(registry).list_assets()
    references = {m.reference for m in menu}
    assert "a-world:0.1.0" not in references
    assert "a-policy:0.1.0" not in references
    assert {m.kind for m in menu} == {"orbiter", "hopper"}  # only the fixture's real assets


# --- preview ---------------------------------------------------------------------


def test_previewer_satisfies_the_seam(previewer: HubAssetPreviewMaterializer) -> None:
    assert isinstance(previewer, AssetPreviewMaterializer)


def test_materializes_a_servable_document_and_geometry(
    previewer: HubAssetPreviewMaterializer,
) -> None:
    preview = previewer.preview(HOPPER)
    assert preview.digest.startswith("sha256:")
    assert preview.document_path.parent == preview.path  # documentUrl target at the served-dir root
    # Geometry blobs laid out at each ref's relative uri, byte-identical to the registry blobs.
    assert (preview.path / "geometry" / "hopper.glb").read_bytes() == b"GLB-BYTES-123"
    assert (preview.path / "geometry" / "hopper.usda").read_bytes() == b"USDA-BYTES"


def test_preview_cache_is_keyed_by_content_address(previewer: HubAssetPreviewMaterializer) -> None:
    first = previewer.preview(HOPPER)
    assert first.digest.replace(":", "-") == first.path.name
    stamp = (first.path / "geometry" / "hopper.glb").stat().st_mtime_ns
    second = previewer.preview(first.digest)  # re-preview by digest is free, not re-written
    assert second.path == first.path
    assert (first.path / "geometry" / "hopper.glb").stat().st_mtime_ns == stamp


def test_preview_of_a_geometry_less_asset_serves_just_the_document(
    previewer: HubAssetPreviewMaterializer,
) -> None:
    preview = previewer.preview(ORBITER)
    assert preview.document_path.is_file()
    assert list(preview.path.rglob("*.glb")) == []


def test_an_untrusted_signer_fails_closed(registry: Registry, tmp_path: Path) -> None:
    _, stranger_pem = generate_keypair()
    previewer = HubAssetPreviewMaterializer(
        HubClient(registry, trusted_public_key_pem=stranger_pem), cache_dir=tmp_path / "a"
    )
    with pytest.raises(PreviewError, match="refusing asset"):
        previewer.preview(HOPPER)


def test_a_tampered_layer_fails_closed(
    previewer: HubAssetPreviewMaterializer,
    registry: Registry,
    keys: tuple[bytes, bytes],
    tmp_path: Path,
) -> None:
    """The preview is refused, and — the sharper claim — the payload fetch itself refuses the bytes.

    The class-level refusal rides on the supply-chain re-verification, which would still hold if the
    geometry were pulled with the unchecked `registry.pull_blob()`. What makes *retrieval* safe is
    the content-address check inside `pull_payload`, so exercise it on its own with the supply-chain
    check switched off (hub.md §2.3; conventions.md §9).
    """
    _, public_pem = keys
    digest = registry.resolve(HOPPER).digest
    layer_digest = registry.read_manifest(digest)["layers"][0]["digest"]
    blob_path(registry.path, layer_digest).write_bytes(b"tampered")

    with pytest.raises(PreviewError):
        previewer.preview(HOPPER)
    assert not any((tmp_path / "assets").rglob("*.glb"))  # nothing unverified reached disk

    client = HubClient(registry, trusted_public_key_pem=public_pem)
    with pytest.raises(IntegrityError):
        client.pull_payload(HOPPER, verify=False)


def test_geometry_that_is_not_a_layer_of_the_artifact_is_refused(
    registry: Registry, keys: tuple[bytes, bytes], tmp_path: Path
) -> None:
    """Geometry must be a layer of *this* verified artifact, not any blob the store happens to hold.

    `registry.pull_blob()` resolved a `source_content_hashes` digest against the whole registry, so
    a manifest could name a blob that no verified manifest ever committed to — an unattested one —
    and Studio would serve it. Verified retrieval only knows the artifact's own layers, so the
    entry now fails closed. The stray blob is written into the store to prove the tightening is
    about attestation, not about a missing file.
    """
    private_pem, public_pem = keys
    sadf = json.dumps({"asset": {"identity": {"id": "ghost"}}}).encode()
    stray = b"UNATTESTED-GLB"
    blob_path(registry.path, _sha(stray)).parent.mkdir(parents=True, exist_ok=True)
    blob_path(registry.path, _sha(stray)).write_bytes(stray)

    manifest = PluginManifest(
        name="ghost",
        version="0.1.0",
        kind=PluginKind.ASSET,
        attributes={"asset_kind": "rover", "asset_name": "Ghost"},
        # Names the stray blob as geometry, but never publishes it as a layer.
        provenance=Provenance(
            digest=_sha(sadf), source_content_hashes={"geometry/ghost.glb": _sha(stray)}
        ),
    )
    HubClient(registry).publish(
        name="ghost",
        version="0.1.0",
        kind="asset",
        manifest=manifest,
        layers=[Blob(MEDIA_SADF_JSON, sadf)],
        private_key_pem=private_pem,
    )

    cache = tmp_path / "a"
    previewer = HubAssetPreviewMaterializer(
        HubClient(registry, trusted_public_key_pem=public_pem), cache_dir=cache
    )
    with pytest.raises(PreviewError, match="not a layer of the verified artifact"):
        previewer.preview("ghost:0.1.0")
    assert not any(cache.rglob("*.glb"))


def test_refuses_an_artifact_that_is_not_an_asset(
    previewer: HubAssetPreviewMaterializer, registry: Registry, keys: tuple[bytes, bytes]
) -> None:
    private_pem, _ = keys
    world = PluginManifest(name="a-world", version="0.1.0", kind=PluginKind.WORLD_PROVIDER)
    HubClient(registry).publish(
        name="a-world", version="0.1.0", kind="world", manifest=world, private_key_pem=private_pem
    )
    with pytest.raises(PreviewError, match="not an asset"):
        previewer.preview("a-world:0.1.0")


def test_refuses_an_asset_with_no_sadf_layer(
    registry: Registry, keys: tuple[bytes, bytes], tmp_path: Path
) -> None:
    private_pem, public_pem = keys
    _publish_asset(
        registry,
        private_pem,
        asset_id="hollow",
        kind="rover",
        name="Hollow",
        tags=[CapabilityTag("mobility.wheeled")],
        geometry={"geometry/x.glb": b"g"},
        with_sadf=False,
    )
    previewer = HubAssetPreviewMaterializer(
        HubClient(registry, trusted_public_key_pem=public_pem), cache_dir=tmp_path / "a"
    )
    with pytest.raises(PreviewError, match=r"carries no .* layer"):
        previewer.preview("hollow:0.1.0")


def test_a_path_traversal_geometry_uri_is_refused(
    registry: Registry, keys: tuple[bytes, bytes], tmp_path: Path
) -> None:
    private_pem, public_pem = keys
    _publish_asset(
        registry,
        private_pem,
        asset_id="evil",
        kind="rover",
        name="Evil",
        tags=[CapabilityTag("mobility.wheeled")],
        geometry={"../escaped.glb": b"PWNED"},
    )
    cache = tmp_path / "a"
    previewer = HubAssetPreviewMaterializer(
        HubClient(registry, trusted_public_key_pem=public_pem), cache_dir=cache
    )
    with pytest.raises(PreviewError, match="escapes its cache"):
        previewer.preview("evil:0.1.0")
    assert not (cache / "escaped.glb").exists()
