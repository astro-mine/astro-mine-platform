"""RM-P1-STUDIO-06 — materializing a Worlds bundle pulled from Hub by digest.

The embedded View fetches ``world.json`` over HTTP, and the Phase-2 View Gateway that would proxy
tiles does not exist. Studio therefore serves a bundle it pulled by content hash and verified
client-side. It stores a *reference*; the cache is disposable (studio.md §5).
"""

from __future__ import annotations

import io
import json
import os
import tarfile
from pathlib import Path

import pytest

from astro_mine.core.registry import PluginKind, PluginManifest
from astro_mine.hub.client import HubClient
from astro_mine.hub.registry import Blob, IntegrityError, Registry
from astro_mine.hub.registry._oci import blob_path
from astro_mine.hub.supply_chain import generate_keypair
from astro_mine.studio.hub import (
    HubWorldMaterializer,
    MaterializeError,
    WorldMaterializer,
)

BUNDLE_MEDIA_TYPE = "application/vnd.astro-mine.world.bundle.v1.tar"
WORLD_REF = "shackleton-de-gerlache-v1:0.2.0"


def _bundle_tar() -> bytes:
    """A minimal Worlds bundle: the manifest View fetches, and the tileset it names."""
    files = {
        "world.json": json.dumps(
            {
                "world_id": "shackleton-de-gerlache-v1",
                "tiles": "tiles/tileset.json",
                "tiles_anchor": {"frame": "MOON_ME", "origin": {"height_m": -984.9}},
            }
        ).encode(),
        "tiles/tileset.json": b'{"asset": {"version": "1.1"}}',
        "tiles/terrain.glb": b"glTF-bytes",
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


@pytest.fixture
def keys() -> tuple[bytes, bytes]:
    return generate_keypair()


@pytest.fixture
def registry(tmp_path: Path, keys: tuple[bytes, bytes]) -> Registry:
    private_pem, _ = keys
    reg = Registry(tmp_path / "registry")
    manifest = PluginManifest(
        name="shackleton-de-gerlache-v1",
        version="0.2.0",
        kind=PluginKind.WORLD_PROVIDER,
        core_interfaces={"world_provider": "0.1.0"},
        attributes={
            "world_id": "shackleton-de-gerlache-v1",
            "bundle_media_type": BUNDLE_MEDIA_TYPE,
        },
    )
    HubClient(reg).publish(
        name=manifest.name,
        version=manifest.version,
        kind="world",
        manifest=manifest,
        layers=[Blob(BUNDLE_MEDIA_TYPE, _bundle_tar())],
        private_key_pem=private_pem,
    )
    return reg


@pytest.fixture
def materializer(
    registry: Registry, keys: tuple[bytes, bytes], tmp_path: Path
) -> HubWorldMaterializer:
    _, public_pem = keys
    client = HubClient(registry, trusted_public_key_pem=public_pem)
    return HubWorldMaterializer(client, cache_dir=tmp_path / "worlds")


def test_materializer_satisfies_the_seam(materializer: HubWorldMaterializer) -> None:
    assert isinstance(materializer, WorldMaterializer)


def test_materializes_a_verified_bundle_the_viewer_can_fetch(
    materializer: HubWorldMaterializer,
) -> None:
    world = materializer.materialize(WORLD_REF)

    assert world.digest.startswith("sha256:")
    assert world.world_id == "shackleton-de-gerlache-v1"
    assert world.manifest_path.is_file()
    assert (world.path / "tiles" / "tileset.json").is_file()
    assert (world.path / "tiles" / "terrain.glb").read_bytes() == b"glTF-bytes"

    # `<GlobeScene world={{ manifestUrl }}>` needs the anchor RM-P1-WORLDS-16 publishes.
    manifest = json.loads(world.manifest_path.read_text())
    assert "tiles_anchor" in manifest


def test_the_cache_is_keyed_by_content_address(materializer: HubWorldMaterializer) -> None:
    first = materializer.materialize(WORLD_REF)
    # A digest is a directory-safe name, and re-materializing is free rather than re-extracting.
    assert first.digest.replace(":", "-") == first.path.name

    stamp = (first.path / "tiles" / "terrain.glb").stat().st_mtime_ns
    second = materializer.materialize(first.digest)
    assert second.path == first.path
    assert (first.path / "tiles" / "terrain.glb").stat().st_mtime_ns == stamp


def test_an_untrusted_signer_fails_closed(registry: Registry, tmp_path: Path) -> None:
    _, stranger_pem = generate_keypair()
    materializer = HubWorldMaterializer(
        HubClient(registry, trusted_public_key_pem=stranger_pem), cache_dir=tmp_path / "w"
    )
    with pytest.raises(MaterializeError, match="refusing world"):
        materializer.materialize(WORLD_REF)
    assert not (tmp_path / "w").exists() or not any((tmp_path / "w").iterdir())


def test_a_tampered_bundle_fails_closed(
    materializer: HubWorldMaterializer,
    registry: Registry,
    keys: tuple[bytes, bytes],
    tmp_path: Path,
) -> None:
    """Two independent guards, and the second one is the point of this test.

    The materializer refuses the world (the supply-chain re-verification catches the swapped blob),
    and nothing is unpacked. But that check is not what makes *payload retrieval* safe: the bytes
    now come back through Hub's verified path, which re-hashes the layer against the digest the
    manifest commits to. Assert that directly, with the supply-chain check switched off, so the
    content-address enforcement is exercised on its own — `registry.pull_blob()` (a plain
    `read_bytes()` on a local OCI layout) would have handed these bytes straight to the unpacker
    (hub.md §2.3; conventions.md §9).
    """
    _, public_pem = keys
    digest = registry.resolve(WORLD_REF).digest
    layer_digest = registry.read_manifest(digest)["layers"][0]["digest"]
    blob_path(registry.path, layer_digest).write_bytes(b"not a tar")

    with pytest.raises(MaterializeError):
        materializer.materialize(WORLD_REF)
    assert not any((tmp_path / "worlds").rglob("world.json"))  # nothing unverified reached disk

    client = HubClient(registry, trusted_public_key_pem=public_pem)
    with pytest.raises(IntegrityError):
        client.pull_layer(WORLD_REF, layer_digest, verify=False)


def test_refuses_an_artifact_that_is_not_a_world(
    materializer: HubWorldMaterializer, registry: Registry, keys: tuple[bytes, bytes]
) -> None:
    private_pem, _ = keys
    policy = PluginManifest(name="a-policy", version="0.1.0", kind=PluginKind.POLICY)
    HubClient(registry).publish(
        name="a-policy",
        version="0.1.0",
        kind="policy",
        manifest=policy,
        private_key_pem=private_pem,
    )
    with pytest.raises(MaterializeError, match="not a world_provider"):
        materializer.materialize("a-policy:0.1.0")


def test_refuses_a_bundle_with_no_world_manifest(
    registry: Registry, keys: tuple[bytes, bytes], tmp_path: Path
) -> None:
    private_pem, public_pem = keys
    empty = io.BytesIO()
    with tarfile.open(fileobj=empty, mode="w"):
        pass
    manifest = PluginManifest(
        name="hollow",
        version="0.1.0",
        kind=PluginKind.WORLD_PROVIDER,
        attributes={"bundle_media_type": BUNDLE_MEDIA_TYPE},
    )
    HubClient(registry).publish(
        name="hollow",
        version="0.1.0",
        kind="world",
        manifest=manifest,
        layers=[Blob(BUNDLE_MEDIA_TYPE, empty.getvalue())],
        private_key_pem=private_pem,
    )
    materializer = HubWorldMaterializer(
        HubClient(registry, trusted_public_key_pem=public_pem), cache_dir=tmp_path / "w"
    )
    with pytest.raises(MaterializeError, match="nothing to fetch"):
        materializer.materialize("hollow:0.1.0")


def test_path_traversal_in_a_bundle_is_refused(
    registry: Registry, keys: tuple[bytes, bytes], tmp_path: Path
) -> None:
    """`filter="data"` is the guard: a bundle cannot write outside its cache directory."""
    private_pem, public_pem = keys
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        info = tarfile.TarInfo("../escaped.txt")
        info.size = 3
        archive.addfile(info, io.BytesIO(b"bad"))
    manifest = PluginManifest(
        name="evil",
        version="0.1.0",
        kind=PluginKind.WORLD_PROVIDER,
        attributes={"bundle_media_type": BUNDLE_MEDIA_TYPE},
    )
    HubClient(registry).publish(
        name="evil",
        version="0.1.0",
        kind="world",
        manifest=manifest,
        layers=[Blob(BUNDLE_MEDIA_TYPE, buffer.getvalue())],
        private_key_pem=private_pem,
    )
    materializer = HubWorldMaterializer(
        HubClient(registry, trusted_public_key_pem=public_pem), cache_dir=tmp_path / "w"
    )
    with pytest.raises(MaterializeError, match="unsafe path"):
        materializer.materialize("evil:0.1.0")
    assert not (tmp_path / "escaped.txt").exists()


@pytest.mark.skipif(
    not os.environ.get("ASTRO_MINE_HUB_REGISTRY"),
    reason="ASTRO_MINE_HUB_REGISTRY not set; opt-in check against the workspace registry",
)
def test_materializes_the_real_anchor_world(tmp_path: Path) -> None:
    """Opt-in: the actual published anchor world, verified against the workspace's signing key."""
    registry_path = Path(os.environ["ASTRO_MINE_HUB_REGISTRY"])
    public_pem = (registry_path / "keys" / "anchor-dev.pub.pem").read_bytes()
    materializer = HubWorldMaterializer(
        HubClient(Registry(registry_path), trusted_public_key_pem=public_pem),
        cache_dir=tmp_path / "worlds",
    )
    world = materializer.materialize(WORLD_REF)
    manifest = json.loads(world.manifest_path.read_text())
    assert "tiles_anchor" in manifest, "the published bundle predates RM-P1-WORLDS-16"
    assert (world.path / "tiles" / "terrain.glb").is_file()


def test_a_layer_that_is_not_a_tar_is_refused(
    registry: Registry, keys: tuple[bytes, bytes], tmp_path: Path
) -> None:
    """The blob hashes to its content address, and is still not a bundle."""
    private_pem, public_pem = keys
    manifest = PluginManifest(
        name="rubbish",
        version="0.1.0",
        kind=PluginKind.WORLD_PROVIDER,
        attributes={"bundle_media_type": BUNDLE_MEDIA_TYPE},
    )
    HubClient(registry).publish(
        name="rubbish",
        version="0.1.0",
        kind="world",
        manifest=manifest,
        layers=[Blob(BUNDLE_MEDIA_TYPE, b"definitely not a tar")],
        private_key_pem=private_pem,
    )
    materializer = HubWorldMaterializer(
        HubClient(registry, trusted_public_key_pem=public_pem), cache_dir=tmp_path / "w"
    )
    with pytest.raises(MaterializeError, match="not a readable bundle tar"):
        materializer.materialize("rubbish:0.1.0")


def test_a_bundle_whose_declared_media_type_is_absent_is_refused(
    registry: Registry, keys: tuple[bytes, bytes], tmp_path: Path
) -> None:
    private_pem, public_pem = keys
    manifest = PluginManifest(
        name="mismatched",
        version="0.1.0",
        kind=PluginKind.WORLD_PROVIDER,
        attributes={"bundle_media_type": "application/vnd.astro-mine.world.bundle.v2.tar"},
    )
    HubClient(registry).publish(
        name="mismatched",
        version="0.1.0",
        kind="world",
        manifest=manifest,
        layers=[Blob(BUNDLE_MEDIA_TYPE, _bundle_tar())],
        private_key_pem=private_pem,
    )
    materializer = HubWorldMaterializer(
        HubClient(registry, trusted_public_key_pem=public_pem), cache_dir=tmp_path / "w"
    )
    with pytest.raises(MaterializeError, match=r"carries no .* layer"):
        materializer.materialize("mismatched:0.1.0")


def test_a_bundle_predating_the_media_type_attribute_uses_its_only_layer(
    registry: Registry, keys: tuple[bytes, bytes], tmp_path: Path
) -> None:
    """`bundle_media_type` post-dates the first world bundles; one layer is unambiguous."""
    private_pem, public_pem = keys
    manifest = PluginManifest(name="old", version="0.1.0", kind=PluginKind.WORLD_PROVIDER)
    HubClient(registry).publish(
        name="old",
        version="0.1.0",
        kind="world",
        manifest=manifest,
        layers=[Blob(BUNDLE_MEDIA_TYPE, _bundle_tar())],
        private_key_pem=private_pem,
    )
    materializer = HubWorldMaterializer(
        HubClient(registry, trusted_public_key_pem=public_pem), cache_dir=tmp_path / "w"
    )
    world = materializer.materialize("old:0.1.0")
    assert world.world_id == "old"  # falls back to the manifest name
    assert world.manifest_path.is_file()


def test_a_world_with_no_layers_is_refused(
    registry: Registry, keys: tuple[bytes, bytes], tmp_path: Path
) -> None:
    private_pem, public_pem = keys
    manifest = PluginManifest(name="bare", version="0.1.0", kind=PluginKind.WORLD_PROVIDER)
    HubClient(registry).publish(
        name="bare", version="0.1.0", kind="world", manifest=manifest, private_key_pem=private_pem
    )
    materializer = HubWorldMaterializer(
        HubClient(registry, trusted_public_key_pem=public_pem), cache_dir=tmp_path / "w"
    )
    with pytest.raises(MaterializeError, match="declares no layers"):
        materializer.materialize("bare:0.1.0")
