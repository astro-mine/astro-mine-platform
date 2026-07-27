"""Registry contract tests (RM-P1-HUB-01): immutability, referrers, integrity, GC, conformance."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema
import pytest

from astro_mine.hub.registry import (
    ARTIFACT_KINDS,
    ArtifactExistsError,
    ArtifactNotFound,
    Blob,
    Descriptor,
    IntegrityError,
    Registry,
    artifact_media_type,
)
from astro_mine.hub.registry._oci import (
    MEDIA_CORE_MANIFEST,
    put_blob,
    read_index,
)

_DATA = Path(__file__).parent / "data"

MANIFEST = {"name": "excavator-policy", "version": "1.2.0", "kind": "policy"}
PAYLOAD = Blob("application/vnd.astro-mine.onnx", b"\x00onnx-bytes\x00")


def _publish(reg: Registry, *, name: str = "excavator-policy", version: str = "1.2.0") -> str:
    config = {"name": name, "version": version, "kind": "policy"}
    art = reg.publish(name=name, version=version, kind="policy", config=config, layers=[PAYLOAD])
    return art.digest


# --- publish / resolve / pull ---------------------------------------------------------------


def test_publish_resolve_roundtrip_by_tag_and_digest(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    digest = _publish(reg)
    assert digest.startswith("sha256:")

    # tag -> the one immutable digest
    assert reg.resolve("excavator-policy:1.2.0").digest == digest
    # digest forms resolve to themselves
    assert reg.resolve(digest).digest == digest
    assert reg.resolve(f"excavator-policy@{digest}").digest == digest

    manifest = reg.read_manifest(digest)
    assert manifest["artifactType"] == artifact_media_type("policy")
    assert reg.pull_blob(manifest["layers"][0]["digest"]) == PAYLOAD.data
    assert json.loads(reg.read_config(digest)) == MANIFEST


def test_config_accepts_raw_bytes(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    art = reg.publish(
        name="w",
        version="0.1.0",
        kind="world",
        config=b'{"raw":true}',
        config_media_type=MEDIA_CORE_MANIFEST,
    )
    assert reg.read_config(art.digest) == b'{"raw":true}'


def test_publish_with_annotations(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    art = reg.publish(
        name="a",
        version="1.0.0",
        kind="plugin",
        config={"k": "v"},
        annotations={"org.opencontainers.image.title": "A"},
    )
    assert reg.read_manifest(art.digest)["annotations"]["org.opencontainers.image.title"] == "A"


def test_republish_is_rejected(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    _publish(reg)
    with pytest.raises(ArtifactExistsError):
        _publish(reg)  # same name:version


def test_new_version_is_allowed_and_dedups_layers(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    d1 = _publish(reg, version="1.2.0")
    d2 = _publish(reg, version="1.3.0")
    assert d1 != d2  # configs differ by version → different manifest digests
    # identical payload layer stored once (content-addressed dedup)
    hexes = [p.name for p in (reg.path / "blobs" / "sha256").iterdir()]
    assert len(hexes) == len(set(hexes))
    assert reg.versions("excavator-policy") == ["1.2.0", "1.3.0"]


def test_resolve_missing_raises(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    with pytest.raises(ArtifactNotFound):
        reg.resolve("nope:0.0.0")
    with pytest.raises(ArtifactNotFound):
        reg.resolve("sha256:" + "0" * 64)


@pytest.mark.parametrize("bad", ["nocolon", "name@notadigest", "name@", ":v", "n:"])
def test_malformed_reference_raises(tmp_path: Path, bad: str) -> None:
    reg = Registry(tmp_path / "reg")
    with pytest.raises(ValueError):
        reg.resolve(bad)


# --- referrers (attestations) ----------------------------------------------------------------


def test_attach_and_query_referrers(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    subject = _publish(reg)
    sig = reg.attach(
        subject=subject,
        artifact_type="application/vnd.astro-mine.signature.v1",
        blob=Blob("application/vnd.astro-mine.signature.v1+json", b'{"sig":"abc"}'),
        annotations={"tool": "cosign"},
    )
    sbom = reg.attach(
        subject=subject,
        artifact_type="application/vnd.astro-mine.sbom.v1",
        blob=Blob("application/vnd.cyclonedx+json", b"{}"),
    )

    all_refs = {d.digest for d in reg.referrers(subject)}
    assert all_refs == {sig.digest, sbom.digest}
    # filtered by artifactType
    only_sig = reg.referrers(subject, artifact_type="application/vnd.astro-mine.signature.v1")
    assert [d.digest for d in only_sig] == [sig.digest]
    # non-subject artifact has no referrers
    assert reg.referrers("sha256:" + "1" * 64) == []


def test_attach_to_missing_subject_raises(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    with pytest.raises(ArtifactNotFound):
        reg.attach(
            subject="sha256:" + "0" * 64,
            artifact_type="application/vnd.astro-mine.signature.v1",
            blob=Blob("application/json", b"{}"),
        )


# --- integrity -------------------------------------------------------------------------------


def test_verify_ok_then_detects_tamper(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    digest = _publish(reg)
    reg.verify(digest)  # clean

    # tamper: overwrite a layer blob's bytes so it no longer matches its content address
    layer_digest = reg.read_manifest(digest)["layers"][0]["digest"]
    from astro_mine.hub.registry._oci import blob_path

    blob_path(reg.path, layer_digest).write_bytes(b"TAMPERED")
    with pytest.raises(IntegrityError):
        reg.verify(digest)


def test_pull_missing_blob_raises(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    with pytest.raises(KeyError):
        reg.pull_blob("sha256:" + "0" * 64)


def test_blob_path_rejects_non_sha256(tmp_path: Path) -> None:
    from astro_mine.hub.registry._oci import blob_path

    with pytest.raises(ValueError):
        blob_path(tmp_path, "md5:abc")
    with pytest.raises(ValueError):
        blob_path(tmp_path, "sha256:")


# --- listing / descriptors -------------------------------------------------------------------


def test_references_and_descriptor_roundtrip(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    _publish(reg, name="a", version="1.0.0")
    _publish(reg, name="b", version="2.0.0")
    assert reg.references() == ["a:1.0.0", "b:2.0.0"]

    desc = reg.resolve("a:1.0.0")
    round_tripped = Descriptor.from_dict(desc.as_dict())
    assert round_tripped == desc
    assert desc.annotations is not None  # tag entry carries the ref annotation


# --- garbage collection ----------------------------------------------------------------------


def test_gc_reclaims_orphan_preserves_tagged_and_kept(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    tagged = _publish(reg)
    reg.attach(
        subject=tagged,
        artifact_type="application/vnd.astro-mine.signature.v1",
        blob=Blob("application/json", b'{"sig":"x"}'),
    )
    # an orphan blob (e.g. a partial/failed publish): referenced by nothing
    orphan = put_blob(reg.path, Blob("application/octet-stream", b"orphan-bytes")).digest

    reclaimed = reg.garbage_collect()
    assert orphan in reclaimed
    reg.verify(tagged)  # tagged artifact + its referrer chain survived
    assert reg.referrers(tagged)  # attestation preserved with its subject

    # a kept digest survives even when orphaned
    orphan2 = put_blob(reg.path, Blob("application/octet-stream", b"pinned-by-bench")).digest
    reclaimed2 = reg.garbage_collect(keep=[orphan2])
    assert orphan2 not in reclaimed2
    assert reg.pull_blob(orphan2) == b"pinned-by-bench"


def test_gc_visits_a_shared_digest_once(tmp_path: Path) -> None:
    # two tags with identical content resolve to the same manifest digest (content-addressed);
    # GC must visit that digest once and reclaim nothing.
    reg = Registry(tmp_path / "reg")
    a = reg.publish(name="x", version="1.0.0", kind="policy", config={"same": 1})
    b = reg.publish(name="y", version="1.0.0", kind="policy", config={"same": 1})
    assert a.digest == b.digest
    assert reg.garbage_collect() == []
    reg.verify(a.digest)


def test_gc_on_empty_registry_is_noop(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    assert reg.garbage_collect() == []


# --- media types -----------------------------------------------------------------------------


def test_artifact_media_type_vocabulary() -> None:
    for kind in ARTIFACT_KINDS:
        assert artifact_media_type(kind) == f"application/vnd.astro-mine.{kind}.v1"
    with pytest.raises(ValueError):
        artifact_media_type("nonsense")


def test_studio_artifact_kinds_are_storable(tmp_path: Path) -> None:
    """RFC-0008: a frozen Studio design/campaign is an ordinary content-addressed artifact.

    Hub stores only the kinds Core describes (hub.md §2 principle 2). That invariant holds *by
    construction* rather than by assertion here: publishing needs a ``PluginManifest``, and a Core
    without ``PluginKind.CAMPAIGN`` cannot build one. This test pins the half Hub owns — that the
    vocabulary yields the media type, and that such an artifact round-trips through the registry.
    """
    for kind in ("design", "campaign"):
        assert artifact_media_type(kind) == f"application/vnd.astro-mine.{kind}.v1"

    reg = Registry(tmp_path / "reg")
    payload = Blob("application/vnd.astro-mine.campaign.bundle.v1.json", b'{"id":"c1"}')
    artifact = reg.publish(
        name="lunar-ice-campaign",
        version="0.1.0",
        kind="campaign",
        config={"name": "lunar-ice-campaign", "version": "0.1.0", "kind": "campaign"},
        layers=[payload],
    )
    manifest = reg.read_manifest(artifact.digest)
    assert manifest["artifactType"] == "application/vnd.astro-mine.campaign.v1"
    assert reg.pull_blob(manifest["layers"][0]["digest"]) == payload.data
    reg.verify(artifact.digest)


# --- OCI image-spec conformance --------------------------------------------------------------


def _schema(name: str) -> dict[str, Any]:
    return json.loads((_DATA / name).read_text(encoding="utf-8"))


def test_layout_conforms_to_oci_image_spec(tmp_path: Path) -> None:
    reg = Registry(tmp_path / "reg")
    subject = _publish(reg)
    reg.attach(
        subject=subject,
        artifact_type="application/vnd.astro-mine.signature.v1",
        blob=Blob("application/json", b'{"sig":"x"}'),
    )
    manifest_schema = _schema("oci-image-manifest.schema.json")
    index_schema = _schema("oci-image-index.schema.json")

    index = read_index(reg.path)
    jsonschema.validate(index, index_schema)
    for entry in index["manifests"]:
        jsonschema.validate(reg.read_manifest(entry["digest"]), manifest_schema)

    assert json.loads((reg.path / "oci-layout").read_text()) == {"imageLayoutVersion": "1.0.0"}
