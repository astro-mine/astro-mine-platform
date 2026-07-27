"""OCI image-layout writer + package_oci determinism (RM-P0-FLEET-06)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from astro_mine.core.sadf import load_sadf
from astro_mine.fleet.packaging import oci, package_oci
from astro_mine.seal import generate_keypair

from .conftest import VALID_SADF

# A valid SADF asset that references one glTF geometry file (frame `base`).
GEOM_SADF = """\
sadf_version: "0.1"
asset:
  identity:
    id: test.rover
    name: Test Rover
    version: "0.1.0"
    kind: rover
  core_interface_versions:
    sadf: "0.1.0"
  root_frame: base
  frames:
    - name: base
  geometry:
    - role: visual
      format: gltf
      uri: mesh.glb
      frame: base
"""


def test_geometry_media_type() -> None:
    assert oci.geometry_media_type("usd") == oci.MEDIA_GEOMETRY_USD
    assert oci.geometry_media_type("gltf") == oci.MEDIA_GEOMETRY_GLTF


def test_layout_is_deterministic_and_repullable(tmp_path: Path) -> None:
    doc = load_sadf(VALID_SADF)
    a = package_oci(doc, tmp_path / "a", base_dir=tmp_path)
    b = package_oci(doc, tmp_path / "b", base_dir=tmp_path)

    assert a.digest == b.digest  # same OCI manifest digest across runs
    # "re-pull by digest yields byte-identical content"
    assert oci.read_blob(a.path, a.digest) == oci.read_blob(b.path, b.digest)

    assert (a.path / "oci-layout").exists()
    assert (a.path / "index.json").exists()
    manifest = json.loads(oci.read_blob(a.path, a.digest))
    assert manifest["artifactType"] == oci.ARTIFACT_TYPE_ASSET
    assert manifest["config"]["mediaType"] == oci.MEDIA_CONFIG
    assert any(layer["mediaType"] == oci.MEDIA_SADF_WIRE for layer in manifest["layers"])


def test_packaging_is_idempotent(tmp_path: Path) -> None:
    doc = load_sadf(VALID_SADF)
    first = package_oci(doc, tmp_path / "oci", base_dir=tmp_path)
    second = package_oci(doc, tmp_path / "oci", base_dir=tmp_path)  # re-put hits the exists path
    assert first.digest == second.digest


def test_signing_does_not_change_the_artifact_digest(tmp_path: Path) -> None:
    doc = load_sadf(VALID_SADF)
    private_pem, _ = generate_keypair()
    unsigned = package_oci(doc, tmp_path / "u", base_dir=tmp_path)
    signed = package_oci(doc, tmp_path / "s", base_dir=tmp_path, sign_key=private_pem)
    # The ECDSA signature lives in a referrer, never in the addressed manifest.
    assert unsigned.digest == signed.digest
    assert signed.signed and not unsigned.signed
    _, signature = oci.load_config_and_signature(signed.path)
    assert signature is not None and signature["scheme"] == "sigstore_cosign"


def test_local_geometry_becomes_a_layer(tmp_path: Path) -> None:
    (tmp_path / "mesh.glb").write_bytes(b"GLB-BYTES")
    doc = load_sadf(GEOM_SADF)
    artifact = package_oci(doc, tmp_path / "oci", base_dir=tmp_path)
    manifest = json.loads(oci.read_blob(artifact.path, artifact.digest))
    assert any(layer["mediaType"] == oci.MEDIA_GEOMETRY_GLTF for layer in manifest["layers"])
    config = json.loads(oci.read_blob(artifact.path, manifest["config"]["digest"]))
    assert "mesh.glb" in config["provenance"]["source_content_hashes"]


def test_package_without_base_dir_skips_geometry(tmp_path: Path) -> None:
    artifact = package_oci(load_sadf(VALID_SADF), tmp_path / "oci")  # base_dir=None
    assert artifact.asset_digest.startswith("sha256:")


def test_writer_annotation_variants(tmp_path: Path) -> None:
    asset = oci.write_asset_artifact(  # no annotations
        tmp_path, config=oci.Blob(oci.MEDIA_CONFIG, b"{}"), layers=[]
    )
    referrer = oci.write_signature_referrer(  # with annotations
        tmp_path,
        subject=asset,
        signature=oci.Blob(oci.MEDIA_SIGNATURE, b"{}"),
        annotations={"k": "v"},
    )
    assert asset.digest.startswith("sha256:")
    assert referrer.digest.startswith("sha256:")


def test_read_blob_errors(tmp_path: Path) -> None:
    oci.write_index(tmp_path, [])
    with pytest.raises(ValueError, match="unsupported digest algorithm"):
        oci.read_blob(tmp_path, "md5:deadbeef")
    with pytest.raises(KeyError):
        oci.read_blob(tmp_path, "sha256:" + "0" * 64)


def test_load_config_without_asset_raises(tmp_path: Path) -> None:
    oci.write_index(tmp_path, [])
    with pytest.raises(ValueError, match="no astro-mine asset artifact"):
        oci.load_config_and_signature(tmp_path)
