"""Content-integrity ("verify before trust") on the verify path (RM-P0-FLEET-06).

The signature commits only to the digest *string*; these tests pin the binding from that
digest to the actual packaged bytes, so a tampered blob or a swapped wire layer is caught
before the manifest loads -- not just a self-inconsistent claim.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astro_mine.core.sadf import load_sadf
from astro_mine.fleet.packaging import oci, package_oci

from .conftest import VALID_SADF




def _wire_blob_path(layout: Path) -> Path:
    _, manifest = oci.read_asset_artifact(layout)
    wire = next(layer for layer in manifest["layers"] if layer["mediaType"] == oci.MEDIA_SADF_WIRE)
    return layout / "blobs" / "sha256" / wire["digest"].split(":", 1)[1]


def test_intact_artifact_passes(tmp_path: Path) -> None:
    artifact = package_oci(load_sadf(VALID_SADF), tmp_path / "oci", base_dir=tmp_path)
    oci.verify_asset_integrity(artifact.path, artifact.asset_digest)  # no raise


def test_corrupted_blob_is_detected(tmp_path: Path) -> None:
    artifact = package_oci(load_sadf(VALID_SADF), tmp_path / "oci", base_dir=tmp_path)
    _wire_blob_path(artifact.path).write_bytes(b"tampered")
    with pytest.raises(ValueError, match="does not match its digest"):
        oci.verify_asset_integrity(artifact.path, artifact.asset_digest)


def test_wire_form_must_match_the_signed_digest(tmp_path: Path) -> None:
    artifact = package_oci(load_sadf(VALID_SADF), tmp_path / "oci", base_dir=tmp_path)
    # All blobs are content-valid, but the signed identity we check against is a different
    # digest -> the wire <-> signature binding must reject it.
    with pytest.raises(ValueError, match="does not match the signed digest"):
        oci.verify_asset_integrity(artifact.path, "sha256:" + "1" * 64)


def test_missing_wire_layer_is_rejected(tmp_path: Path) -> None:
    layout = tmp_path / "oci"
    desc = oci.write_asset_artifact(
        layout,
        config=oci.Blob(oci.MEDIA_CONFIG, b"{}"),
        layers=[oci.Blob(oci.MEDIA_SADF_JSON, b"{}")],
    )
    oci.write_index(layout, [(desc, None)])
    with pytest.raises(ValueError, match="no SADF wire-form layer"):
        oci.verify_asset_integrity(layout, "sha256:" + "0" * 64)


