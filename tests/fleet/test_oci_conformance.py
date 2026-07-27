"""Hermetic OCI image-spec conformance of the produced layout (RM-P0-FLEET-06).

Validates the asset + signature-referrer manifests and the index against a faithful
subset of the OCI Image Manifest/Index v1.1 JSON Schemas. The *authoritative* check is
the `oras` step in `.github/workflows/ci.yml` (a real OCI client); this test runs
everywhere with no binary, so a spec regression is caught locally too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import jsonschema

from astro_mine.core.sadf import load_sadf
from astro_mine.fleet.packaging import oci, package_oci
from astro_mine.seal import generate_keypair

from .conftest import VALID_SADF

_DATA = Path(__file__).parent / "data"


def _schema(name: str) -> dict[str, Any]:
    return json.loads((_DATA / name).read_text(encoding="utf-8"))


def test_layout_conforms_to_oci_image_spec_schemas(tmp_path: Path) -> None:
    private_pem, _ = generate_keypair()
    artifact = package_oci(
        load_sadf(VALID_SADF), tmp_path / "oci", base_dir=tmp_path, sign_key=private_pem
    )
    manifest_schema = _schema("oci-image-manifest.schema.json")
    index_schema = _schema("oci-image-index.schema.json")

    index = oci.read_index(artifact.path)
    jsonschema.validate(index, index_schema)
    # Every referenced manifest -- the asset artifact and the signature referrer -- must
    # conform to the OCI image-manifest shape.
    for entry in index["manifests"]:
        manifest = json.loads(oci.read_blob(artifact.path, entry["digest"]))
        jsonschema.validate(manifest, manifest_schema)

    assert json.loads((artifact.path / "oci-layout").read_text()) == {"imageLayoutVersion": "1.0.0"}
