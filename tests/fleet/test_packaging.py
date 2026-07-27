"""Content-addressed packaging (RM-P0-FLEET-01)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from astro_mine.core.sadf import from_wire, load_sadf, to_wire
from astro_mine.fleet import __version__
from astro_mine.fleet.packaging import BUNDLE_SCHEMA, package_asset

from .conftest import VALID_SADF


def test_bundle_layout_and_digest(tmp_path: Path) -> None:
    doc = load_sadf(VALID_SADF)
    bundle = package_asset(doc, tmp_path)

    expected = hashlib.sha256(to_wire(doc)).hexdigest()
    assert bundle.digest == f"sha256:{expected}"
    assert bundle.path == tmp_path / "sha256" / expected
    assert (bundle.path / "asset.sadf.pb").exists()
    assert (bundle.path / "asset.sadf.json").exists()
    assert (bundle.path / "manifest.json").exists()


def test_wire_round_trips_to_the_same_document(tmp_path: Path) -> None:
    doc = load_sadf(VALID_SADF)
    bundle = package_asset(doc, tmp_path)
    restored = from_wire((bundle.path / "asset.sadf.pb").read_bytes())
    assert restored.asset.identity.id == doc.asset.identity.id


def test_manifest_fields_are_deterministic(tmp_path: Path) -> None:
    doc = load_sadf(VALID_SADF)
    bundle = package_asset(doc, tmp_path)
    manifest = json.loads((bundle.path / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["schema"] == BUNDLE_SCHEMA
    assert manifest["digest"] == bundle.digest
    assert manifest["asset_id"] == "test.rover"
    assert manifest["asset_version"] == "0.1.0"
    assert manifest["asset_kind"] == "rover"
    assert manifest["sadf_version"] == "0.1"
    assert manifest["core_interface_versions"] == {"sadf": "0.1.0"}
    assert manifest["toolchain_version"] == __version__
    # Content-addressed → no wall-clock fields may leak into the manifest.
    assert not any("time" in key or "date" in key for key in manifest)


def test_packaging_is_idempotent_and_byte_identical(tmp_path: Path) -> None:
    doc = load_sadf(VALID_SADF)
    first = package_asset(doc, tmp_path)
    first_bytes = {p.name: p.read_bytes() for p in first.path.iterdir()}

    second = package_asset(doc, tmp_path)  # same out dir, same doc
    assert second.digest == first.digest
    assert second.path == first.path
    for name, data in first_bytes.items():
        assert (second.path / name).read_bytes() == data


def test_distinct_documents_get_distinct_digests(tmp_path: Path) -> None:
    a = load_sadf(VALID_SADF)
    b = load_sadf(VALID_SADF.replace("test.rover", "test.hauler"))
    assert package_asset(a, tmp_path).digest != package_asset(b, tmp_path).digest
