"""Core plugin-manifest construction from a SADF asset (RM-P0-FLEET-06)."""

from __future__ import annotations

from astro_mine.core.registry import PluginKind
from astro_mine.core.sadf import load_sadf
from astro_mine.fleet import __version__
from astro_mine.fleet.packaging.manifest import build_plugin_manifest

from .conftest import VALID_SADF

DIGEST = "sha256:" + "c" * 64


def test_builds_an_asset_manifest_from_identity() -> None:
    doc = load_sadf(VALID_SADF)
    manifest = build_plugin_manifest(doc, asset_digest=DIGEST)

    assert manifest.kind == PluginKind.ASSET
    assert manifest.name == "test.rover"
    assert manifest.version == "0.1.0"
    assert manifest.core_interfaces == {"sadf": "0.1.0"}
    assert manifest.attributes["asset_kind"] == "rover"
    assert manifest.signature is None  # attached to the OCI artifact separately


def test_provenance_records_the_content_digest_and_sources() -> None:
    doc = load_sadf(VALID_SADF)
    sources = {"mesh.glb": "sha256:" + "d" * 64}
    manifest = build_plugin_manifest(doc, asset_digest=DIGEST, source_content_hashes=sources)

    assert manifest.provenance is not None
    assert manifest.provenance.digest == DIGEST
    assert manifest.provenance.toolchain_version == __version__
    assert manifest.provenance.source_content_hashes == sources
