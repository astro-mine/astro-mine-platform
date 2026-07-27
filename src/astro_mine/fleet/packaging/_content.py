"""The content-addressed asset payload shared by the OCI and Hub publish paths.

Both the pre-Hub local OCI layout (:func:`astro_mine.fleet.packaging.package_oci`) and the
Hub publish path (:mod:`astro_mine.fleet.packaging.hub`) push the **same** bytes: the Core
plugin manifest as the artifact config, and content-addressed layers = the canonical SADF
wire form, its JSON projection, and any resolvable geometry. Building that payload once here
guarantees the two paths are byte-identical — a Hub-published asset and a locally packaged
one carry the same digest (``fleet.md`` §5; ``hub.md`` §2.1).

The SADF **JSON** layer keeps the :data:`~astro_mine.fleet.packaging.oci.MEDIA_SADF_JSON`
media type so [Sim](sim.md) rehydrates the Asset from it via :func:`astro_mine.core.sadf.load_sadf`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from astro_mine.core.registry import PluginManifest
from astro_mine.core.sadf import SadfDocument, to_wire
from astro_mine.fleet._core import canonical_json
from astro_mine.fleet.packaging import oci
from astro_mine.fleet.packaging.manifest import build_plugin_manifest

__all__ = ["AssetContent", "build_asset_content"]


@dataclass(frozen=True)
class AssetContent:
    """The content-addressed payload of an asset: its manifest + layers + signed identity."""

    asset_digest: str  # sha256 of the SADF wire form -- the signed content identity
    layers: tuple[oci.Blob, ...]  # SADF wire + SADF JSON (+ geometry), content-addressed
    source_content_hashes: dict[str, str]  # geometry uri -> sha256 (provenance)
    manifest: PluginManifest  # the Core kind=asset plugin manifest (artifact config)


def build_asset_content(doc: SadfDocument, base_dir: str | Path | None = None) -> AssetContent:
    """Build the manifest + content-addressed layers for *doc* (geometry resolved under *base_dir*).

    Deterministic: the same document yields the same asset digest and byte-identical layers,
    so re-packaging (locally or to Hub) is reproducible.
    """
    wire = to_wire(doc)
    asset_digest = f"sha256:{hashlib.sha256(wire).hexdigest()}"
    json_text = canonical_json(doc) + "\n"

    layers = [
        oci.Blob(oci.MEDIA_SADF_WIRE, wire),
        oci.Blob(oci.MEDIA_SADF_JSON, json_text.encode("utf-8")),
    ]
    source_hashes: dict[str, str] = {}
    if base_dir is not None:
        for ref in doc.asset.geometry:
            geom = Path(base_dir) / ref.uri
            if geom.is_file():
                data = geom.read_bytes()
                layers.append(oci.Blob(oci.geometry_media_type(ref.format), data))
                source_hashes[ref.uri] = f"sha256:{hashlib.sha256(data).hexdigest()}"

    manifest = build_plugin_manifest(
        doc, asset_digest=asset_digest, source_content_hashes=source_hashes
    )
    return AssetContent(asset_digest, tuple(layers), source_hashes, manifest)
