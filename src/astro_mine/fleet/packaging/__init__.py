"""Content-addressed asset packaging (RM-P0-FLEET-01; signing + OCI in RM-P0-FLEET-06).

``fleet package`` turns a validated SADF document into a **content-addressed** bundle:
the asset's canonical, byte-stable Protobuf wire form (:func:`astro_mine.core.sadf.to_wire`)
is hashed with SHA-256, and the bundle is written under ``sha256/<digest>/``. The same
document always produces the same digest and the same bytes, so a bundle re-pulled by
digest is byte-identical — the reproducibility property Bench relies on (conventions.md
§5) and the foundation the OCI/cosign layer builds on next.

Scope boundary: this issue ships the content-addressing primitive. **Signing
(Sigstore/cosign), OCI media types, and the object-store push** are RM-P0-FLEET-06
(issue #6); they wrap — they do not replace — the bundle produced here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from astro_mine.core.registry import PluginManifest, Signature
from astro_mine.core.sadf import SadfDocument, to_wire
from astro_mine.fleet import __version__ as _FLEET_VERSION
from astro_mine.fleet._core import CORE_INTERFACES, canonical_json
from astro_mine.fleet.packaging import oci
from astro_mine.fleet.packaging._content import build_asset_content
from astro_mine.fleet.packaging.verifier import sign_asset

__all__ = ["BUNDLE_SCHEMA", "AssetBundle", "OciArtifact", "package_asset", "package_oci"]

#: Schema id stamped into every manifest so a reader can identify the bundle format.
BUNDLE_SCHEMA = "astro-mine-fleet/asset-bundle/v0.1"

_WIRE_NAME = "asset.sadf.pb"
_JSON_NAME = "asset.sadf.json"
_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class AssetBundle:
    """Descriptor for a written asset bundle."""

    digest: str  # "sha256:<hex>"
    path: Path  # the sha256/<hex>/ directory the bundle was written to
    manifest: dict[str, object]


def _manifest(doc: SadfDocument, digest: str) -> dict[str, object]:
    """Build the deterministic bundle manifest (no timestamps — content-addressed)."""
    identity = doc.asset.identity
    return {
        "schema": BUNDLE_SCHEMA,
        "digest": digest,
        "asset_id": identity.id,
        "asset_version": identity.version,
        "asset_kind": identity.kind,
        "sadf_version": doc.sadf_version,
        "core_interface_versions": dict(CORE_INTERFACES),
        "toolchain_version": _FLEET_VERSION,
        "files": {"wire": _WIRE_NAME, "json": _JSON_NAME},
    }


def package_asset(doc: SadfDocument, out_dir: str | Path) -> AssetBundle:
    """Write ``doc`` as a content-addressed bundle under ``out_dir`` and return it.

    The bundle directory is ``<out_dir>/sha256/<hex>/`` containing the canonical wire
    form, a human-readable JSON projection, and a deterministic manifest. Writing is
    idempotent: the same document yields the same digest and identical bytes.
    """
    wire = to_wire(doc)
    hexdigest = hashlib.sha256(wire).hexdigest()
    digest = f"sha256:{hexdigest}"

    bundle_dir = Path(out_dir) / "sha256" / hexdigest
    bundle_dir.mkdir(parents=True, exist_ok=True)

    manifest = _manifest(doc, digest)
    (bundle_dir / _WIRE_NAME).write_bytes(wire)
    (bundle_dir / _JSON_NAME).write_text(canonical_json(doc) + "\n", encoding="utf-8")
    manifest_text = json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    (bundle_dir / _MANIFEST_NAME).write_text(manifest_text, encoding="utf-8")

    return AssetBundle(digest=digest, path=bundle_dir, manifest=manifest)


@dataclass(frozen=True)
class OciArtifact:
    """Descriptor for a written OCI asset artifact."""

    digest: str  # the OCI image-manifest digest -- the distribution address (re-pull by this)
    asset_digest: str  # sha256 of the SADF wire form -- the signed content identity
    path: Path  # the OCI image-layout directory
    signed: bool


def _canonical_manifest(manifest: PluginManifest) -> bytes:
    return json.dumps(
        manifest.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()


def _canonical_signature(signature: Signature) -> bytes:
    return json.dumps(
        signature.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    ).encode()


def package_oci(
    doc: SadfDocument,
    out_dir: str | Path,
    *,
    base_dir: str | Path | None = None,
    sign_key: bytes | None = None,
) -> OciArtifact:
    """Write ``doc`` as a (optionally signed) content-addressed OCI artifact.

    The OCI image manifest wraps the same content-addressed payload as
    :func:`package_asset` -- the SADF wire form and its JSON projection as layers, the
    Core plugin manifest as config -- plus any geometry resolvable under *base_dir*.
    Output is deterministic: re-packaging the same document yields the same OCI digest
    and byte-identical blobs. With *sign_key* (an ECDSA P-256 private-key PEM) the
    artifact carries a signature (Core's cosign-modeled ``Signature`` envelope, not a
    cosign artifact) as an OCI referrer.
    """
    content = build_asset_content(doc, base_dir)
    asset_digest = content.asset_digest
    config = oci.Blob(oci.MEDIA_CONFIG, _canonical_manifest(content.manifest))

    layout = Path(out_dir)
    title = f"{doc.asset.identity.id}:{doc.asset.identity.version}"
    asset_desc = oci.write_asset_artifact(
        layout,
        config=config,
        layers=list(content.layers),
        annotations={"org.opencontainers.image.title": title},
    )
    entries: list[tuple[oci.Descriptor, dict[str, str] | None]] = [
        (asset_desc, {"org.opencontainers.image.ref.name": title})
    ]
    signed = False
    if sign_key is not None:
        signature = sign_asset(asset_digest, sign_key)
        sig_desc = oci.write_signature_referrer(
            layout,
            subject=asset_desc,
            signature=oci.Blob(oci.MEDIA_SIGNATURE, _canonical_signature(signature)),
        )
        entries.append((sig_desc, None))
        signed = True
    oci.write_index(layout, entries)
    return OciArtifact(
        digest=asset_desc.digest, asset_digest=asset_digest, path=layout, signed=signed
    )
