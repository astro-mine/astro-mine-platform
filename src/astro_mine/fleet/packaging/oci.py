"""Deterministic OCI image-layout writer for asset artifacts (RM-P0-FLEET-06).

Hand-rolled (no ``oras``/``oci`` runtime dependency) so the output is byte-stable and
content-addressed: the OCI image manifest is canonical JSON with no timestamps, so the
same asset always yields the same manifest digest and a re-pull by digest returns
identical bytes -- the acceptance property (``hub.md`` §2.1). The layout is spec-exact
**OCI image-layout v1.0.0** (``oci-layout`` + ``index.json`` + ``blobs/sha256/<hex>``)
so any OCI-native tool (``oras``, and Hub in P1) can push/pull it unchanged -- "Hub
publish" becomes an ``oras push`` of this same directory. The attached signature is a
Phase-0 astro-mine referrer (see :mod:`astro_mine.fleet.packaging.verifier`), **not** a
cosign artifact, so ``cosign verify`` does not apply to it yet -- the cosign upgrade is P1.

Media types follow the astro-mine artifact vocabulary (``hub.md`` §3): the asset artifact
is ``application/vnd.astro-mine.asset.v1`` with the Core plugin manifest as its config.
The signature (Core's cosign-*modeled* ``Signature`` envelope -- not a cosign artifact)
attaches as an OCI **referrer** (a manifest whose ``subject`` is the asset manifest),
matching Hub's Referrers-API model so a P1 cosign/SLSA/SBOM handler hangs attestations on
the same hook.

Backlog: RM-P0-FLEET-06 -- astro-mine-fleet#6
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "ARTIFACT_TYPE_ASSET",
    "ARTIFACT_TYPE_SIGNATURE",
    "MEDIA_CONFIG",
    "MEDIA_SADF_JSON",
    "MEDIA_SADF_WIRE",
    "MEDIA_SIGNATURE",
    "Blob",
    "Descriptor",
    "geometry_media_type",
    "load_config_and_signature",
    "read_asset_artifact",
    "read_blob",
    "read_index",
    "verify_asset_integrity",
    "verify_blob_integrity",
    "write_asset_artifact",
    "write_index",
    "write_signature_referrer",
]

# OCI image-spec v1.1 media types.
MEDIA_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
MEDIA_INDEX = "application/vnd.oci.image.index.v1+json"
MEDIA_EMPTY = "application/vnd.oci.empty.v1+json"

# astro-mine artifact vocabulary. `...asset.v1` is the type hub.md §3 publishes for SADF
# bundles; the finer media types below are the pre-Hub sub-vocabulary Fleet mints so the
# artifact is well-typed before Hub exists -- Hub (P1) formalizes these in hub.md §3/§5.
ARTIFACT_TYPE_ASSET = "application/vnd.astro-mine.asset.v1"
MEDIA_CONFIG = "application/vnd.astro-mine.asset.config.v1+json"
MEDIA_SADF_WIRE = "application/vnd.astro-mine.asset.sadf.wire.v1"
MEDIA_SADF_JSON = "application/vnd.astro-mine.asset.sadf.json.v1+json"
MEDIA_GEOMETRY_USD = "application/vnd.astro-mine.asset.geometry.usd.v1"
MEDIA_GEOMETRY_GLTF = "application/vnd.astro-mine.asset.geometry.gltf.v1"
ARTIFACT_TYPE_SIGNATURE = "application/vnd.astro-mine.signature.v1"
MEDIA_SIGNATURE = "application/vnd.astro-mine.signature.v1+json"

_LAYOUT_VERSION = "1.0.0"
_EMPTY_CONFIG = b"{}"


def geometry_media_type(fmt: str) -> str:
    """Map a SADF ``GeometryFormat`` (``usd``/``gltf``) to its layer media type."""
    return MEDIA_GEOMETRY_USD if fmt == "usd" else MEDIA_GEOMETRY_GLTF


def _canonical(obj: Any) -> bytes:
    """Byte-stable JSON: sorted keys, no whitespace -- so digests are reproducible."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()


@dataclass(frozen=True)
class Blob:
    """A payload to store: its media type and raw bytes."""

    media_type: str
    data: bytes


@dataclass(frozen=True)
class Descriptor:
    """An OCI content descriptor: ``{mediaType, digest, size}``."""

    media_type: str
    digest: str  # "sha256:<hex>"
    size: int

    def as_dict(self) -> dict[str, Any]:
        return {"mediaType": self.media_type, "digest": self.digest, "size": self.size}


def _digest(data: bytes) -> str:
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def _blobs_dir(layout: Path) -> Path:
    return layout / "blobs" / "sha256"


def _put(layout: Path, blob: Blob) -> Descriptor:
    """Write *blob* to ``blobs/sha256/<hex>`` (idempotent) and return its descriptor."""
    digest = _digest(blob.data)
    blobs = _blobs_dir(layout)
    blobs.mkdir(parents=True, exist_ok=True)
    path = blobs / digest.split(":", 1)[1]
    if not path.exists():
        path.write_bytes(blob.data)
    return Descriptor(blob.media_type, digest, len(blob.data))


def write_asset_artifact(
    layout: Path,
    *,
    config: Blob,
    layers: list[Blob],
    artifact_type: str = ARTIFACT_TYPE_ASSET,
    annotations: dict[str, str] | None = None,
) -> Descriptor:
    """Write the config + layer blobs and the OCI image manifest; return its descriptor."""
    config_desc = _put(layout, config)
    layer_descs = [_put(layout, blob) for blob in layers]
    manifest: dict[str, Any] = {
        "schemaVersion": 2,
        "mediaType": MEDIA_MANIFEST,
        "artifactType": artifact_type,
        "config": config_desc.as_dict(),
        "layers": [desc.as_dict() for desc in layer_descs],
    }
    if annotations:
        manifest["annotations"] = dict(annotations)
    return _put(layout, Blob(MEDIA_MANIFEST, _canonical(manifest)))


def write_signature_referrer(
    layout: Path,
    *,
    subject: Descriptor,
    signature: Blob,
    annotations: dict[str, str] | None = None,
) -> Descriptor:
    """Write a signature as an OCI referrer manifest whose ``subject`` is *subject*."""
    empty = _put(layout, Blob(MEDIA_EMPTY, _EMPTY_CONFIG))
    blob_desc = _put(layout, signature)
    manifest: dict[str, Any] = {
        "schemaVersion": 2,
        "mediaType": MEDIA_MANIFEST,
        "artifactType": ARTIFACT_TYPE_SIGNATURE,
        "config": empty.as_dict(),
        "layers": [blob_desc.as_dict()],
        "subject": subject.as_dict(),
    }
    if annotations:
        manifest["annotations"] = dict(annotations)
    return _put(layout, Blob(MEDIA_MANIFEST, _canonical(manifest)))


def write_index(layout: Path, manifests: list[tuple[Descriptor, dict[str, str] | None]]) -> None:
    """Write ``oci-layout`` and ``index.json`` referencing every top-level manifest."""
    layout.mkdir(parents=True, exist_ok=True)
    (layout / "oci-layout").write_bytes(_canonical({"imageLayoutVersion": _LAYOUT_VERSION}))
    entries: list[dict[str, Any]] = []
    for desc, annotation in manifests:
        entry = desc.as_dict()
        if annotation:
            entry["annotations"] = dict(annotation)
        entries.append(entry)
    index = {"schemaVersion": 2, "mediaType": MEDIA_INDEX, "manifests": entries}
    (layout / "index.json").write_bytes(_canonical(index))


def read_index(layout: Path) -> dict[str, Any]:
    """Parse ``index.json`` from an OCI layout."""
    index: dict[str, Any] = json.loads((layout / "index.json").read_bytes())
    return index


def read_blob(layout: Path, digest: str) -> bytes:
    """Return the bytes stored at *digest* (``sha256:<hex>``); raise ``KeyError`` if absent."""
    algorithm, _, hexpart = digest.partition(":")
    if algorithm != "sha256":
        raise ValueError(f"unsupported digest algorithm {algorithm!r}; expected 'sha256'")
    try:
        return (_blobs_dir(layout) / hexpart).read_bytes()
    except FileNotFoundError:
        raise KeyError(digest) from None


def _read_manifest(layout: Path, digest: str) -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(read_blob(layout, digest))
    return manifest


def read_asset_artifact(layout: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return ``(index-entry, image-manifest)`` for the asset artifact; raise if absent."""
    for entry in read_index(layout)["manifests"]:
        manifest = _read_manifest(layout, entry["digest"])
        if manifest.get("artifactType") == ARTIFACT_TYPE_ASSET:
            return entry, manifest
    raise ValueError("no astro-mine asset artifact in OCI layout")


def verify_blob_integrity(layout: Path, descriptor: dict[str, Any]) -> None:
    """Assert the stored blob hashes to the descriptor's digest (content-addressing on read).

    Content-addressing is enforced on *write*; a consumer pulling an artifact from an
    untrusted store must re-check it on read (``hub.md`` §2.3, "verify before trust").
    Raises ``ValueError`` on a mismatch.
    """
    digest = descriptor["digest"]
    actual = f"sha256:{hashlib.sha256(read_blob(layout, digest)).hexdigest()}"
    if actual != digest:
        raise ValueError(f"blob content does not match its digest {digest} (got {actual})")


def verify_asset_integrity(layout: Path, expected_wire_digest: str) -> None:
    """Assert the packaged content matches its addresses and the signed identity.

    Every blob (config + layers) must hash to its content address, and the SADF wire-form
    layer must equal *expected_wire_digest* (the signed ``provenance.digest``). This binds
    the signature -- which commits only to the digest *string* -- to the actual packaged
    bytes, so a tampered blob or a swapped wire layer is caught ("verify before trust",
    ``hub.md`` §2.3). Raises ``ValueError`` on any mismatch.
    """
    _, manifest = read_asset_artifact(layout)
    for descriptor in [manifest["config"], *manifest["layers"]]:
        verify_blob_integrity(layout, descriptor)
    wire = next(
        (layer for layer in manifest["layers"] if layer["mediaType"] == MEDIA_SADF_WIRE), None
    )
    if wire is None:
        raise ValueError("asset artifact has no SADF wire-form layer")
    if wire["digest"] != expected_wire_digest:
        raise ValueError(
            f"packaged SADF wire form {wire['digest']} does not match the signed digest "
            f"{expected_wire_digest}"
        )


def load_config_and_signature(
    layout: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Return the asset config (the plugin manifest) and its signature blob, if any.

    Walks ``index.json`` to find the asset artifact and, via the Referrers model, the
    signature manifest whose ``subject`` is that artifact. The signature is ``None`` for
    an unsigned layout.
    """
    asset_desc, asset_manifest = read_asset_artifact(layout)
    config: dict[str, Any] = json.loads(read_blob(layout, asset_manifest["config"]["digest"]))

    signature: dict[str, Any] | None = None
    for entry in read_index(layout)["manifests"]:
        manifest = _read_manifest(layout, entry["digest"])
        subject = manifest.get("subject")
        if (
            manifest.get("artifactType") == ARTIFACT_TYPE_SIGNATURE
            and isinstance(subject, dict)
            and subject.get("digest") == asset_desc["digest"]
        ):
            signature = json.loads(read_blob(layout, manifest["layers"][0]["digest"]))
            break
    return config, signature
