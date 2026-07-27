"""Spec-exact OCI image-layout primitives — the content-addressed substrate (RM-P1-HUB-01).

Hand-rolled (no ``oras``/``oci`` runtime dependency) so the output is **byte-stable and
content-addressed**: manifests are canonical JSON with no timestamps, so the same artifact always
yields the same manifest digest and a re-pull by digest returns identical bytes (hub.md §2.1). The
on-disk form is **OCI image-layout v1.1** (``oci-layout`` + ``index.json`` + ``blobs/sha256/<hex>``)
so any OCI-native tool (``oras``/``skopeo``/``cosign``) and the ``astro-mine-hub`` client push and
pull it unchanged — "standards in, standards out" (hub.md §2, principle 6). Content addressing uses
the platform primitive :mod:`astro_mine.core.hashing`, so a Hub digest is the same address Bench,
Fleet, and Cloud compute (never a private hash).

This module is the low-level layer: descriptors, blobs, and the on-disk layout. The higher-level
name→digest registry (immutability, referrers, GC) is :mod:`astro_mine.hub.registry._store`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astro_mine.hub._content import canonical_json, content_hash

__all__ = [
    "ARTIFACT_KINDS",
    "EMPTY_CONFIG",
    "LAYOUT_VERSION",
    "MEDIA_CORE_MANIFEST",
    "MEDIA_EMPTY",
    "MEDIA_INDEX",
    "MEDIA_MANIFEST",
    "REF_ANNOTATION",
    "Blob",
    "Descriptor",
    "IntegrityError",
    "artifact_kind_of",
    "artifact_media_type",
    "blob_path",
    "has_blob",
    "put_blob",
    "read_blob",
    "read_index",
    "verify_blob",
    "write_index",
    "write_layout_marker",
]

# --- OCI image-spec v1.1 media types ---------------------------------------------------------
MEDIA_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
MEDIA_INDEX = "application/vnd.oci.image.index.v1+json"
MEDIA_EMPTY = "application/vnd.oci.empty.v1+json"

#: The Core plugin manifest carried as an artifact's OCI config blob (hub.md §3): every artifact
#: references the Core manifest as its config, so the index reads it back without a private schema.
MEDIA_CORE_MANIFEST = "application/vnd.astro-mine.manifest.v1+json"

#: **Hub's container vocabulary — deliberately coarser than Core's ``PluginKind``, and not a
#: projection of it.** Each maps to ``application/vnd.astro-mine.<kind>.v1``: what *shape* of
#: payload an artifact carries, which is a packaging concern and therefore Hub's to name.
#:
#: This is not the interface vocabulary. ``PluginKind`` enumerates the *contracts* a plugin
#: implements, and Core owns it; a new one is a Core RFC. The two overlap on four names
#: (``asset``/``campaign``/``design``/``policy``) and diverge everywhere else — ``world``,
#: ``schema``, ``surrogate``, and ``plugin`` describe no Core interface, and twelve ``PluginKind``
#: members describe no distinct container.
#:
#: They cannot be unified by projection, and that is a fact rather than a preference: a served
#: surrogate's Core kind is ``FIELD_MODEL`` *or* ``REGIME_ENGINE`` depending on its physics domain
#: (astro-mine-surrogate ``manifest.py``), so no total map from container to interface exists.
#: Widening Core to absorb the container names would import Hub's packaging problem into the
#: narrow waist (core.md §2 principle 1: every addition to Core is a permanent liability).
#:
#: ``plugin`` is the deliberate **generic container** for a payload with no more specific shape —
#: Link's ``comms_model`` bundle and Prospect's ``resource_field_backend``/``prior_recipe`` bundles
#: both use it. That is a chosen catch-all, not an accident, and it is why the container vocabulary
#: is *less* informative than the manifest it wraps: read ``manifest.kind`` for the contract.
#:
#: ``design`` and ``campaign`` are the frozen Studio artifacts of RFC-0008 — a published trade
#: study, and the campaign Ops consumes unchanged.
#:
#: The tuple is append-only: a published ``name:version`` resolves to one immutable digest, so
#: renaming or removing a kind would break every artifact already carrying its media type.
ARTIFACT_KINDS: tuple[str, ...] = (
    "policy",
    "world",
    "asset",
    "surrogate",
    "plugin",
    "schema",
    "design",
    "campaign",
)

#: The standard OCI annotation carrying a tag reference (``name:version``) on an index entry.
REF_ANNOTATION = "org.opencontainers.image.ref.name"

LAYOUT_VERSION = "1.0.0"
EMPTY_CONFIG = b"{}"


class IntegrityError(Exception):
    """A stored blob's bytes do not hash to its content address (tamper / corruption)."""


def artifact_media_type(kind: str) -> str:
    """The ``application/vnd.astro-mine.<kind>.v1`` media type for a container ``kind``.

    ``kind`` must be one of :data:`ARTIFACT_KINDS` — Hub's **container** vocabulary, which is its
    own and is not derived from Core's ``PluginKind``. An unknown kind raises ``ValueError``.
    """
    if kind not in ARTIFACT_KINDS:
        known = ", ".join(ARTIFACT_KINDS)
        raise ValueError(f"unknown artifact kind {kind!r}; known kinds are: {known}")
    return f"application/vnd.astro-mine.{kind}.v1"


def artifact_kind_of(media_type: str | None) -> str | None:
    """The container kind an ``artifactType`` media type denotes, or ``None`` if it is not one.

    The inverse of :func:`artifact_media_type`, so an indexer can recover the kind from the bytes
    that were actually stored rather than from a field a caller asserted (hub.md §2 principle 3).
    Unrecognized or absent media types yield ``None``: an artifact published by some other tool is
    still storable and pullable, it simply carries no Hub container kind.
    """
    if not media_type:
        return None
    prefix, suffix = "application/vnd.astro-mine.", ".v1"
    if not (media_type.startswith(prefix) and media_type.endswith(suffix)):
        return None
    kind = media_type[len(prefix) : -len(suffix)]
    return kind if kind in ARTIFACT_KINDS else None


@dataclass(frozen=True)
class Descriptor:
    """An OCI content descriptor: ``{mediaType, digest, size}`` (+ optional fields)."""

    media_type: str
    digest: str  # "sha256:<hex>"
    size: int
    artifact_type: str | None = None
    annotations: Mapping[str, str] | None = None

    def as_dict(self) -> dict[str, Any]:
        """The OCI-wire ``dict`` form (only non-empty optional fields are emitted)."""
        out: dict[str, Any] = {
            "mediaType": self.media_type,
            "digest": self.digest,
            "size": self.size,
        }
        if self.artifact_type is not None:
            out["artifactType"] = self.artifact_type
        if self.annotations:
            out["annotations"] = dict(self.annotations)
        return out

    @classmethod
    def from_dict(cls, obj: Mapping[str, Any]) -> Descriptor:
        """Parse an OCI descriptor ``dict`` (as read back from a manifest / index)."""
        annotations = obj.get("annotations")
        return cls(
            media_type=obj["mediaType"],
            digest=obj["digest"],
            size=int(obj["size"]),
            artifact_type=obj.get("artifactType"),
            annotations=dict(annotations) if annotations else None,
        )


@dataclass(frozen=True)
class Blob:
    """A payload to store: its media type and raw bytes; content-addressed by :attr:`digest`."""

    media_type: str
    data: bytes

    @property
    def digest(self) -> str:
        """The ``sha256:<hex>`` content address of :attr:`data`."""
        return content_hash(self.data)

    @property
    def size(self) -> int:
        return len(self.data)

    def descriptor(self, *, artifact_type: str | None = None) -> Descriptor:
        return Descriptor(self.media_type, self.digest, self.size, artifact_type=artifact_type)


def _blobs_dir(layout: Path) -> Path:
    return layout / "blobs" / "sha256"


def blob_path(layout: Path, digest: str) -> Path:
    """The on-disk path of the blob at ``digest`` (``sha256:<hex>``); validates the algorithm."""
    algorithm, _, hexpart = digest.partition(":")
    if algorithm != "sha256":
        raise ValueError(f"unsupported digest algorithm {algorithm!r}; expected 'sha256'")
    if not hexpart:
        raise ValueError(f"malformed digest {digest!r}")
    return _blobs_dir(layout) / hexpart


def put_blob(layout: Path, blob: Blob) -> Descriptor:
    """Write ``blob`` to ``blobs/sha256/<hex>`` (idempotent) and return its descriptor."""
    path = blob_path(layout, blob.digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(blob.data)
    return blob.descriptor()


def has_blob(layout: Path, digest: str) -> bool:
    """Whether a blob at ``digest`` is present in the layout."""
    return blob_path(layout, digest).exists()


def read_blob(layout: Path, digest: str) -> bytes:
    """Return the bytes stored at ``digest``; ``KeyError`` if absent, ``ValueError`` on bad alg."""
    try:
        return blob_path(layout, digest).read_bytes()
    except FileNotFoundError:
        raise KeyError(digest) from None


def verify_blob(layout: Path, digest: str) -> None:
    """Assert the stored blob hashes to ``digest`` — content-addressing enforced on read.

    Addressing is enforced on write; a consumer pulling from an untrusted store must re-check on
    read ("verify before trust", hub.md §2.3). Raises :class:`IntegrityError` on a mismatch.
    """
    actual = content_hash(read_blob(layout, digest))
    if actual != digest:
        raise IntegrityError(f"blob {digest} content-address mismatch (stored bytes hash {actual})")


def write_layout_marker(layout: Path) -> None:
    """Write the ``oci-layout`` marker file (idempotent)."""
    layout.mkdir(parents=True, exist_ok=True)
    marker = layout / "oci-layout"
    if not marker.exists():
        marker.write_bytes(canonical_json({"imageLayoutVersion": LAYOUT_VERSION}))


def read_index(layout: Path) -> dict[str, Any]:
    """Parse ``index.json`` (an empty index if the file is absent)."""
    path = layout / "index.json"
    if not path.exists():
        return {"schemaVersion": 2, "mediaType": MEDIA_INDEX, "manifests": []}
    index: dict[str, Any] = json.loads(path.read_bytes())
    return index


def write_index(layout: Path, entries: list[dict[str, Any]]) -> None:
    """Write ``index.json`` referencing every top-level manifest ``entries``."""
    layout.mkdir(parents=True, exist_ok=True)
    index = {"schemaVersion": 2, "mediaType": MEDIA_INDEX, "manifests": entries}
    (layout / "index.json").write_bytes(canonical_json(index))


def canonical_manifest(manifest: Mapping[str, Any]) -> Blob:
    """A manifest ``dict`` as a byte-stable OCI manifest :class:`Blob` (canonical JSON)."""
    return Blob(MEDIA_MANIFEST, canonical_json(dict(manifest)))
