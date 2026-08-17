# SPDX-License-Identifier: Apache-2.0
"""The content-addressed artifact registry — name→digest, referrers, and GC (RM-P1-HUB-01).

:class:`Registry` is the OCI-backed store hub.md §1 opens with: it *stores* every commons content
kind as a **typed, content-addressed** artifact under an **immutable** ``name:version→digest``, and
attaches supply-chain attestations by digest via the **OCI Referrers** model. It is a thin,
OCI-faithful layer over :mod:`astro_mine.hub.registry._oci` — the on-disk form is a real OCI
image-layout, so ``oras``/``skopeo``/``cosign`` and the ``astro-mine-hub`` client read it unchanged
(the integration job exercises exactly that against a real Zot registry).

Guarantees (hub.md §2, §5):

- **Immutable** ``name:version``: a re-publish to an existing version is rejected — publish a new
  version (:class:`ArtifactExistsError`). Digests are forever; tags are the only mutable pointers.
- **Content-addressed**: an artifact's identity is its manifest digest; identical layers dedup.
- **Referrers**: attestations (signatures, SLSA provenance, SBOM) attach to an artifact by digest
  and are fetchable back — the hook :mod:`astro_mine.hub.supply_chain` hangs cosign/SLSA/SBOM on.
- **GC never reclaims a referenced digest**: :meth:`garbage_collect` reclaims only blobs
  unreachable from a tag, a referrer chain, or an explicit ``keep`` root (e.g. a digest a Bench
  result pinned) — the reproducibility guarantee (hub.md §5; negative test fails closed).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astro_mine.hub._content import canonical_json
from astro_mine.hub.registry._oci import (
    EMPTY_CONFIG,
    MEDIA_CORE_MANIFEST,
    MEDIA_EMPTY,
    MEDIA_MANIFEST,
    REF_ANNOTATION,
    Blob,
    Descriptor,
    artifact_media_type,
    blob_path,
    canonical_manifest,
    has_blob,
    put_blob,
    read_blob,
    read_index,
    verify_blob,
    write_index,
    write_layout_marker,
)

__all__ = [
    "ArtifactExistsError",
    "ArtifactNotFound",
    "PublishedArtifact",
    "Registry",
]


class ArtifactExistsError(Exception):
    """A publish targets a ``name:version`` that already exists (immutability — hub.md §2.1)."""


class ArtifactNotFound(KeyError):
    """A reference (``name:version`` or digest) does not resolve to a stored artifact."""


@dataclass(frozen=True)
class PublishedArtifact:
    """A published artifact: its name, version, ``name:version`` reference, and manifest digest."""

    name: str
    version: str
    reference: str
    digest: str  # the image-manifest digest — the artifact's content identity


def _parse_reference(reference: str) -> tuple[str, str, str | None]:
    """Split a reference into ``(name, version, digest)``.

    Accepts ``name:version`` (tag → ``digest`` None), ``name@sha256:…`` and bare ``sha256:…``
    (digest form → ``version`` ""). Raises ``ValueError`` on an unparseable reference.
    """
    if reference.startswith("sha256:"):
        return "", "", reference
    if "@" in reference:
        name, _, digest = reference.partition("@")
        if not name or not digest.startswith("sha256:"):
            raise ValueError(f"malformed digest reference {reference!r}")
        return name, "", digest
    name, sep, version = reference.partition(":")
    if not sep or not name or not version:
        raise ValueError(f"malformed reference {reference!r}; expected 'name:version' or a digest")
    return name, version, None


class Registry:
    """A content-addressed OCI artifact registry backed by a local OCI image-layout directory."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        write_layout_marker(self.path)

    # -- internal index helpers ---------------------------------------------------------------

    def _entries(self) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = read_index(self.path)["manifests"]
        return entries

    def _blob_size(self, digest: str) -> int:
        return blob_path(self.path, digest).stat().st_size

    # -- publish / resolve / pull -------------------------------------------------------------

    def publish(
        self,
        *,
        name: str,
        version: str,
        kind: str,
        config: bytes | Mapping[str, Any],
        layers: Sequence[Blob] = (),
        config_media_type: str = MEDIA_CORE_MANIFEST,
        annotations: Mapping[str, str] | None = None,
    ) -> PublishedArtifact:
        """Store a typed artifact under an immutable ``name:version`` and return its digest.

        ``config`` is the artifact's OCI config blob — the **Core plugin manifest** (bytes or a
        JSON-able mapping), by which the index catalogs it. ``layers`` are the payload blobs (ONNX,
        SADF bundle, Zarr/COG world, …). ``kind`` selects the ``artifactType`` (one of
        :data:`ARTIFACT_KINDS`). Raises :class:`ArtifactExistsError` if ``name:version`` exists.
        """
        reference = f"{name}:{version}"
        if any(
            (e.get("annotations") or {}).get(REF_ANNOTATION) == reference for e in self._entries()
        ):
            raise ArtifactExistsError(
                f"{reference} already published; digests are immutable — publish a new version"
            )
        config_bytes = config if isinstance(config, bytes) else canonical_json(dict(config))
        config_desc = put_blob(self.path, Blob(config_media_type, config_bytes))
        layer_descs = [put_blob(self.path, blob) for blob in layers]
        artifact_type = artifact_media_type(kind)
        image_manifest: dict[str, Any] = {
            "schemaVersion": 2,
            "mediaType": MEDIA_MANIFEST,
            "artifactType": artifact_type,
            "config": config_desc.as_dict(),
            "layers": [desc.as_dict() for desc in layer_descs],
        }
        if annotations:
            image_manifest["annotations"] = dict(annotations)
        manifest_desc = put_blob(self.path, canonical_manifest(image_manifest))

        entry = manifest_desc.as_dict()
        entry["artifactType"] = artifact_type
        entry["annotations"] = {REF_ANNOTATION: reference}
        entries = self._entries()
        entries.append(entry)
        write_index(self.path, entries)
        return PublishedArtifact(name, version, reference, manifest_desc.digest)

    def resolve(self, reference: str) -> Descriptor:
        """Resolve a ``name:version`` tag or a digest to its manifest :class:`Descriptor`.

        A tag resolves to its one immutable digest; a digest form resolves to itself if present.
        Raises :class:`ArtifactNotFound` if nothing matches.
        """
        name, version, digest = _parse_reference(reference)
        if digest is not None:
            if not has_blob(self.path, digest):
                raise ArtifactNotFound(reference)
            return Descriptor(MEDIA_MANIFEST, digest, self._blob_size(digest))
        target = f"{name}:{version}"
        for entry in self._entries():
            if (entry.get("annotations") or {}).get(REF_ANNOTATION) == target:
                return Descriptor.from_dict(entry)
        raise ArtifactNotFound(reference)

    def read_manifest(self, digest: str) -> dict[str, Any]:
        """The parsed image manifest at ``digest``."""
        manifest: dict[str, Any] = json.loads(read_blob(self.path, digest))
        return manifest

    def pull_blob(self, digest: str) -> bytes:
        """The raw bytes of the blob at ``digest`` (``KeyError`` if absent)."""
        return read_blob(self.path, digest)

    def read_config(self, manifest_digest: str) -> bytes:
        """The artifact's config blob (its Core plugin manifest) given the manifest digest."""
        return read_blob(self.path, self.read_manifest(manifest_digest)["config"]["digest"])

    # -- referrers (attestations) -------------------------------------------------------------

    def attach(
        self,
        *,
        subject: str,
        artifact_type: str,
        blob: Blob,
        annotations: Mapping[str, str] | None = None,
    ) -> Descriptor:
        """Attach an attestation ``blob`` to ``subject`` (a manifest digest) via OCI Referrers.

        Writes a referrer manifest whose ``subject`` is the subject artifact and whose single layer
        is ``blob`` (e.g. a cosign signature / SLSA provenance / SBOM). Returns the referrer's
        descriptor; fetch the set back with :meth:`referrers`.
        """
        if not has_blob(self.path, subject):
            raise ArtifactNotFound(subject)
        empty = put_blob(self.path, Blob(MEDIA_EMPTY, EMPTY_CONFIG))
        blob_desc = put_blob(self.path, blob)
        referrer: dict[str, Any] = {
            "schemaVersion": 2,
            "mediaType": MEDIA_MANIFEST,
            "artifactType": artifact_type,
            "config": empty.as_dict(),
            "layers": [blob_desc.as_dict()],
            "subject": {
                "mediaType": MEDIA_MANIFEST,
                "digest": subject,
                "size": self._blob_size(subject),
            },
        }
        if annotations:
            referrer["annotations"] = dict(annotations)
        referrer_desc = put_blob(self.path, canonical_manifest(referrer))
        entry = referrer_desc.as_dict()
        entry["artifactType"] = artifact_type
        entries = self._entries()
        entries.append(entry)
        write_index(self.path, entries)
        return referrer_desc

    def referrers(self, subject: str, *, artifact_type: str | None = None) -> list[Descriptor]:
        """Attestation manifests whose subject is ``subject`` (optional artifactType filter)."""
        found: list[Descriptor] = []
        for entry in self._entries():
            manifest = self.read_manifest(entry["digest"])
            ref = manifest.get("subject")
            if not (isinstance(ref, Mapping) and ref.get("digest") == subject):
                continue
            if artifact_type is not None and manifest.get("artifactType") != artifact_type:
                continue
            found.append(Descriptor.from_dict(entry))
        return found

    # -- catalog / listing --------------------------------------------------------------------

    def references(self) -> list[str]:
        """Every published ``name:version`` tag, sorted."""
        refs = [
            ann[REF_ANNOTATION]
            for e in self._entries()
            if (ann := e.get("annotations")) and REF_ANNOTATION in ann
        ]
        return sorted(refs)

    def versions(self, name: str) -> list[str]:
        """The published versions of ``name``, sorted."""
        prefix = f"{name}:"
        return sorted(ref[len(prefix) :] for ref in self.references() if ref.startswith(prefix))

    # -- integrity + GC -----------------------------------------------------------------------

    def verify(self, digest: str) -> None:
        """Re-check the manifest and all its blobs against their content addresses (hub.md §2.3)."""
        verify_blob(self.path, digest)
        manifest = self.read_manifest(digest)
        for descriptor in [manifest["config"], *manifest["layers"]]:
            verify_blob(self.path, descriptor["digest"])

    def garbage_collect(self, *, keep: Iterable[str] = ()) -> list[str]:
        """Reclaim unreferenced blobs; never reclaim a referenced digest. Returns reclaimed digests.

        Reachable = every tagged artifact, every referrer of a reachable artifact (transitively),
        their config/layer blobs, and every explicit ``keep`` root (a digest a Bench result or
        provenance chain pinned). A blob reachable by none of these is orphaned and reclaimed. This
        is the reproducibility guarantee: a pinned digest survives GC (hub.md §5).
        """
        keep_set = set(keep)
        entries = self._entries()
        by_digest = {e["digest"]: e for e in entries}

        reachable: set[str] = set(keep_set)
        seeds = [e["digest"] for e in entries if (e.get("annotations") or {}).get(REF_ANNOTATION)]
        seeds += [d for d in keep_set if d in by_digest]  # a kept manifest keeps its blobs too

        processed: set[str] = set()
        worklist = list(seeds)
        while worklist:
            md = worklist.pop()
            if md in processed:
                continue
            processed.add(md)
            reachable.add(md)
            manifest = self.read_manifest(md)
            reachable.add(manifest["config"]["digest"])
            reachable.update(layer["digest"] for layer in manifest["layers"])
            # a reachable artifact makes its referrers reachable (attestations survive with it)
            for entry in entries:
                other = self.read_manifest(entry["digest"])
                subject = other.get("subject")
                if isinstance(subject, Mapping) and subject.get("digest") == md:
                    worklist.append(entry["digest"])

        reclaimed: list[str] = []
        blobs_dir = self.path / "blobs" / "sha256"
        if blobs_dir.exists():
            for blob_file in blobs_dir.iterdir():
                digest = f"sha256:{blob_file.name}"
                if digest not in reachable:
                    blob_file.unlink()
                    reclaimed.append(digest)
        if reclaimed:
            write_index(self.path, [e for e in entries if e["digest"] not in set(reclaimed)])
        return sorted(reclaimed)
