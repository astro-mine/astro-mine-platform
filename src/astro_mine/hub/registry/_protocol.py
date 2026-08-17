# SPDX-License-Identifier: Apache-2.0
"""The registry transport contract — one interface, a local layout or any OCI registry.

:class:`RegistryClient` is the seam hub.md §7 requires: the ``astro-mine-hub`` client
"resolves/verifies/pulls against **any** OCI registry ... so a researcher needs no hosted Hub".
:class:`~astro_mine.hub.registry._store.Registry` (a local OCI image-layout directory — the tier-1
path that MUST work fully offline, hub.md principle 7) and
:class:`~astro_mine.hub.registry._remote.RemoteRegistry` (the OCI **Distribution-Spec** HTTP
transport — ghcr.io, Zot, Harbor, …) both satisfy it, so every layer above the transport —
:mod:`astro_mine.hub.supply_chain`, :class:`~astro_mine.hub.client.HubClient`, the CLI — is written
once against the protocol and works against either.

The protocol is deliberately the *content-addressed* surface only: publish, resolve, read, attach,
and re-verify. Garbage collection is a property of a store you own (the local layout), not of a
transport, so it stays off the protocol.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from astro_mine.hub.registry._oci import MEDIA_CORE_MANIFEST, Blob, Descriptor
from astro_mine.hub.registry._store import PublishedArtifact

__all__ = ["RegistryClient"]


@runtime_checkable
class RegistryClient(Protocol):
    """Publish / resolve / pull / attach / verify against a content-addressed OCI registry."""

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
        """Store a typed artifact under an immutable ``name:version`` and return its digest."""
        ...

    def resolve(self, reference: str) -> Descriptor:
        """Resolve a ``name:version`` tag or a digest to its manifest descriptor."""
        ...

    def read_manifest(self, digest: str) -> dict[str, Any]:
        """The parsed image manifest at ``digest``."""
        ...

    def pull_blob(self, digest: str) -> bytes:
        """The raw bytes of the blob at ``digest``."""
        ...

    def read_config(self, manifest_digest: str) -> bytes:
        """The artifact's config blob (its Core plugin manifest)."""
        ...

    def attach(
        self,
        *,
        subject: str,
        artifact_type: str,
        blob: Blob,
        annotations: Mapping[str, str] | None = None,
    ) -> Descriptor:
        """Attach an attestation ``blob`` to ``subject`` via OCI Referrers."""
        ...

    def referrers(self, subject: str, *, artifact_type: str | None = None) -> list[Descriptor]:
        """Attestation manifests whose subject is ``subject`` (optional artifactType filter)."""
        ...

    def references(self) -> list[str]:
        """Every published ``name:version`` tag, sorted."""
        ...

    def versions(self, name: str) -> list[str]:
        """The published versions of ``name``, sorted."""
        ...

    def verify(self, digest: str) -> None:
        """Re-check the manifest and all its blobs against their content addresses."""
        ...
