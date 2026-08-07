"""Content-addressed OCI artifact registry — local layout or any remote registry (RM-P1-HUB-01/06).

The store hub.md §1 opens with: worlds, SADF assets, ONNX policies, surrogate models, plugins, and
schema bundles kept as **typed, content-addressed** OCI artifacts under an **immutable**
``name:version→digest``, with supply-chain attestations attached by digest via the **OCI Referrers**
model. There is no proprietary push/pull protocol (hub.md §2, principle 6) — two transports satisfy
one contract:

- :class:`Registry` — a local **OCI image-layout** directory. The tier-1 path that MUST work fully
  offline (hub.md principle 7); ``oras``/``skopeo``/``cosign`` read it unchanged.
- :class:`RemoteRegistry` — the **OCI Distribution Spec** over HTTP (ghcr.io, Zot, Harbor, …), with
  standard Docker credential resolution and the bearer-token handshake (RM-P1-HUB-06: the client
  "resolves/verifies/pulls against *any* OCI registry"). Every fetch re-hashes its own bytes, so
  verify-twice survives the wire.
- :class:`RegistryClient` — the protocol both satisfy; everything above the transport
  (:mod:`astro_mine.hub.supply_chain`, the client, the CLI) is written against it.
- :func:`open_registry` — the one-line "path or registry URL" factory the CLI ``--registry`` uses.

Backlog: RM-P1-HUB-01 — https://github.com/astro-mine/astro-mine-hub/issues/1
"""

from __future__ import annotations

from pathlib import Path

from astro_mine.hub.registry._auth import Credentials, credentials_for
from astro_mine.hub.registry._names import (
    ARTIFACT_NAME_PATTERN,
    InvalidArtifactName,
    is_valid_artifact_name,
    validate_artifact_name,
)
from astro_mine.hub.registry._oci import (
    ARTIFACT_KINDS,
    MEDIA_CORE_MANIFEST,
    Blob,
    Descriptor,
    IntegrityError,
    artifact_kind_of,
    artifact_media_type,
)
from astro_mine.hub.registry._protocol import RegistryClient
from astro_mine.hub.registry._remote import RegistryHttpError, RemoteRegistry, is_remote
from astro_mine.hub.registry._store import (
    ArtifactExistsError,
    ArtifactNotFound,
    PublishedArtifact,
    Registry,
)

__all__ = [
    "ARTIFACT_KINDS",
    "ARTIFACT_NAME_PATTERN",
    "MEDIA_CORE_MANIFEST",
    "ArtifactExistsError",
    "ArtifactNotFound",
    "Blob",
    "Credentials",
    "Descriptor",
    "IntegrityError",
    "InvalidArtifactName",
    "PublishedArtifact",
    "Registry",
    "RegistryClient",
    "RegistryHttpError",
    "RemoteRegistry",
    "artifact_kind_of",
    "artifact_media_type",
    "credentials_for",
    "is_remote",
    "is_valid_artifact_name",
    "open_registry",
    "validate_artifact_name",
]


def open_registry(location: str | Path) -> RegistryClient:
    """Open ``location`` as a registry — a local OCI-layout **path** or a remote **registry URL**.

    ``./reg`` / ``/var/lib/hub`` → :class:`Registry` (offline, no network);
    ``ghcr.io/astro-mine`` / ``https://registry.example.org/commons`` / ``http://localhost:5000`` →
    :class:`RemoteRegistry`. One flag, either transport (hub.md §7).
    """
    text = str(location)
    return RemoteRegistry(text) if is_remote(text) else Registry(text)
