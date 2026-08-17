# SPDX-License-Identifier: Apache-2.0
"""Ingest a Core plugin manifest into the catalog (RM-P1-HUB-02).

On publish, Hub projects the artifact's Core plugin manifest into a :class:`CatalogEntry` and
computes its semantic embedding through the pluggable
:class:`~astro_mine.hub.search.EmbeddingProvider` (hub.md §3 "Publish"; the §3/§11 provider seam).
:func:`index_document` is the text the entry is embedded from and full-text-indexed on — manifest
name, description, capability tags, and I/O type names — so "find an excavation policy" and
capability queries both hit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_mine.core.registry import PluginManifest
from astro_mine.hub.index._catalog import Catalog, CatalogEntry

if TYPE_CHECKING:  # the provider seam lives in search/; import lazily to keep index/ standalone
    from astro_mine.hub.search import EmbeddingProvider

__all__ = ["index_document", "ingest"]


def index_document(manifest: PluginManifest) -> str:
    """The free-text document a manifest is embedded from and full-text indexed on."""
    parts = [
        manifest.name,
        manifest.description or "",
        *(tag.value for tag in manifest.capability_tags),
        *manifest.inputs,
        *manifest.outputs,
    ]
    return " ".join(part for part in parts if part)


def ingest(
    catalog: Catalog,
    manifest: PluginManifest,
    *,
    digest: str,
    publisher: str,
    namespace: str = "open",
    artifact_kind: str | None = None,
    provider: EmbeddingProvider | None = None,
) -> CatalogEntry:
    """Project ``manifest`` into a :class:`CatalogEntry`, embed it, and add it to ``catalog``.

    ``digest`` is the artifact's content address (from :class:`~astro_mine.hub.registry.Registry`);
    ``publisher`` and ``namespace`` are the Hub-side facets. ``artifact_kind`` is Hub's *container*
    kind, a separate axis from the Core interface ``manifest.kind`` declares — admission recovers it
    from the stored OCI ``artifactType`` so it cannot be asserted by a caller. ``provider`` selects
    the embedding
    backend (default: the offline hashing provider); its name is recorded on the entry, because
    vectors produced by different providers are not comparable. Returns the indexed entry.
    """
    from astro_mine.hub.search import default_provider

    backend = provider if provider is not None else default_provider()
    entry = CatalogEntry(
        manifest=manifest,
        digest=digest,
        publisher=publisher,
        namespace=namespace,
        artifact_kind=artifact_kind,
        embedding=backend.embed(index_document(manifest)),
        embedding_provider=backend.name,
    )
    catalog.add(entry)
    return entry
