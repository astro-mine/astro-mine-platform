"""Manifest indexing — the Core-manifest catalog (RM-P1-HUB-02).

Ingest every artifact's Core plugin manifest into a catalog record with Hub-side facets — the
projection :mod:`astro_mine.hub.search` discovers over. Indexed **by the Core manifest**, never a
Hub-private schema (hub.md §2, principle 2): a missing discovery field is a Core RFC, not a Hub
extension.

- :class:`CatalogEntry` — a manifest + facets (namespace, publisher, downloads, embedding).
- :func:`ingest` — project + embed + add.
- :class:`Catalog` / :class:`InMemoryCatalog` — the store contract + dependency-clean default;
  :class:`~astro_mine.hub.index._sql.SqlCatalog` (``[service]`` extra) is the durable backend.

Backlog: RM-P1-HUB-02 — astro-mine-hub#2
"""

from __future__ import annotations

from astro_mine.hub.index._catalog import Catalog, CatalogEntry, InMemoryCatalog
from astro_mine.hub.index._ingest import index_document, ingest

__all__ = ["Catalog", "CatalogEntry", "InMemoryCatalog", "index_document", "ingest"]


def sql_catalog(url: str | None = None, *, embedding_dim: int | None = None) -> Catalog:
    """Construct the SQLAlchemy-backed catalog (``[service]`` extra); imported lazily.

    A thin constructor so callers get a :class:`Catalog` without importing SQLAlchemy at module load
    (keeping the base package dependency-clean); raises ``ImportError`` if the extra is absent.
    ``embedding_dim`` MUST match the :class:`~astro_mine.hub.search.EmbeddingProvider` the catalog
    is ingested with — it is the width of the PostgreSQL ``vector(dim)`` column (defaults to the
    offline hashing provider's).
    """
    from astro_mine.hub.index._sql import SqlCatalog
    from astro_mine.hub.search import default_provider

    dim = embedding_dim if embedding_dim is not None else default_provider().dim
    return SqlCatalog(url, embedding_dim=dim)
