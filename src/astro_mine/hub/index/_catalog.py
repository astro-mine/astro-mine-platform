"""The catalog record + store contract, indexed by the Core plugin manifest (RM-P1-HUB-02).

Hub indexes every artifact **by its Core plugin manifest** — never a Hub-private schema (hub.md §2,
principle 2). A :class:`CatalogEntry` wraps the ingested
:class:`~astro_mine.core.registry.PluginManifest` (the source of truth, projected to Core's
:class:`~astro_mine.core.registry.CatalogRecord`) with the **Hub-side facets** Core deliberately
does *not* own — namespace, publisher, download count, deprecation/yank, and the semantic embedding
(the ``CatalogRecord`` docstring names exactly these as Hub's to add). Capability negotiation
delegates to :meth:`PluginManifest.satisfies` — Core's rule, consumed, not reimplemented.

:class:`Catalog` is the store contract; :class:`InMemoryCatalog` is the dependency-clean default
(tier-1 + tests). The SQLAlchemy-backed :class:`~astro_mine.hub.index._sql.SqlCatalog` (SQLite for
tests, PostgreSQL for the hosted tier) runs the *same* search code through the ``[service]`` extra.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.core.registry import CatalogRecord, PluginManifest
from astro_mine.core.sadf.enums import CapabilityTag

__all__ = ["Catalog", "CatalogEntry", "InMemoryCatalog"]


class CatalogEntry(BaseModel):
    """An indexed artifact: its Core manifest (source of truth) + Hub-side facets."""

    model_config = ConfigDict(frozen=False)

    manifest: PluginManifest
    digest: str = Field(min_length=1)  # the artifact's content address
    publisher: str = Field(min_length=1)
    namespace: str = "open"  # open (self-published) | curated (reviewed) — hub.md §9
    #: Hub's **container** kind (``ARTIFACT_KINDS``) — what shape of payload this artifact carries.
    #: A separate axis from :attr:`kind`, which is the Core *interface* the manifest declares, and
    #: the two do not map onto each other (a surrogate is ``FIELD_MODEL`` or ``REGIME_ENGINE`` by
    #: physics domain). Recovered from the stored OCI ``artifactType`` at admission rather than
    #: taken from a caller, so it cannot drift from the bytes. ``None`` for an artifact published
    #: by some other tool, or indexed before this facet existed.
    artifact_kind: str | None = None
    downloads: int = 0
    deprecated: bool = False
    yanked: bool = False
    embedding: tuple[float, ...] = ()
    #: The :class:`~astro_mine.hub.search.EmbeddingProvider` that produced :attr:`embedding` —
    #: vectors from different providers are not comparable, so the catalog records which one.
    embedding_provider: str = ""

    @property
    def record(self) -> CatalogRecord:
        """The Core projection of the manifest — the indexed schema (no Hub-private fields)."""
        return self.manifest.to_catalog_record()

    @property
    def name(self) -> str:
        return self.manifest.name

    @property
    def version(self) -> str:
        return self.manifest.version

    @property
    def reference(self) -> str:
        """The ``name:version`` catalog key."""
        return f"{self.manifest.name}:{self.manifest.version}"

    @property
    def kind(self) -> str:
        return self.manifest.kind.value

    @property
    def license(self) -> str | None:
        return self.manifest.license

    @property
    def capability_tags(self) -> list[str]:
        return [tag.value for tag in self.manifest.capability_tags]

    def satisfies(
        self,
        *,
        interfaces: Mapping[str, str] | None = None,
        capability_tags: Iterable[CapabilityTag | str] | None = None,
    ) -> bool:
        """Whether the manifest satisfies the requested interfaces + tags (Core's rule)."""
        return self.manifest.satisfies(interfaces=interfaces, capability_tags=capability_tags)


class Catalog(Protocol):
    """Persistence for catalog entries — add (idempotent on reference), get, and list all."""

    def add(self, entry: CatalogEntry) -> None:
        """Index ``entry`` (re-adding the same ``reference`` replaces it)."""
        ...

    def get(self, reference: str) -> CatalogEntry | None:
        """The entry for ``name:version``, or ``None``."""
        ...

    def all(self) -> list[CatalogEntry]:
        """Every indexed entry."""
        ...


class InMemoryCatalog:
    """A process-local :class:`Catalog` — the dependency-clean default (tier-1 + tests)."""

    def __init__(self) -> None:
        self._by_ref: dict[str, CatalogEntry] = {}

    def add(self, entry: CatalogEntry) -> None:
        self._by_ref[entry.reference] = entry

    def get(self, reference: str) -> CatalogEntry | None:
        return self._by_ref.get(reference)

    def all(self) -> list[CatalogEntry]:
        return list(self._by_ref.values())
