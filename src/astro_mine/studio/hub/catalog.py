"""The Fleet/Hub asset catalog as Studio's robot menu + geometry preview (RM-P1-STUDIO-09).

Studio surfaces the assets a designer can choose from, and previews one in the embedded View, by
reading the Hub catalog and the selected asset's layers **directly from Hub by content hash** — the
same "pure Core consumer, no sibling-package imports" stance as
:class:`~astro_mine.studio.hub.HubCapabilityResolver` and
:class:`~astro_mine.studio.hub.HubWorldMaterializer` (studio.md §2 principle 4, §6). A Hub-published
asset therefore appears in the menu with **no Studio edit**: the vehicle kind and name are on the
Core plugin manifest's ``attributes``, and the capability tags are the Core-vocabulary strings
Mind/Allocate reason over. `astro_mine.fleet` is never imported.

:class:`HubAssetPreviewMaterializer` clones :class:`HubWorldMaterializer`: resolve → **verify-before
-trust** pull → content-addressed cache → write the SADF-JSON layer and each geometry blob (mapped
``uri → digest`` through the Core manifest's ``provenance.source_content_hashes``) at its relative
``uri``, so the embedded View ``<AssetPreview source={{documentUrl}}/>`` widget (RM-P1-VIEW-03) can
fetch it. Studio opens no SADF document — it copies the JSON layer and the geometry blobs the Core
manifest names, and serves nothing it did not verify: every byte arrives through Hub's **verified
payload retrieval** (each layer re-hashed against the verified manifest's digest), and a geometry
``uri`` whose digest is not a layer of *this* artifact is refused rather than fetched from the
registry as a loose blob.

The asset-bundle **layer media types** below are matched as a stable wire contract (the astro-mine
-fleet packaging format), not imported: Studio reads Hub-indexed bytes and depends on no Fleet
code (a Core/Hub-owned media-type vocabulary would let this share one definition — a future RFC).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from astro_mine.core.registry import CapabilityTag, PluginKind, PluginManifest

from .._base import FrozenStudioModel, StudioModel

if TYPE_CHECKING:  # pragma: no cover - the [hub] extra is not in the base wheel
    from astro_mine.hub.client import HubClient
    from astro_mine.hub.index import CatalogEntry
    from astro_mine.hub.registry import Registry

__all__ = [
    "AssetCatalog",
    "AssetPreviewMaterializer",
    "HubAssetCatalog",
    "HubAssetPreviewMaterializer",
    "MaterializedPreview",
    "MenuEntry",
    "PreviewError",
    "StudioCatalog",
    "WorldCatalog",
    "WorldEntry",
]

#: The Fleet asset-bundle layer media types (astro-mine-fleet ``packaging/oci.py``). Matched, not
#: imported — Studio identifies Hub-indexed layers by their published media type, a stable wire
#: contract, and depends on no Fleet code.
MEDIA_SADF_JSON = "application/vnd.astro-mine.asset.sadf.json.v1+json"
MEDIA_GEOMETRY_GLTF = "application/vnd.astro-mine.asset.geometry.gltf.v1"
MEDIA_GEOMETRY_USD = "application/vnd.astro-mine.asset.geometry.usd.v1"


class PreviewError(RuntimeError):
    """An asset could not be resolved, verified, or materialized for preview."""


class MenuEntry(StudioModel):
    """One selectable robot-menu row: an asset's identity + the Core capability tags it declares.

    Projected from the Hub-indexed Core plugin manifest, so a Hub-published asset yields a row with
    no Studio edit. ``kind`` is the **vehicle** kind the menu groups by (rover/orbiter/…), carried
    on the manifest as ``attributes["asset_kind"]`` — never the plugin kind (always ``asset``).
    ``capability_tags`` are the Core-vocabulary strings; ``digest`` is the content address a preview
    pulls by.
    """

    reference: str
    digest: str
    name: str
    version: str
    kind: str
    namespace: str
    capability_tags: list[str]


class WorldEntry(StudioModel):
    """One selectable world-menu row: a published world bundle's identity and the body it models.

    The counterpart of :class:`MenuEntry` for ``PluginKind.WORLD_PROVIDER`` artifacts. Terrain used
    to be reachable only by hand-editing a ``?world=`` query parameter, which no part of the UI
    offered or documented, so ``UC-F5`` had no front door at all.

    ``body`` is carried on the manifest as ``attributes["body"]`` where the producer stamps it; it
    lets a surface narrow the menu to worlds applicable to a study's own ``GeoRegion``.
    """

    reference: str
    digest: str
    name: str
    version: str
    namespace: str
    body: str | None = None


class MaterializedPreview(FrozenStudioModel):
    """A verified asset's geometry on local disk, ready to be served to the embedded View.

    ``document_path`` is the SADF JSON the ``<AssetPreview source={{documentUrl}}/>`` widget loads;
    its geometry ``uri``\\ s resolve relative to it. ``digest`` is the content address and the only
    durable identity; ``path`` is a cache directory derived from it, disposable.
    """

    reference: str
    digest: str
    path: Path
    document_path: Path


@runtime_checkable
class AssetCatalog(Protocol):
    """The ←Hub seam for the asset menu. A deployment binds it to Hub, a test to a temp registry."""

    def list_assets(
        self, *, requires: Sequence[CapabilityTag] | None = None
    ) -> list[MenuEntry]: ...


@runtime_checkable
class WorldCatalog(Protocol):
    """The ←Hub seam for the world menu. Separate from :class:`AssetCatalog` because the two menus
    answer different questions, though one Hub-backed object satisfies both."""

    def list_worlds(self) -> list[WorldEntry]: ...


@runtime_checkable
class StudioCatalog(AssetCatalog, WorldCatalog, Protocol):
    """Both menus behind one injected seam. :class:`HubAssetCatalog` satisfies it; the two halves
    stay separately declared so a consumer can depend on only the one it uses."""


@runtime_checkable
class AssetPreviewMaterializer(Protocol):
    """The ←Hub seam for a single selected asset's geometry preview."""

    def preview(self, reference: str) -> MaterializedPreview: ...


def _menu_entry(entry: CatalogEntry) -> MenuEntry:
    # The vehicle kind + display name ride on the manifest attributes Fleet stamps at publish; a
    # non-Fleet asset without them falls back to the catalog record's own fields.
    attributes = entry.manifest.attributes
    return MenuEntry(
        reference=entry.reference,
        digest=entry.digest,
        name=str(attributes.get("asset_name") or entry.name),
        version=entry.version,
        kind=str(attributes.get("asset_kind") or entry.kind),
        namespace=entry.namespace,
        capability_tags=list(entry.capability_tags),
    )


class HubAssetCatalog:
    """Enumerate a Hub registry as the robot menu — Hub-direct, opening no SADF document."""

    def __init__(self, registry: Registry) -> None:
        self._registry = registry

    def list_assets(self, *, requires: Sequence[CapabilityTag] | None = None) -> list[MenuEntry]:
        from astro_mine.hub.client import catalog_from_registry

        # The robot menu is **assets only**. A Hub registry also holds worlds, resource fields,
        # policies, comms models, … — none of those is a selectable robot (and the preview
        # materializer refuses a non-asset), so a menu that listed them would offer rows that error
        # the moment they are clicked. Filter to `PluginKind.ASSET` (the manifest's own kind), then
        # by the requested capabilities.
        catalog = catalog_from_registry(self._registry)
        entries = [
            _menu_entry(entry)
            for entry in catalog.all()
            if entry.manifest.kind is PluginKind.ASSET
            and (requires is None or entry.satisfies(capability_tags=requires))
        ]
        return sorted(entries, key=lambda menu: menu.reference)

    def list_worlds(self) -> list[WorldEntry]:
        """Enumerate the registry's world bundles — the same query as the robot menu, a different
        kind filter. ``GET /worlds/{reference}`` already materializes whichever of these is chosen;
        only the front door was missing."""
        from astro_mine.hub.client import catalog_from_registry

        catalog = catalog_from_registry(self._registry)
        entries = [
            WorldEntry(
                reference=entry.reference,
                digest=entry.digest,
                name=str(entry.manifest.attributes.get("world_name") or entry.name),
                version=entry.version,
                namespace=entry.namespace,
                body=(
                    str(entry.manifest.attributes["body"])
                    if entry.manifest.attributes.get("body") is not None
                    else None
                ),
            )
            for entry in catalog.all()
            if entry.manifest.kind is PluginKind.WORLD_PROVIDER
        ]
        return sorted(entries, key=lambda world: world.reference)


class HubAssetPreviewMaterializer:
    """Pull an asset by digest, verify it, and lay its geometry out in a content-addressed cache."""

    def __init__(self, client: HubClient, *, cache_dir: str | Path) -> None:
        self._client = client
        self._cache_dir = Path(cache_dir)

    def preview(self, reference: str) -> MaterializedPreview:
        registry = self._client.registry
        try:
            digest = registry.resolve(reference).digest
        except Exception as exc:
            raise PreviewError(f"no asset {reference!r} in Hub: {exc}") from exc

        # Re-verify client-side before a single byte is trusted (verify twice, hub.md §2.3).
        try:
            config = self._client.pull(digest, verify=True)
        except Exception as exc:
            raise PreviewError(f"refusing asset {reference}: {exc}") from exc

        manifest = PluginManifest.model_validate_json(config)
        if manifest.kind is not PluginKind.ASSET:
            raise PreviewError(f"{reference} is a {manifest.kind.value} artifact, not an asset")

        # The digest keys the cache, so a re-preview of the same asset is free and two assets never
        # collide. `:` is not portable in a path segment.
        dest = self._cache_dir / digest.replace(":", "-")
        document = dest / f"{_safe_stem(manifest.name)}.sadf.json"
        if document.is_file():
            return MaterializedPreview(
                reference=reference, digest=digest, path=dest, document_path=document
            )

        # One pull, then index it: every verified call re-runs the supply-chain check, and an
        # asset's layers are a SADF document plus its geometry — not a multi-GB world.
        try:
            layers = self._client.pull_payload(digest, verify=True)
        except Exception as exc:
            raise PreviewError(f"asset {reference} has no readable layers: {exc}") from exc
        by_digest = {layer.digest: layer.data for layer in layers}
        sadf_json = next(
            (layer.data for layer in layers if layer.media_type == MEDIA_SADF_JSON), None
        )
        if sadf_json is None:
            raise PreviewError(f"asset {reference} carries no {MEDIA_SADF_JSON} layer to preview")

        source_hashes = (
            dict(manifest.provenance.source_content_hashes) if manifest.provenance else {}
        )

        dest.mkdir(parents=True, exist_ok=True)
        _write_within(dest, document, sadf_json, reference)
        for uri, blob_digest in source_hashes.items():
            geometry = by_digest.get(blob_digest)
            if geometry is None:
                # Geometry must be a layer of *this verified artifact*, not merely a blob the store
                # happens to hold at that content address — a digest no verified manifest committed
                # to is unattested, so it is refused rather than served.
                raise PreviewError(
                    f"asset {reference} names geometry {uri!r} at {blob_digest}, which is not a "
                    "layer of the verified artifact"
                )
            _write_within(dest, dest / uri, geometry, reference)

        return MaterializedPreview(
            reference=reference, digest=digest, path=dest, document_path=document
        )


def _write_within(root: Path, target: Path, data: bytes, reference: str) -> None:
    """Write *data* to *target*, refusing any path that escapes *root* — fail closed.

    Core does not reject a ``../`` geometry ``uri``, so this guard is load-bearing, exactly like the
    ``filter="data"`` guard on Hub's world-bundle extractor.
    """
    if not target.resolve().is_relative_to(root.resolve()):
        raise PreviewError(f"asset {reference} names a path that escapes its cache: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)


def _safe_stem(name: str) -> str:
    # Asset ids are dotted reverse-DNS (no separator), but guard the document filename defensively.
    return name.replace("/", "_").replace("\\", "_") or "asset"
