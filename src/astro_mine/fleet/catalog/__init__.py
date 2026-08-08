"""The Fleet/Hub catalog as a selectable robot menu (RM-P1-FLEET-11).

A read-only projection of a [Hub](https://github.com/astro-mine/docs) registry into the two
surfaces autonomy and design need (``fleet.md`` §6, §12 Phase 1):

- **the selectable robot menu** ([Studio](https://github.com/astro-mine/docs)) — one
  :class:`MenuEntry` per published asset, carrying its identity, kind, namespace, and the Core
  capability tags it declares;
- **capability declarations** ([Mind](https://github.com/astro-mine/docs)/[Allocate](https://github.com/astro-mine/docs))
  — those same tags drive role negotiation and heterogeneous task allocation. Filtering a menu by
  *required* tags delegates to Core's own
  :meth:`~astro_mine.hub.index.CatalogEntry.satisfies` — Fleet embeds **no planner logic**
  (``fleet.md`` §2.2, §2.5).

The **contract is the Core capability vocabulary + the Hub catalog, not a Fleet API**: a
Hub-published vehicle type appears here with **no Fleet code change** — the Phase-1 Fleet exit
criterion (``fleet.md`` §12). Because a menu row is projected straight from an entry's Core plugin
manifest, a brand-new kind arrives as content, never as a Fleet edit.

:func:`asset_preview` pulls a *selected* asset's SADF and returns its glTF/USD
:class:`~astro_mine.core.sadf.model.GeometryRef`\\ s — the geometry the single-asset preview widget
shows (``fleet.md`` §6; the View widget of RM-P1-VIEW-03) — the one thing Hub's catalog record
deliberately does not carry. :func:`materialize_preview` goes one step further: it writes the SADF
JSON **and** its geometry blobs to disk laid out so a static server can hand the View
``<AssetPreview source={{documentUrl}}/>`` widget a URL — the renderable form of a preview, and the
reference layout the Studio-side materializer reproduces Hub-direct. Enumerating the menu
(:func:`list_menu`) needs no pull; previewing one asset does, so the cost tracks what a menu
actually does: list everything, pull the one clicked.

This module imports **only Core + the Hub client** (Hub lazily, like
:mod:`astro_mine.fleet.packaging.hub`) — never Mind/Allocate/Studio (``fleet.md`` §2.2, §6).

Backlog: RM-P1-FLEET-11 -- astro-mine-fleet#22
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from astro_mine.core.registry import PluginManifest
from astro_mine.core.registry.enums import PluginKind
from astro_mine.core.sadf import load_sadf
from astro_mine.core.sadf.enums import CapabilityTag, GeometryFormat
from astro_mine.core.sadf.model import GeometryRef
from astro_mine.fleet.capabilities import as_tags
from astro_mine.fleet.packaging import oci
from astro_mine.fleet.packaging.hub import HubError, _require_hub, pull_asset

if TYPE_CHECKING:  # avoid importing the Hub stack just to type a helper
    from astro_mine.hub.index import CatalogEntry
    from astro_mine.hub.registry import RegistryClient

__all__ = ["MenuEntry", "asset_preview", "list_menu", "materialize_preview"]


@dataclass(frozen=True)
class MenuEntry:
    """One selectable robot-menu row: an asset's identity + the Core capability tags it declares.

    Projected from a Hub :class:`~astro_mine.hub.index.CatalogEntry` (its Core plugin manifest),
    so a Hub-published asset yields a row with no Fleet code change. ``kind`` is the **vehicle**
    kind the menu groups by (``rover``, ``orbiter``, ``excavator``, …) — carried on the manifest
    as ``attributes["asset_kind"]``, not the plugin kind (always ``"asset"``). ``capability_tags``
    are the Core-vocabulary strings Mind/Allocate reason over; ``digest`` is the content address to
    re-pull/verify (or preview) the asset by.
    """

    reference: str  # name:version catalog key -- the pull/preview key
    digest: str  # content address (sha256:...) -- re-pull/verify/preview by this
    name: str  # human-readable display name (attributes["asset_name"], falling back to the id)
    version: str
    kind: str  # vehicle kind (rover/orbiter/...), from attributes["asset_kind"]
    namespace: str  # open (self-published) | curated (reviewed) -- hub.md §9
    capability_tags: tuple[str, ...]


def _menu_entry(entry: CatalogEntry) -> MenuEntry:
    # The vehicle kind + display name ride on the manifest attributes Fleet stamps at publish
    # (packaging/manifest.py); a non-Fleet asset without them falls back to the catalog record.
    attributes = entry.manifest.attributes
    return MenuEntry(
        reference=entry.reference,
        digest=entry.digest,
        name=str(attributes.get("asset_name") or entry.name),
        version=entry.version,
        kind=str(attributes.get("asset_kind") or entry.kind),
        namespace=entry.namespace,
        capability_tags=tuple(entry.capability_tags),
    )


def list_menu(
    registry: RegistryClient,
    *,
    requires: Iterable[CapabilityTag | str] | None = None,
) -> list[MenuEntry]:
    """Enumerate a Hub registry's **assets** as the selectable robot menu (``fleet.md`` §6).

    Returns one :class:`MenuEntry` per published ``asset``, built from its Core plugin manifest —
    so a Hub-published asset appears with **no Fleet code change**. With *requires*, only assets
    whose declared capabilities satisfy **every** requested tag are returned, delegating to Core's
    :meth:`~astro_mine.hub.index.CatalogEntry.satisfies` (the Mind/Allocate negotiation rule, not
    Fleet-side planner logic). *requires* is validated against Core's closed capability vocabulary
    (:func:`~astro_mine.fleet.capabilities.as_tags`), so an unknown tag raises
    :class:`~astro_mine.fleet.capabilities.CapabilityError` — the signal to open a Core RFC, not a
    Fleet-private tag. Entries are sorted by reference for deterministic output.

    **Only ``kind == asset``.** A registry holds every kind the platform publishes — worlds,
    resource priors, contact plans, surrogates, policies, campaigns — and capability tags are not
    exclusive to assets: Core's own shipped example *policy* manifest declares
    ``mobility.wheeled``, and that is the canonical shape, not a malformed manifest. Enumerating
    every kind therefore put a policy and a campaign in the robot menu, and ``requires`` returned
    them as candidates for digging. Capability tags are Core's **negotiation** vocabulary; matching
    them across kinds negotiates between things that do not compose. Registry-wide discovery is
    `hub search`'s job, and this is the taskability query — the same one Mind and Allocate run.
    """
    _require_hub()
    from astro_mine.hub.client import catalog_from_registry

    required = as_tags(requires) if requires is not None else None
    catalog = catalog_from_registry(registry)
    entries = [
        _menu_entry(entry)
        for entry in catalog.all()
        if entry.kind == PluginKind.ASSET
        and (required is None or entry.satisfies(capability_tags=required))
    ]
    return sorted(entries, key=lambda menu: menu.reference)


def asset_preview(
    registry: RegistryClient,
    reference: str,
    *,
    fmt: GeometryFormat | str = GeometryFormat.GLTF,
    verify: bool = True,
    trusted_public_key_pem: bytes | None = None,
    require: Sequence[str] | None = None,
) -> list[GeometryRef]:
    """The glTF/USD geometry a selected asset offers for preview (``fleet.md`` §6; RM-P1-VIEW-03).

    Pulls *reference* (a ``name:version`` tag or ``sha256:…`` digest) from the Hub registry **by
    content hash** — verify-before-trust unless *verify* is ``False``, see
    :func:`~astro_mine.fleet.packaging.hub.pull_asset` — and returns the asset-level
    :class:`~astro_mine.core.sadf.model.GeometryRef`\\ s in the requested *fmt*: glTF for the web
    View, USD for Sim. An asset that ships no geometry (e.g. a mass-model reference asset) yields
    an empty list rather than an error. *require* forwards the demanded attestations to
    :func:`~astro_mine.fleet.packaging.hub.pull_asset` (default: a signed publish; pass ``()`` for
    an unsigned artifact).
    """
    want = GeometryFormat(fmt)
    doc = pull_asset(
        registry,
        reference,
        trusted_public_key_pem=trusted_public_key_pem,
        verify=verify,
        require=require,
    )
    return [ref for ref in doc.asset.geometry if ref.format is want]


def materialize_preview(
    registry: RegistryClient,
    reference: str,
    out_dir: str | Path,
    *,
    verify: bool = True,
    trusted_public_key_pem: bytes | None = None,
    require: Sequence[str] | None = None,
) -> Path:
    """Reconstruct a servable preview directory; return its SADF-JSON path (the documentUrl).

    Pulls *reference* from the Hub registry **by content hash** (verify-before-trust unless *verify*
    is ``False``), writes its canonical SADF JSON to *out_dir*, and writes every published geometry
    blob at the ref's **relative ``uri`` next to it** — so a static file server can hand the View
    ``<AssetPreview source={{documentUrl}}/>`` widget the path and Cesium resolves each glTF at
    ``new URL(ref.uri, documentUrl)`` (``fleet.md`` §6; RM-P1-VIEW-03). This is the byte-for-byte
    reference layout the Studio-side preview materializer (RM-P1-STUDIO) reproduces Hub-direct.

    Geometry bytes are mapped ``uri → blob digest`` via the Core manifest's
    :attr:`~astro_mine.core.registry.Provenance.source_content_hashes`; a ref whose file was absent
    at publish (no blob) is skipped, so a mass-model asset yields just the SADF JSON. A geometry
    ``uri`` that would escape *out_dir* is rejected (:class:`HubError`) — fail closed, like Hub's
    bundle extractor.
    """
    _require_hub()
    from astro_mine.hub.client import HubClient
    from astro_mine.hub.supply_chain import DEFAULT_REQUIRED


    client = HubClient(registry, trusted_public_key_pem=trusted_public_key_pem)
    demanded = tuple(DEFAULT_REQUIRED) if require is None else tuple(require)
    digest = (
        client.verify(reference, require=demanded) if verify else registry.resolve(reference).digest
    )

    manifest = registry.read_manifest(digest)
    sadf_json = next(
        (
            registry.pull_blob(layer["digest"])
            for layer in manifest["layers"]
            if layer["mediaType"] == oci.MEDIA_SADF_JSON
        ),
        None,
    )
    if sadf_json is None:
        raise HubError(f"artifact {reference} has no SADF JSON layer to materialize")
    doc = load_sadf(sadf_json)
    plugin_manifest = PluginManifest.model_validate_json(registry.read_config(digest))
    source_hashes = (
        dict(plugin_manifest.provenance.source_content_hashes) if plugin_manifest.provenance else {}
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    root = out.resolve()
    document = out / f"{doc.asset.identity.id}.sadf.json"
    document.write_bytes(sadf_json)
    for ref in doc.asset.geometry:
        blob_digest = source_hashes.get(ref.uri)
        if blob_digest is None:  # ref whose geometry file was absent at publish -> nothing to serve
            continue
        target = out / ref.uri
        if not target.resolve().is_relative_to(root):
            raise HubError(f"geometry uri {ref.uri!r} escapes the output directory")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(registry.pull_blob(blob_digest))
    return document
