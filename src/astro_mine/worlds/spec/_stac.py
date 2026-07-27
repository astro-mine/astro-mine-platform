"""A minimal, deterministic STAC 1.0.0 catalog writer (RM-P0-WORLDS-07; worlds.md §5).

Worlds catalogs its layers with **STAC** so the bundle is browsable/discoverable. This writes a
spec-compliant STAC *core* Catalog + one Item per raster layer by hand — no library — so the JSON
is byte-reproducible (the determinism gate, conventions.md §1.5). Spatial metadata for the
non-Earth planetary CRS is carried in an ``astromine:`` property namespace (the projected
transform/shape + the PROJ string) rather than GeoJSON lon/lat, which is degenerate at the pole;
geometry is therefore ``null`` (valid STAC core for a non-WGS84 raster). Richer STAC
(the projection extension, a Collection, stac-validator conformance in CI) is a follow-up.

Backlog: RM-P0-WORLDS-07 — https://github.com/astro-mine/astro-mine-worlds/issues/7
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = ["COG_MEDIA_TYPE", "STAC_VERSION", "StacLayer", "write_stac_catalog"]

STAC_VERSION = "1.0.0"

#: The Cloud-Optimized GeoTIFF media type (the layer assets are COGs, RM-P0-WORLDS-01/05).
COG_MEDIA_TYPE = "image/tiff; application=geotiff; profile=cloud-optimized"


@dataclass(frozen=True)
class StacLayer:
    """One catalog layer: a STAC Item id and the asset it points at.

    ``media_type`` defaults to the COG type the 2-D raster layers use; a chunked N-D field layer
    (worlds.md §5's Zarr store — the horizon map, the thermal curve stack) passes
    :data:`~astro_mine.worlds.fields.ZARR_MEDIA_TYPE` so the catalog says which store a
    consumer is about to range-read, rather than mislabelling a Zarr directory as a GeoTIFF.
    """

    item_id: str
    asset_href: str  # relative to the STAC directory (e.g. "../terrain/elevation.tif")
    title: str
    units: str
    roles: tuple[str, ...] = ("data",)
    media_type: str = COG_MEDIA_TYPE


def _dump(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def write_stac_catalog(
    stac_dir: str | Path,
    *,
    world_id: str,
    description: str,
    proj4: str,
    shape: tuple[int, int],
    transform: tuple[float, float, float, float, float, float],
    datetime_iso: str,
    layers: list[StacLayer],
) -> Path:
    """Write a STAC Catalog (``catalog.json``) + one Item per layer; return the catalog path.

    All hrefs are relative, so the catalog is location-independent; all JSON is key-sorted with a
    fixed indent, so two builds of the same world produce byte-identical files.
    """
    out = Path(stac_dir)
    out.mkdir(parents=True, exist_ok=True)

    for layer in layers:
        item = {
            "type": "Feature",
            "stac_version": STAC_VERSION,
            "id": layer.item_id,
            "geometry": None,
            "properties": {
                "datetime": datetime_iso,
                "title": layer.title,
                "astromine:units": layer.units,
                "astromine:proj4": proj4,
                "astromine:shape": [shape[0], shape[1]],
                "astromine:transform": list(transform),
            },
            "links": [
                {"rel": "root", "href": "./catalog.json", "type": "application/json"},
                {"rel": "parent", "href": "./catalog.json", "type": "application/json"},
                {"rel": "self", "href": f"./{layer.item_id}.json", "type": "application/geo+json"},
            ],
            "assets": {
                "data": {
                    "href": layer.asset_href,
                    "type": layer.media_type,
                    "title": layer.title,
                    "roles": list(layer.roles),
                }
            },
        }
        _dump(out / f"{layer.item_id}.json", item)

    catalog: dict[str, Any] = {
        "type": "Catalog",
        "stac_version": STAC_VERSION,
        "id": world_id,
        "description": description,
        "links": [
            {"rel": "root", "href": "./catalog.json", "type": "application/json"},
            {"rel": "self", "href": "./catalog.json", "type": "application/json"},
            *[
                {
                    "rel": "item",
                    "href": f"./{layer.item_id}.json",
                    "type": "application/geo+json",
                }
                for layer in layers
            ],
        ],
    }
    catalog_path = out / "catalog.json"
    _dump(catalog_path, catalog)
    return catalog_path
