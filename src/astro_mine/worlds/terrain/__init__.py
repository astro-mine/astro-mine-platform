"""Terrain ingest — real polar DEM to simulatable terrain (RM-P0-WORLDS-01).

LOLA DEM for the Shackleton-de Gerlache region → Cloud-Optimized GeoTIFF layers via
GDAL, reprojected to an explicit lunar body-fixed CRS (Core's
:class:`~astro_mine.core.units.PlanetaryCRS`, PROJ planetary ``+R``), with derived
slope / aspect / roughness layers and carried vertical / void-fill uncertainty. The
product is content-addressed: the same DEM + pinned toolchain reproduce the same
terrain hash.

This establishes the **reference grid and CRS** every later layer georeferences against
(illumination RM-P0-WORLDS-03, regolith RM-P0-WORLDS-05) and the deterministic,
hashable layers the WorldSpec bundle (RM-P0-WORLDS-07) composes. The Environment-API
provider and ``ray_intersect`` LOS service that *consume* this product are
RM-P0-WORLDS-06.

The pipeline runs on **any** GDAL-readable DEM with a CRS; the real Shackleton LOLA DEM
is fetched via ``scripts/fetch_shackleton_dem.py`` (documented, run outside CI).

Backlog: RM-P0-WORLDS-01 — https://github.com/astro-mine/astro-mine-worlds/issues/1
"""

from __future__ import annotations

import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio

from astro_mine.core.units import PlanetaryCRS, require_crs
from astro_mine.worlds.crs import LUNAR_SOUTH_POLAR_STEREOGRAPHIC, to_rasterio_crs
from astro_mine.worlds.terrain._ingest import RasterLayer, reproject_dem, write_cog
from astro_mine.worlds.terrain._layers import (
    fill_voids,
    normal_from_slope_aspect,
    roughness,
    slope_aspect,
    terrain_hash,
    vertical_uncertainty,
)
from astro_mine.worlds.terrain._synthetic import NODATA, SYNTHETIC_SOURCE_ID, synthesize_dem

__all__ = [
    "BUNDLE_SCHEMA",
    "LAYER_UNITS",
    "NODATA",
    "SYNTHETIC_SOURCE_ID",
    "SamplePoint",
    "TerrainModel",
    "TerrainProduct",
    "ingest_dem",
    "synthesize_dem",
]

BUNDLE_SCHEMA = "astro-mine-worlds/terrain/v0.1"

#: Explicit SI/units label per layer (carried in the manifest; conventions.md §5).
LAYER_UNITS: dict[str, str] = {
    "elevation": "m",
    "slope": "degree",
    "aspect": "degree",
    "roughness": "m",
    "vertical_uncertainty": "m",
    "void_mask": "bool",
}

_MANIFEST_NAME = "manifest.json"


@dataclass(frozen=True)
class TerrainProduct:
    """A reprojected, content-addressed terrain product on disk."""

    path: Path
    crs: PlanetaryCRS
    width: int
    height: int
    resolution_m: float
    transform: tuple[float, float, float, float, float, float]
    terrain_hash: str
    layers: dict[str, Path]
    manifest: dict[str, Any]


@dataclass(frozen=True)
class SamplePoint:
    """A point query against a terrain product."""

    elevation_m: float
    slope_deg: float
    aspect_deg: float
    normal: tuple[float, float, float]
    is_void: bool
    in_bounds: bool


def _worlds_version() -> str:
    try:
        return importlib.metadata.version("astro-mine-platform")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - always installed in dev/CI
        return "0+unknown"


def ingest_dem(
    source: str | Path,
    out_dir: str | Path,
    *,
    target_crs: PlanetaryCRS = LUNAR_SOUTH_POLAR_STEREOGRAPHIC,
    resolution_m: float = 20.0,
    baseline_vertical_uncertainty_m: float = 1.0,
    void_uncertainty_factor: float = 5.0,
) -> TerrainProduct:
    """Ingest a polar DEM into a reprojected, content-addressed COG terrain product.

    Reprojects ``source`` to ``target_crs`` at ``resolution_m``, derives slope / aspect /
    roughness, carries a vertical-uncertainty field and a void mask, writes one COG per
    layer plus a deterministic manifest, and returns the :class:`TerrainProduct`.
    """
    require_crs(target_crs)
    rep = reproject_dem(source, target_crs, resolution_m)

    filled = fill_voids(rep.elevation, rep.void)
    slope_deg, aspect_deg = slope_aspect(filled, resolution_m)
    layers_arr: dict[str, Any] = {
        "elevation": rep.elevation,
        "slope": slope_deg,
        "aspect": aspect_deg,
        "roughness": roughness(filled),
        "vertical_uncertainty": vertical_uncertainty(
            rep.void, baseline_vertical_uncertainty_m, void_uncertainty_factor
        ),
        "void_mask": rep.void.astype(np.uint8),
    }

    meta: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "crs": target_crs.model_dump(mode="json"),
        "grid": {
            "width": rep.width,
            "height": rep.height,
            "resolution_m": resolution_m,
            "transform": list(rep.transform),
        },
        "layers": {name: {"units": LAYER_UNITS[name]} for name in layers_arr},
        "uncertainty": {
            "baseline_vertical_uncertainty_m": baseline_vertical_uncertainty_m,
            "void_uncertainty_factor": void_uncertainty_factor,
        },
        "toolchain": {
            "astro_mine_worlds": _worlds_version(),
            "gdal": rasterio.__gdal_version__,
            "rasterio": rasterio.__version__,
            "numpy": np.__version__,
        },
    }
    digest = terrain_hash(layers_arr, meta)
    meta = {**meta, "terrain_hash": digest}

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dst_crs = to_rasterio_crs(target_crs)
    layer_paths: dict[str, Path] = {}
    for name, arr in layers_arr.items():
        layer_path = out / f"{name}.tif"
        nodata = float("nan") if name == "elevation" else None
        write_cog(layer_path, arr, rep.transform, dst_crs, nodata)
        layer_paths[name] = layer_path
    (out / _MANIFEST_NAME).write_text(
        json.dumps(meta, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    return TerrainProduct(
        path=out,
        crs=target_crs,
        width=rep.width,
        height=rep.height,
        resolution_m=resolution_m,
        transform=rep.transform,
        terrain_hash=digest,
        layers=layer_paths,
        manifest=meta,
    )


class TerrainModel:
    """Read access to a terrain product: point queries of elevation/slope/aspect/void.

    The minimal reader this issue ships. The Core Environment-API world provider and the
    ``ray_intersect`` LOS service that build on it are RM-P0-WORLDS-06.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.manifest: dict[str, Any] = json.loads(
            (self.path / _MANIFEST_NAME).read_text(encoding="utf-8")
        )
        self.crs = PlanetaryCRS.model_validate(self.manifest["crs"])
        self._layers: dict[str, RasterLayer] = {}

    @classmethod
    def open(cls, product: TerrainProduct | str | Path) -> TerrainModel:
        """Open a terrain product from a :class:`TerrainProduct` or its directory path."""
        path = product.path if isinstance(product, TerrainProduct) else product
        return cls(path)

    def _layer(self, name: str) -> RasterLayer:
        """The named layer, materialized in memory once and reused — no per-query reopen (#48)."""
        layer = self._layers.get(name)
        if layer is None:
            layer = RasterLayer.open(self.path / f"{name}.tif")
            self._layers[name] = layer
        return layer

    def sample(self, x: float, y: float) -> SamplePoint:
        """Sample the terrain at world coordinates ``(x, y)`` in the product's CRS."""
        elevation, in_bounds = self._layer("elevation").sample(x, y)
        slope, _ = self._layer("slope").sample(x, y)
        aspect, _ = self._layer("aspect").sample(x, y)
        void_value, _ = self._layer("void_mask").sample(x, y)
        is_void = bool(np.isnan(elevation)) or void_value >= 0.5
        normal = (
            normal_from_slope_aspect(slope, aspect)
            if in_bounds and not np.isnan(slope)
            else (0.0, 0.0, 1.0)
        )
        return SamplePoint(
            elevation_m=elevation,
            slope_deg=slope,
            aspect_deg=aspect,
            normal=normal,
            is_void=is_void,
            in_bounds=in_bounds,
        )
