"""Diviner / LEND / M³ conditioning-layer ingest (RM-P1-WORLDS-14).

Ingests the three PDS conditioning products Prospect conditions real priors on, reprojected
onto an existing world's CRS/grid so they co-register cell-for-cell with the LOLA DEM and PSR
mask (RM-P1-PROSPECT-12):

- **Diviner** bolometric/measured surface temperature (K) — the *measured* field, kept
  deliberately distinct from the RM-P0-WORLDS-04 1-D thermal *model* (data vs model);
- **LEND** epithermal-neutron count-rate → **water-equivalent hydrogen** (WEH, wt %) via a
  reduced-order suppression conversion;
- **M³** surficial band-depth → **OH/H₂O** abundance (wt %) via a reduced-order linear map.

A layer is a **source adapter + field layer**, not a Worlds/Core narrow-waist change: each is a
reprojected COG on the shared grid, catalogued in STAC with explicit CRS + per-product
provenance, and read back through :class:`ConditioningField` (which structurally satisfies the
provider's conditioning-source seam). Real PDS rasters are fetched outside CI; the pipeline runs
on any GDAL-readable raster with a CRS.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from numpy.typing import NDArray

from astro_mine.core.units import PlanetaryCRS
from astro_mine.worlds.crs import lunar_geographic_proj4, to_rasterio_crs
from astro_mine.worlds.spec._stac import StacLayer, write_stac_catalog
from astro_mine.worlds.terrain import TerrainModel, TerrainProduct
from astro_mine.worlds.terrain._ingest import read_layer, reproject_onto_grid, write_cog

__all__ = [
    "CONDITIONING_SPECS",
    "DIVINER_TEMPERATURE",
    "LEND_WEH",
    "M3_WATER",
    "ConditioningField",
    "ConditioningLayer",
    "ConditioningLayerSet",
    "ConditioningSpec",
    "ingest_conditioning_layers",
    "lend_epithermal_to_weh",
    "m3_band_depth_to_water",
]

BUNDLE_SCHEMA = "astro-mine-worlds/conditioning/v0.1"
F32 = np.float32


def lend_epithermal_to_weh(
    count_rate: NDArray[np.float32],
    *,
    dry_count_rate: float = 5.0,
    max_weh_wt_percent: float = 5.0,
) -> NDArray[np.float32]:
    """Reduced-order LEND epithermal-neutron → water-equivalent-hydrogen (WEH, wt %).

    Hydrogen suppresses epithermal neutrons, so WEH rises as the count rate falls below the
    hydrogen-free ``dry_count_rate``: ``WEH = max_weh * clip((dry - rate)/dry, 0, 1)`` — a
    documented monotone-decreasing stand-in for the real LEND retrieval, giving 0 at the dry
    rate and ``max_weh_wt_percent`` at zero count (lunar-plausible polar WEH of a few wt %).
    The point is a real, co-registered field Prospect can condition on, not the retrieval
    physics.
    """
    fraction = np.clip((dry_count_rate - count_rate) / dry_count_rate, 0.0, 1.0)
    weh: NDArray[np.float32] = (max_weh_wt_percent * fraction).astype(F32)
    return weh


def m3_band_depth_to_water(
    band_depth: NDArray[np.float32],
    *,
    scale_wt_percent: float = 0.5,
    max_water_wt_percent: float = 1.0,
) -> NDArray[np.float32]:
    """Reduced-order M³ 2.8-3.0 µm band-depth → surficial OH/H₂O abundance (wt %).

    A monotone-increasing linear map ``water = scale * band_depth`` clipped to
    ``[0, max_water_wt_percent]`` — the reduced-order stand-in for the real M³ retrieval.
    """
    water: NDArray[np.float32] = np.clip(
        scale_wt_percent * band_depth, 0.0, max_water_wt_percent
    ).astype(F32)
    return water


@dataclass(frozen=True)
class ConditioningSpec:
    """A conditioning layer's identity + its source→value conversion.

    ``convert`` maps the raw reprojected source raster to the layer's ``units`` (identity for
    an already-physical field like Diviner temperature). ``role`` is the semantic tag Prospect
    keys its prior recipe on (``measured_temperature`` / ``weh`` / ``water``).
    """

    name: str
    units: str
    instrument: str
    product: str
    role: str
    convert: Callable[[NDArray[np.float32]], NDArray[np.float32]]


DIVINER_TEMPERATURE = ConditioningSpec(
    name="diviner_temperature",
    units="K",
    instrument="Diviner",
    product="LRO Diviner bolometric brightness temperature",
    role="measured_temperature",
    convert=lambda a: a.astype(F32),
)

LEND_WEH = ConditioningSpec(
    name="lend_weh",
    units="wt_percent",
    instrument="LEND",
    product="LRO LEND epithermal-neutron count rate",
    role="weh",
    convert=lend_epithermal_to_weh,
)

M3_WATER = ConditioningSpec(
    name="m3_water",
    units="wt_percent",
    instrument="M3",
    product="Chandrayaan-1 M³ surficial OH/H₂O band depth",
    role="water",
    convert=m3_band_depth_to_water,
)

CONDITIONING_SPECS: dict[str, ConditioningSpec] = {
    spec.name: spec for spec in (DIVINER_TEMPERATURE, LEND_WEH, M3_WATER)
}


@dataclass(frozen=True)
class ConditioningLayer:
    """One ingested conditioning layer on disk, with its per-product provenance."""

    name: str
    path: Path
    units: str
    instrument: str
    role: str
    source_hash: str


@dataclass(frozen=True)
class ConditioningLayerSet:
    """The ingested conditioning layers for a world, co-registered to its terrain grid."""

    path: Path
    crs: PlanetaryCRS
    layers: dict[str, ConditioningLayer]
    stac_catalog: Path
    manifest: dict[str, Any]


def _worlds_version() -> str:
    try:
        return importlib.metadata.version("astro-mine-platform")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - always installed in dev/CI
        return "0+unknown"


def _hash_file(path: str | Path) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return f"sha256:{h.hexdigest()}"


def _terrain_grid(
    terrain: TerrainModel | TerrainProduct,
) -> tuple[PlanetaryCRS, tuple[float, float, float, float, float, float], int, int]:
    """The ``(crs, transform, width, height)`` grid a conditioning layer co-registers to."""
    if isinstance(terrain, TerrainProduct):
        return terrain.crs, terrain.transform, terrain.width, terrain.height
    grid = terrain.manifest["grid"]
    t = grid["transform"]
    transform = (t[0], t[1], t[2], t[3], t[4], t[5])
    return terrain.crs, transform, int(grid["width"]), int(grid["height"])


def ingest_conditioning_layers(
    sources: Mapping[str, str | Path],
    terrain: TerrainModel | TerrainProduct,
    out_dir: str | Path,
    *,
    datetime_iso: str = "2009-06-23T00:00:00Z",
) -> ConditioningLayerSet:
    """Ingest conditioning-layer source rasters onto ``terrain``'s CRS/grid.

    ``sources`` maps a :data:`CONDITIONING_SPECS` name (``diviner_temperature`` / ``lend_weh``
    / ``m3_water``) to a GDAL-readable source raster. Each is reprojected onto the terrain
    grid (co-registered with the LOLA DEM + PSR mask), converted to its physical units, written
    as a COG with per-product provenance, and catalogued in STAC. Returns the
    :class:`ConditioningLayerSet`. Unknown layer names raise ``KeyError``.
    """
    crs, transform, width, height = _terrain_grid(terrain)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dst_crs = to_rasterio_crs(crs)

    layers: dict[str, ConditioningLayer] = {}
    stac_layers: list[StacLayer] = []
    for name, source in sources.items():
        spec = CONDITIONING_SPECS[name]
        raw = reproject_onto_grid(source, crs, transform, width, height)
        values = spec.convert(raw)
        # Preserve voids (no source coverage) as NaN through the conversion.
        values = np.where(np.isnan(raw), np.nan, values).astype(F32)
        layer_path = out / f"{name}.tif"
        write_cog(layer_path, values, transform, dst_crs, float("nan"))
        layers[name] = ConditioningLayer(
            name=name,
            path=layer_path,
            units=spec.units,
            instrument=spec.instrument,
            role=spec.role,
            source_hash=_hash_file(source),
        )
        stac_layers.append(
            StacLayer(
                item_id=name,
                asset_href=f"./{name}.tif",
                title=f"{spec.instrument} — {spec.product}",
                units=spec.units,
            )
        )

    proj4 = crs.projection or lunar_geographic_proj4(float(crs.reference_radius_m))
    manifest: dict[str, Any] = {
        "schema": BUNDLE_SCHEMA,
        "crs": crs.model_dump(mode="json"),
        "grid": {"width": width, "height": height, "transform": list(transform)},
        "layers": {
            name: {
                "units": layer.units,
                "instrument": layer.instrument,
                "role": CONDITIONING_SPECS[name].role,
                "product": CONDITIONING_SPECS[name].product,
                "source_hash": layer.source_hash,
            }
            for name, layer in layers.items()
        },
        "toolchain": {
            "astro_mine_worlds": _worlds_version(),
            "gdal": rasterio.__gdal_version__,
            "numpy": np.__version__,
        },
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    catalog = write_stac_catalog(
        out / "stac",
        world_id="conditioning-shackleton-de-gerlache",
        description="RM-P1-WORLDS-14 conditioning layers (Diviner/LEND/M³) on the DEM grid.",
        proj4=proj4,
        shape=(height, width),
        transform=transform,
        datetime_iso=datetime_iso,
        layers=stac_layers,
    )
    return ConditioningLayerSet(
        path=out, crs=crs, layers=layers, stac_catalog=catalog, manifest=manifest
    )


class ConditioningField:
    """Read access to an ingested conditioning-layer set — the field Prospect conditions on.

    Same CRS/grid as the terrain product it was ingested against, so ``sample`` returns each
    layer's co-registered value at a world coordinate. Structurally satisfies the
    :class:`~astro_mine.worlds.provider.ConditioningSource` seam the world provider exposes.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.manifest: dict[str, Any] = json.loads(
            (self.path / "manifest.json").read_text(encoding="utf-8")
        )
        self.crs = PlanetaryCRS.model_validate(self.manifest["crs"])
        self._layer_paths = {name: self.path / f"{name}.tif" for name in self.manifest["layers"]}

    @classmethod
    def open(cls, layer_set: ConditioningLayerSet | str | Path) -> ConditioningField:
        """Open a conditioning field from a :class:`ConditioningLayerSet` or its directory."""
        path = layer_set.path if isinstance(layer_set, ConditioningLayerSet) else layer_set
        return cls(path)

    @property
    def layers(self) -> tuple[str, ...]:
        """The conditioning-layer names available (Diviner/LEND/M³)."""
        return tuple(self._layer_paths)

    def sample(self, x: float, y: float) -> dict[str, float]:
        """Every conditioning layer's value at world coordinate ``(x, y)`` (NaN off-grid)."""
        return {name: read_layer(path, x, y)[0] for name, path in self._layer_paths.items()}

    def sample_layer(self, name: str, x: float, y: float) -> float:
        """One conditioning layer's value at ``(x, y)``. Raises ``KeyError`` if unknown."""
        return read_layer(self._layer_paths[name], x, y)[0]
