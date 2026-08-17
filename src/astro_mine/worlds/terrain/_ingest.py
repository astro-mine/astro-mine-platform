# SPDX-License-Identifier: Apache-2.0
"""Reprojection and COG IO for terrain ingest (RM-P0-WORLDS-01).

The rasterio/GDAL boundary: open a GDAL-readable DEM, reproject it to the explicit
lunar CRS, and read/write Cloud-Optimized GeoTIFF layers. Kept apart from the pure
NumPy kernels (``_layers``) so the IO surface is small and the rest stays engine-neutral.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rasterio.crs
from numpy.typing import NDArray
from rasterio.transform import rowcol
from rasterio.warp import Resampling, calculate_default_transform, reproject

from astro_mine.core.units import PlanetaryCRS, require_crs
from astro_mine.worlds.crs import to_rasterio_crs

__all__ = [
    "RasterLayer",
    "ReprojectedDem",
    "TerrainIngestError",
    "read_layer",
    "reproject_dem",
    "reproject_onto_grid",
    "write_cog",
]


class TerrainIngestError(Exception):
    """Raised when a source DEM cannot be ingested (e.g. it carries no CRS)."""


@dataclass(frozen=True)
class ReprojectedDem:
    """A DEM reprojected onto the target CRS grid; void cells are NaN in ``elevation``."""

    elevation: NDArray[np.float32]
    void: NDArray[np.bool_]
    transform: tuple[float, float, float, float, float, float]
    width: int
    height: int


def reproject_dem(
    source: str | Path, target_crs: PlanetaryCRS, resolution_m: float
) -> ReprojectedDem:
    """Reproject a GDAL-readable DEM to ``target_crs`` at ``resolution_m`` pixel spacing.

    The source MUST carry an explicit CRS — a DEM with none is rejected loudly rather
    than assumed to be Earth/WGS84 (conventions.md §5). Cells with no source coverage or
    a source nodata value become voids (NaN), flagged in the returned mask.
    """
    require_crs(target_crs)
    dst_crs = to_rasterio_crs(target_crs)
    with rasterio.open(str(source)) as src:
        if src.crs is None:
            raise TerrainIngestError(
                f"source DEM {source!r} has no CRS; an explicit planetary CRS is required "
                "(no implicit Earth/WGS84)"
            )
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds, resolution=resolution_m
        )
        source_band = src.read(1).astype(np.float32)
        src_nodata = src.nodata
        destination = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=source_band,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src_nodata,
            dst_transform=transform,
            dst_crs=dst_crs,
            dst_nodata=float("nan"),
            resampling=Resampling.bilinear,
        )
    void = np.isnan(destination)
    return ReprojectedDem(
        elevation=destination,
        void=void,
        transform=tuple(transform)[:6],
        width=int(width),
        height=int(height),
    )


def reproject_onto_grid(
    source: str | Path,
    target_crs: PlanetaryCRS,
    transform: tuple[float, float, float, float, float, float],
    width: int,
    height: int,
    *,
    resampling: Resampling = Resampling.bilinear,
) -> NDArray[np.float32]:
    """Reproject a GDAL-readable raster onto a **fixed** target grid (co-registration).

    Unlike :func:`reproject_dem` (which derives its own default transform), this warps the
    source onto the exact ``(target_crs, transform, width, height)`` grid of an existing
    terrain product, so an ingested conditioning layer (Diviner/LEND/M³, RM-P1-WORLDS-14)
    co-registers cell-for-cell with the LOLA DEM and PSR mask. The source MUST carry an
    explicit CRS (no implicit Earth/WGS84); uncovered/nodata cells become NaN.
    """
    require_crs(target_crs)
    dst_crs = to_rasterio_crs(target_crs)
    with rasterio.open(str(source)) as src:
        if src.crs is None:
            raise TerrainIngestError(
                f"source raster {source!r} has no CRS; an explicit planetary CRS is required "
                "(no implicit Earth/WGS84)"
            )
        band = src.read(1).astype(np.float32)
        destination = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=band,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=rasterio.transform.Affine(*transform),
            dst_crs=dst_crs,
            dst_nodata=float("nan"),
            resampling=resampling,
        )
    return destination


def write_cog(
    path: Path,
    array: NDArray[Any],
    transform: tuple[float, float, float, float, float, float],
    crs: rasterio.crs.CRS,
    nodata: float | None,
) -> None:
    """Write a single-band Cloud-Optimized GeoTIFF (GDAL ``COG`` driver)."""
    with rasterio.open(
        str(path),
        "w",
        driver="COG",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=str(array.dtype),
        crs=crs,
        transform=rasterio.transform.Affine(*transform),
        nodata=nodata,
    ) as dst:
        dst.write(array, 1)


@dataclass(frozen=True)
class RasterLayer:
    """A single-band layer materialized in memory for repeated point sampling.

    :func:`read_layer` re-opens the COG on **every** query — a ``rasterio.open`` per point that
    dominates the world-provider tick (a ``sample()`` fans out to nine of them; issue #48).
    Opening once into this holder and sampling in-memory answers point queries without touching
    the filesystem again, with semantics **identical** to
    :meth:`rasterio.DatasetReader.sample` on a single ``(x, y)``: an inclusive-bounds reject,
    then a floored ``rowcol`` index, then the pixel value — or the layer ``nodata`` for a point
    inside the bounds but off the sampled array edge (matching ``rasterio``'s ``sample_gen``).

    The materialized array is invariant for the life of a :class:`TerrainModel` /
    :class:`~astro_mine.worlds.regolith.RegolithField`, so the substitution is exact and the
    provider's determinism contract is unchanged. Sized for the anchor's ~1264² f32 grids
    (tens of MB per layer); not intended for out-of-core rasters.
    """

    array: NDArray[Any]
    transform: Any
    bounds: tuple[float, float, float, float]
    height: int
    width: int
    nodata: float

    @classmethod
    def open(cls, path: str | Path) -> RasterLayer:
        """Materialize the single-band layer at ``path`` (the one and only ``rasterio.open``)."""
        with rasterio.open(str(path)) as ds:
            left, bottom, right, top = ds.bounds
            return cls(
                array=ds.read(1),
                transform=ds.transform,
                bounds=(float(left), float(bottom), float(right), float(top)),
                height=int(ds.height),
                width=int(ds.width),
                nodata=float(ds.nodata) if ds.nodata is not None else 0.0,
            )

    def sample(self, x: float, y: float) -> tuple[float, bool]:
        """Sample at world ``(x, y)`` → ``(value, in_bounds)`` — as :func:`read_layer` would."""
        left, bottom, right, top = self.bounds
        if not (left <= x <= right and bottom <= y <= top):
            return (float("nan"), False)
        row, col = rowcol(self.transform, x, y)
        row, col = int(row), int(col)
        if 0 <= row < self.height and 0 <= col < self.width:
            return (float(self.array[row, col]), True)
        return (self.nodata, True)


def read_layer(path: Path, x: float, y: float) -> tuple[float, bool]:
    """Sample a layer at world coordinates ``(x, y)``; return ``(value, in_bounds)``.

    Opens ``path`` on every call. A caller that samples one layer repeatedly (the world
    provider, per tick) should materialize it once with :class:`RasterLayer` instead — see
    issue #48."""
    with rasterio.open(str(path)) as ds:
        left, bottom, right, top = ds.bounds
        if not (left <= x <= right and bottom <= y <= top):
            return (float("nan"), False)
        value = next(ds.sample([(x, y)]))[0]
        return (float(value), True)
