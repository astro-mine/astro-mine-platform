# SPDX-License-Identifier: Apache-2.0
"""A synthetic DEM generator — so the shipped WorldSpec example has a copyable input (issue #60).

`synthetic_polar.world.yaml` is a spec you can copy, validate and hash without downloading anything.
It was **not** a spec you could build: building starts at
:func:`~astro_mine.worlds.terrain.ingest_dem`, which needs a raster, and no public API produced one.
Worlds' own tests fabricated one through a private `conftest` fixture that an installed wheel does
not expose — so the example's promise ("a 10 km x 10 km lunar south-polar basin … needs neither
the LOLA DEM nor SPICE kernels") held for authoring and validating a spec and broke at the build.

:func:`synthesize_dem` closes that: it writes a GeoTIFF **in the target projected CRS**, covering a
region you name, that `ingest_dem` consumes like any other DEM. It is the same pipeline the real
LOLA product runs through — nothing here is a special build path.

**It is a stand-in, not science.** The surface is an analytic basin: a bowl, a rim, a couple of
craters and a little roughness, chosen to exercise slope / aspect / roughness / void-fill rather
than to resemble any real terrain. Nothing sampled it from a spacecraft, so a bundle built from it
must never be published as a model of a real place — hence :data:`SYNTHETIC_SOURCE_ID`, which
stamps that into the product's provenance where a reviewer will see it.

Deterministic by construction: the surface is a closed-form function of the grid plus a seeded
`numpy` generator, so the same arguments produce byte-identical output and a bundle built from it
has a stable ``world_hash`` (CX-REPRO; conventions.md §11).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from astro_mine.core.units import PlanetaryCRS, require_crs
from astro_mine.worlds.crs import LUNAR_SOUTH_POLAR_STEREOGRAPHIC, to_rasterio_crs

__all__ = [
    "SYNTHETIC_SOURCE_ID",
    "synthesize_dem",
]

#: The `source_dem.id` a spec should declare when its terrain came from :func:`synthesize_dem`.
#: A world built from a synthesized DEM is illustrative, and its provenance has to say so — the same
#: reason `SourceRef.content_hash` is ``None`` for a synthetic source rather than a made-up digest.
SYNTHETIC_SOURCE_ID = "synthetic-polar-dem"

#: The nodata value written for void cells — GDAL's conventional Int16 sentinel, matching what the
#: real LOLA products use, so `ingest_dem`'s void handling is exercised rather than bypassed.
NODATA = -32768.0

_DEFAULT_FLOOR_M = -1_000.0
_DEFAULT_RELIEF_M = 900.0


def _surface(
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    floor_m: float,
    relief_m: float,
    roughness_m: float,
    seed: int,
) -> np.ndarray:
    """A closed-form polar-basin surface over normalized coordinates in ``[-1, 1]``.

    Four terms, each earning its place in what the ingest pipeline then derives:

    * a **paraboloid basin**, so elevation has a large-scale gradient and slope/aspect vary smoothly
      and predictably;
    * a **raised rim** near the edge, so the domain has both a floor and a high stand — a PSR mask
      computed over it has somewhere to be shadowed *by*;
    * two **craters** of different radii, which put steep local walls in the field (the case where
      slope and roughness disagree with the basin's gentle trend);
    * **seeded fine noise**, so `roughness` is not identically zero.
    """
    radius = np.hypot(xx, yy)
    basin = floor_m + relief_m * radius**2
    rim = 0.35 * relief_m * np.exp(-(((radius - 0.92) / 0.06) ** 2))
    craters = np.zeros_like(basin)
    for centre_x, centre_y, crater_radius, depth in (
        (-0.35, 0.28, 0.16, 0.22),
        (0.42, -0.31, 0.10, 0.15),
    ):
        distance = np.hypot(xx - centre_x, yy - centre_y) / crater_radius
        # A bowl inside the radius, a small raised ejecta lip just outside it.
        bowl = -depth * relief_m * np.clip(1.0 - distance**2, 0.0, None)
        lip = 0.06 * relief_m * np.exp(-((distance - 1.15) ** 2) / 0.05)
        craters = craters + bowl + lip
    noise = np.random.default_rng(seed).normal(0.0, roughness_m, size=basin.shape)
    return np.asarray(basin + rim + craters + noise, dtype=np.float64)


def synthesize_dem(
    path: str | Path,
    *,
    crs: PlanetaryCRS = LUNAR_SOUTH_POLAR_STEREOGRAPHIC,
    min_x_m: float = -5_000.0,
    min_y_m: float = -5_000.0,
    max_x_m: float = 5_000.0,
    max_y_m: float = 5_000.0,
    resolution_m: float = 20.0,
    floor_m: float = _DEFAULT_FLOOR_M,
    relief_m: float = _DEFAULT_RELIEF_M,
    roughness_m: float = 1.5,
    void_fraction: float = 0.02,
    seed: int = 0,
) -> Path:
    """Write a synthetic DEM over ``[min_x_m, max_x_m) x [min_y_m, max_y_m)``; return its path.

    The raster is written **in** ``crs`` — the same CRS a WorldSpec declares — so
    :func:`~astro_mine.worlds.terrain.ingest_dem` reprojects it to itself and the realized grid
    matches the region you asked for. The defaults are the shipped
    `synthetic_polar.world.yaml` example's region and resolution, so the zero-argument call is
    exactly the input that example was missing.

    ``void_fraction`` punches a contiguous no-data patch of roughly that share of the grid, because
    a DEM with no voids does not exercise `fill_voids` or the void-uncertainty field — the real
    product has them, so the stand-in should too. Pass ``0.0`` for a gap-free surface.

    A stand-in, not science: see the module docstring, and declare
    :data:`SYNTHETIC_SOURCE_ID` as the spec's ``source_dem.id``.

    :raises ValueError: if the region is empty, the resolution non-positive, or the CRS is not an
        explicit planetary CRS (:func:`~astro_mine.core.units.require_crs` — an implicit Earth datum
        on a lunar body is a defaulting bug, not a default).
    """
    require_crs(crs)
    if max_x_m <= min_x_m or max_y_m <= min_y_m:
        raise ValueError(
            f"empty region: max ({max_x_m}, {max_y_m}) must exceed min ({min_x_m}, {min_y_m})"
        )
    if resolution_m <= 0.0:
        raise ValueError(f"resolution_m must be positive, got {resolution_m}")
    if not 0.0 <= void_fraction < 1.0:
        raise ValueError(f"void_fraction must be in [0, 1), got {void_fraction}")

    width = max(1, math.ceil((max_x_m - min_x_m) / resolution_m))
    height = max(1, math.ceil((max_y_m - min_y_m) / resolution_m))

    # Normalized grid coordinates in [-1, 1], so the surface's shape is resolution-independent: the
    # same basin at 20 m and at 5 m, sampled more finely.
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float64)
    xx = 2.0 * (cols + 0.5) / width - 1.0
    yy = 2.0 * (rows + 0.5) / height - 1.0

    elevation = _surface(
        xx, yy, floor_m=floor_m, relief_m=relief_m, roughness_m=roughness_m, seed=seed
    ).astype(np.float32)

    if void_fraction > 0.0:
        # One contiguous square patch, offset from centre so it lands on the basin wall rather
        # than in the flat middle — a void where the terrain varies is the harder fill case.
        side = max(1, round(math.sqrt(void_fraction * width * height)))
        row0 = min(max(0, height // 3), max(0, height - side))
        col0 = min(max(0, width // 4), max(0, width - side))
        elevation[row0 : row0 + side, col0 : col0 + side] = NODATA

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        out,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="float32",
        crs=to_rasterio_crs(crs),
        transform=from_bounds(min_x_m, min_y_m, max_x_m, max_y_m, width, height),
        nodata=NODATA,
    ) as dst:
        dst.write(elevation, 1)
    return out
