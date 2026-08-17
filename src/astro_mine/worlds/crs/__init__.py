# SPDX-License-Identifier: Apache-2.0
"""Planetary CRS for the lunar south-polar world (RM-P0-WORLDS-01).

Worlds reuses Core's :class:`~astro_mine.core.units.PlanetaryCRS` vocabulary rather than
inventing its own — the body, body-fixed frame, PROJ planetary reference radius (``+R``),
and an explicit projection string. The canonical anchor-scenario CRS is a lunar
**south-polar stereographic** projection on the Moon sphere (R = 1 737 400 m). No field
ever defaults to an Earth/WGS84 value (conventions.md §5).

**Soft coupling — one Moon radius (RM-P1-WORLDS-17).** ``PlanetaryCRS.reference_radius_m`` and
:data:`astro_mine.spice.MOON_RADIUS_M` (``astro-mine-spice`` ``_geometry.py``; ``1_737_400.0`` m)
carry the *same* physical quantity and **must not disagree** — a CRS on a different sphere than the
one SPICE resolves geometry against would silently mis-place every anchor. There is no runtime
enforcement (RFC-0002 keeps the constant in ``astro-mine-spice``, not Core), so the invariant is
held by *construction*: this module imports the radius from ``astro_mine.spice`` and threads that
one value into both the ``+R`` PROJ term and ``reference_radius_m`` below, rather than re-typing the
literal. Do not hard-code ``1737400`` here; source it from the import.

The shared CRS/grid every later layer georeferences against — illumination & PSR
(RM-P0-WORLDS-03), regolith fields (RM-P0-WORLDS-05), and the WorldSpec bundle
(RM-P0-WORLDS-07) all align to the grid established here.
"""

from __future__ import annotations

import rasterio.crs
import rasterio.warp

from astro_mine.core.units import MOON, MOON_BODY_FIXED, PlanetaryCRS
from astro_mine.spice import MOON_RADIUS_M

__all__ = [
    "LUNAR_SOUTH_POLAR_STEREOGRAPHIC",
    "MOON_RADIUS_M",
    "lunar_geographic_proj4",
    "to_lonlat",
    "to_rasterio_crs",
]

#: The canonical anchor-scenario CRS: lunar south-polar stereographic on the Moon sphere.
LUNAR_SOUTH_POLAR_STEREOGRAPHIC = PlanetaryCRS(
    body=MOON,
    body_fixed_frame=MOON_BODY_FIXED.name,
    reference_radius_m=MOON_RADIUS_M,
    projection=(
        f"+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 "
        f"+x_0=0 +y_0=0 +R={MOON_RADIUS_M:.1f} +units=m +no_defs"
    ),
)


def lunar_geographic_proj4(reference_radius_m: float = MOON_RADIUS_M) -> str:
    """A lunar body-fixed geographic (lat/lon) PROJ string on the Moon sphere."""
    return f"+proj=longlat +R={reference_radius_m:.1f} +no_defs"


def to_rasterio_crs(crs: PlanetaryCRS) -> rasterio.crs.CRS:
    """Convert a Core :class:`PlanetaryCRS` to a rasterio CRS.

    Uses the explicit ``projection`` PROJ string for a projected CRS, or a lunar
    geographic (lat/lon) CRS on the body's reference sphere when ``projection`` is None.
    """
    proj4 = crs.projection or lunar_geographic_proj4(crs.reference_radius_m)
    return rasterio.crs.CRS.from_proj4(proj4)


def to_lonlat(crs: PlanetaryCRS, x_m: float, y_m: float) -> tuple[float, float]:
    """Inverse-project a projected map coordinate to body-fixed geographic degrees.

    The inverse of the CRS's ``projection``, onto the body's ``+R`` reference sphere.
    Returns ``(longitude_deg, latitude_deg)``. A CRS with no ``projection`` is already
    geographic, so its coordinates pass through unchanged.

    Goes through rasterio's bundled PROJ (the same path
    :mod:`astro_mine.worlds.illumination` and :mod:`astro_mine.worlds.provider` use) rather
    than a hand-rolled inverse, so every projection Worlds can *emit* it can also invert.
    """
    if crs.projection is None:
        return float(x_m), float(y_m)
    geographic = rasterio.crs.CRS.from_proj4(lunar_geographic_proj4(crs.reference_radius_m))
    lons, lats = rasterio.warp.transform(
        to_rasterio_crs(crs), geographic, [float(x_m)], [float(y_m)]
    )
    return float(lons[0]), float(lats[0])
