# SPDX-License-Identifier: Apache-2.0
"""DSN ground-station contact windows (LINK-02): when an Earth antenna can see the target.

A ground station is an Earth :class:`~astro_mine.spice.Site` plus a minimum-elevation mask. A
contact is open while the target (the relay orbiter, or the Moon) sits at or above that mask
in the station's local topocentric frame — the standard antenna-pointing horizon, *not* a
terrain DEM (that is the Moon-side relay path's concern, link.md §2.1). Topocentric elevation
comes from the shared :mod:`astro_mine.spice` foundation (``body_geometry``), kept injectable
behind :class:`TopocentricProvider` so the window logic is unit-testable without kernels.

Phase-0 baseline: DSN antennas as user-supplied stations. A richer ground-station catalog
(ESTRACK, custom) is P1; deep-space DSN scheduling and light-time is P3.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

from astro_mine import spice
from astro_mine.core.units import EARTH, Epoch, EpochWindow, FrameClass, ReferenceFrame
from astro_mine.link.windows._search import search_windows
from astro_mine.link.windows._window import ContactWindow
from astro_mine.spice import EARTH_RADIUS_M

__all__ = [
    "EARTH_BODY_FIXED",
    "EARTH_RADIUS_M",
    "GroundStation",
    "SpiceTopocentric",
    "TopocentricProvider",
    "dsn_contact_windows",
]

#: Earth's body-fixed frame for station geometry. ``IAU_EARTH`` is the always-available
#: built-in (a standard PCK furnishes it); the high-accuracy ``ITRF93`` is an opt-in
#: override for oracle-grade pass times (LINK-05).
EARTH_BODY_FIXED = ReferenceFrame(name="IAU_EARTH", frame_class=FrameClass.BODY_FIXED, center=EARTH)

# The Earth reference radius comes from the shared astro_mine.spice datum (RFC-0002: body
# reference radii are geometry, held once in Spice), re-exported here so callers can keep
# importing it from Link. A station's exact ITRF position barely shifts the elevation of a
# target ~384,000 km away; precise station locations are LINK-05.


class TopocentricProvider(Protocol):
    """Resolves the elevation (deg) of ``target`` above ``site``'s local horizontal at ``epoch``."""

    def elevation_deg(self, target: str, site: spice.Site, epoch: Epoch) -> float:
        """Elevation of ``target`` above ``site``'s local horizontal, in degrees."""
        ...


class SpiceTopocentric:
    """:class:`TopocentricProvider` backed by :func:`astro_mine.spice.body_geometry`.

    Kernels must be furnished first; a missing kernel raises loudly through
    :mod:`astro_mine.spice` rather than yielding a guessed elevation.
    """

    def __init__(self, *, abcorr: str | None = None) -> None:
        self._abcorr = abcorr or spice.DEFAULT_ABCORR

    def elevation_deg(self, target: str, site: spice.Site, epoch: Epoch) -> float:
        return spice.body_geometry(target, site, epoch, abcorr=self._abcorr).elevation_deg


@dataclass(frozen=True)
class GroundStation:
    """An Earth antenna: a body-fixed :class:`~astro_mine.spice.Site` plus an elevation mask.

    ``min_elevation_deg`` is the lowest elevation at which the station tracks (a DSN antenna
    masks the local horizon, terrain, and near-horizon noise — ~10° is a conservative default).
    """

    name: str
    site: spice.Site
    min_elevation_deg: float = 10.0

    @classmethod
    def from_latlon(
        cls,
        name: str,
        lat_deg: float,
        lon_deg: float,
        *,
        min_elevation_deg: float = 10.0,
        radius_m: float = EARTH_RADIUS_M,
        frame: ReferenceFrame = EARTH_BODY_FIXED,
    ) -> GroundStation:
        """A station from planetocentric latitude/longitude on a spherical Earth."""
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        position_m = (
            radius_m * math.cos(lat) * math.cos(lon),
            radius_m * math.cos(lat) * math.sin(lon),
            radius_m * math.sin(lat),
        )
        site = spice.Site(body=EARTH, position_m=position_m, frame=frame)
        return cls(name=name, site=site, min_elevation_deg=min_elevation_deg)


def dsn_contact_windows(
    station: GroundStation,
    target: str,
    window: EpochWindow,
    step_s: float,
    *,
    topocentric: TopocentricProvider,
    refine_s: float | None = None,
) -> list[ContactWindow]:
    """Contact windows for ``station -> target`` over ``window``, sampled every ``step_s``.

    ``target`` is the NAIF body the station tracks — the relay orbiter's SPK id, or ``"MOON"``
    for a direct lunar look. A contact is open while the target's topocentric elevation is at
    or above ``station.min_elevation_deg``. Provider errors (missing kernel, no coverage)
    propagate; visibility is never assumed.
    """

    def visible_at(epoch: Epoch) -> bool:
        return topocentric.elevation_deg(target, station.site, epoch) >= station.min_elevation_deg

    return search_windows((station.name, target), visible_at, window, step_s, refine_s=refine_s)
