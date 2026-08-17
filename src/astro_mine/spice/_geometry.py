# SPDX-License-Identifier: Apache-2.0
"""SPICE-backed frames, epochs, and body/topocentric geometry (RFC-0002).

The shared SPICE foundation. It *resolves* Core's units/frames vocabulary via SPICE — the
name->geometry step Core defers (units docstrings; core.md §2.3) — for every consumer:
Worlds illumination/PSR, Link LOS/contact windows, and (later) Sim's orbital engine and
Transit:

- epoch helpers — a Core :class:`Epoch`'s ``tdb_seconds`` is SPICE ET directly;
- raw primitives — :func:`body_position` (``spkpos``) and :func:`frame_transform`
  (``pxform``), the building blocks a consumer (e.g. Link's ``gfposc`` window search)
  drives over the same furnished kernel pool;
- topocentric geometry — :func:`body_geometry` gives a target's elevation/azimuth/range
  at a :class:`Site`, the shared scalar consumers threshold against terrain horizons
  (the horizon maps and window search themselves live in the consumers, not here).

All positions are SI metres (SPICE works in km; converted at the boundary); angles are
degrees. Frames are Core :class:`ReferenceFrame`s, resolved by name.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, NoReturn

import numpy as np
import spiceypy as sp
from numpy.typing import NDArray
from spiceypy.utils.exceptions import SpiceyError

from astro_mine.core.units import (
    EARTH,
    INERTIAL_J2000,
    MOON,
    MOON_BODY_FIXED,
    SUN,
    Epoch,
    EpochWindow,
    ReferenceFrame,
    TimeScale,
)
from astro_mine.spice._kernels import SpiceKernelError

__all__ = [
    "DEFAULT_ABCORR",
    "EARTH_RADIUS_M",
    "MOON_RADIUS_M",
    "BodyGeometry",
    "Site",
    "SpiceGeometryError",
    "body_geometry",
    "body_position",
    "earth_geometry",
    "epoch_from_utc",
    "epoch_range",
    "et",
    "frame_transform",
    "sun_geometry",
]

#: Apparent-direction correction (light-time + stellar aberration) — the right default
#: for "where do I see the Sun/Earth". Tests against minimal synthetic kernels use
#: ``"NONE"`` (geometric), which needs only the direct target->observer SPK segment.
DEFAULT_ABCORR = "LT+S"

#: The Moon modelled as a sphere (PROJ ``+R``); the reference radius for lunar sites.
#: Lives here (geometry) per RFC-0002; ``worlds.crs`` re-imports it for its CRS datum.
MOON_RADIUS_M = 1_737_400.0

#: Earth modelled as a sphere — the mean volumetric radius (IUGG). Lives here (geometry,
#: not CRS) per RFC-0002, mirroring :data:`MOON_RADIUS_M`; ``Link`` re-imports it to place
#: Earth ground stations in a body-fixed frame from lat/lon rather than defining its own.
EARTH_RADIUS_M = 6_371_000.0

_M_PER_KM = 1000.0


class SpiceGeometryError(Exception):
    """Raised on an invalid epoch, frame, or degenerate geometry query."""


#: SPICE short-error fragments that mean a *kernel/coverage* fault (empty pool, an epoch
#: outside SPK/PCK coverage, a missing leapsecond kernel) rather than a bad body/frame
#: identifier. Used to route a raw SpiceyPy error onto the right typed boundary error.
_KERNEL_ERROR_MARKERS = ("INSUFFDATA", "NOLOADEDFILES", "NOLEAPSECONDS", "KERNELPOOL", "NOKERNEL")


def _raise_boundary_error(exc: SpiceyError, message: str) -> NoReturn:
    """Re-raise a raw SpiceyPy error as the typed, offender-naming boundary error.

    A coverage/pool fault (e.g. ``SPICE(SPKINSUFFDATA)`` for an epoch outside SPK
    coverage, or a missing leapsecond kernel) becomes :class:`SpiceKernelError`; an
    unresolvable body or frame becomes :class:`SpiceGeometryError`. Either way the caller
    catches a *named* boundary error — never a raw ``spiceypy`` exception (``spice.md``
    §10) — and never a silent default (``degrade, don't lie``; §2.5). The SPICE short
    message is appended so the underlying cause survives.
    """
    marker = f"{type(exc).__name__} {exc}".upper()
    detail = f"{message}: {exc}"
    if any(m in marker for m in _KERNEL_ERROR_MARKERS):
        raise SpiceKernelError(detail) from exc
    raise SpiceGeometryError(detail) from exc


@dataclass(frozen=True)
class Site:
    """A surface site: a body-fixed position realizing Core's ``FrameClass.TOPOCENTRIC``.

    The shared site type consumers (Worlds, Link) build topocentric geometry against.
    ``position_m`` is in ``frame`` (a body-fixed frame), relative to the body centre.
    """

    body: str
    position_m: tuple[float, float, float]
    frame: ReferenceFrame

    @classmethod
    def lunar_from_latlon(
        cls,
        lat_deg: float,
        lon_deg: float,
        *,
        radius_m: float = MOON_RADIUS_M,
        frame: ReferenceFrame = MOON_BODY_FIXED,
    ) -> Site:
        """A lunar surface site from planetocentric latitude/longitude on the Moon sphere."""
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        x = radius_m * math.cos(lat) * math.cos(lon)
        y = radius_m * math.cos(lat) * math.sin(lon)
        z = radius_m * math.sin(lat)
        return cls(body=MOON, position_m=(x, y, z), frame=frame)


@dataclass(frozen=True)
class BodyGeometry:
    """A target body's geometry as seen from a :class:`Site` at an epoch."""

    target: str
    position_m: tuple[float, float, float]  # vector site -> target, in the site frame
    direction: tuple[float, float, float]  # unit vector site -> target
    range_m: float
    elevation_deg: float  # above the local horizontal plane
    azimuth_deg: float  # clockwise from local north, [0, 360)


def et(epoch: Epoch) -> float:
    """SPICE ephemeris time (seconds) for a Core :class:`Epoch`.

    A Core ``Epoch``'s ``tdb_seconds`` is SPICE ET directly (``TimeScale`` admits only
    the SPICE ephemeris scales TDB/ET — civil scales are unrepresentable at the waist),
    so this is the named conversion seam, not a unit change.
    """
    return epoch.tdb_seconds


def epoch_from_utc(utc: str) -> Epoch:
    """Parse a UTC/calendar string into a TDB :class:`Epoch` (needs a leapsecond kernel loaded)."""
    try:
        tdb_seconds = float(sp.str2et(utc))
    except SpiceyError as exc:
        _raise_boundary_error(exc, f"cannot parse UTC epoch {utc!r}")
    return Epoch(tdb_seconds=tdb_seconds, scale=TimeScale.TDB)


def epoch_range(window: EpochWindow, step_s: float) -> Iterator[Epoch]:
    """Yield epochs across the half-open ``[start, end)`` window at ``step_s`` spacing."""
    if step_s <= 0.0:
        raise SpiceGeometryError(f"step_s must be positive, got {step_s}")
    t = window.start.tdb_seconds
    end = window.end.tdb_seconds
    while t < end:
        yield Epoch(tdb_seconds=t, scale=TimeScale.TDB)
        t += step_s


def body_position(
    target: str,
    observer: str,
    epoch: Epoch,
    *,
    frame: ReferenceFrame = INERTIAL_J2000,
    abcorr: str = DEFAULT_ABCORR,
) -> tuple[float, float, float]:
    """Position (SI metres) of ``target`` relative to ``observer`` in ``frame`` (``spkpos``)."""
    try:
        pos_km, _light_time = sp.spkpos(target, et(epoch), frame.name, abcorr, observer)
    except SpiceyError as exc:
        _raise_boundary_error(
            exc,
            f"cannot resolve position of {target!r} relative to {observer!r} "
            f"in frame {frame.name!r} at ET {et(epoch)}",
        )
    return (
        float(pos_km[0]) * _M_PER_KM,
        float(pos_km[1]) * _M_PER_KM,
        float(pos_km[2]) * _M_PER_KM,
    )


def frame_transform(
    from_frame: ReferenceFrame, to_frame: ReferenceFrame, epoch: Epoch
) -> NDArray[np.float64]:
    """The 3x3 rotation from ``from_frame`` to ``to_frame`` at ``epoch`` (``pxform``)."""
    try:
        matrix: Any = sp.pxform(from_frame.name, to_frame.name, et(epoch))
    except SpiceyError as exc:
        _raise_boundary_error(
            exc,
            f"cannot resolve rotation from {from_frame.name!r} to {to_frame.name!r} "
            f"at ET {et(epoch)}",
        )
    return np.asarray(matrix, dtype=np.float64)


def body_geometry(
    target: str, site: Site, epoch: Epoch, *, abcorr: str = DEFAULT_ABCORR
) -> BodyGeometry:
    """Topocentric geometry of ``target`` from ``site`` at ``epoch``.

    Returns the site->target vector, unit direction, range, and the elevation above the
    local horizontal plane and azimuth clockwise from local north — all in the site's
    body-fixed frame. Elevation/azimuth are *geometric* (spherical local vertical); the
    comparison against the terrain horizon is the consumer's job.
    """
    target_pos = np.array(body_position(target, site.body, epoch, frame=site.frame, abcorr=abcorr))
    site_pos = np.array(site.position_m, dtype=np.float64)
    delta = target_pos - site_pos
    range_m = float(np.linalg.norm(delta))
    if range_m == 0.0:  # pragma: no cover - a target at the site is not physically reachable
        raise SpiceGeometryError("target coincides with the site; geometry is undefined")
    direction = delta / range_m

    up = site_pos / np.linalg.norm(site_pos)
    elevation = math.degrees(math.asin(float(np.clip(np.dot(direction, up), -1.0, 1.0))))

    pole = np.array([0.0, 0.0, 1.0])
    east = np.cross(pole, up)
    east_norm = float(np.linalg.norm(east))
    if east_norm < 1e-9:  # the site is on the spin axis (a pole) — azimuth is undefined
        azimuth = 0.0
    else:
        east = east / east_norm
        north = np.cross(up, east)
        bearing = math.atan2(float(np.dot(direction, east)), float(np.dot(direction, north)))
        azimuth = math.degrees(bearing) % 360.0

    return BodyGeometry(
        target=target,
        position_m=(float(delta[0]), float(delta[1]), float(delta[2])),
        direction=(float(direction[0]), float(direction[1]), float(direction[2])),
        range_m=range_m,
        elevation_deg=elevation,
        azimuth_deg=azimuth,
    )


def sun_geometry(site: Site, epoch: Epoch, *, abcorr: str = DEFAULT_ABCORR) -> BodyGeometry:
    """Sun geometry at a site — the input illumination/PSR thresholds against the horizon."""
    return body_geometry(SUN, site, epoch, abcorr=abcorr)


def earth_geometry(site: Site, epoch: Epoch, *, abcorr: str = DEFAULT_ABCORR) -> BodyGeometry:
    """Earth geometry at a site — the input Link thresholds for Earth-link LOS."""
    return body_geometry(EARTH, site, epoch, abcorr=abcorr)
