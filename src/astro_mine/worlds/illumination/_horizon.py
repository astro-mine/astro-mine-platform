"""Per-azimuth horizon maps + sun-visibility kernels — pure NumPy (RM-P0-WORLDS-03).

The comms/sun-denied core of the anchor scenario. A *horizon map* gives, for each surface
cell and each azimuth bin, the maximum terrain-skyline elevation angle; the Sun is visible
at a cell iff its (SPICE-derived) elevation exceeds the horizon angle in the Sun's azimuth.
Precomputing the map turns per-epoch visibility into an O(1) lookup.

These are engine-neutral array kernels (no rasterio, no SPICE) so they stay cheap to
unit-test and fully type-checked, mirroring ``terrain/_layers.py``. Planetary curvature is
carried explicitly (a ``d**2 / 2R`` skyline drop) and off-grid directions keep the flat
0-degree horizon — honest, flagged approximations rather than silent ones
(conventions.md §1.6, §5).

**Azimuth convention:** clockwise from world ``+y`` / north, matching ``terrain`` aspect.
The SPICE Sun azimuth is *topocentric* (clockwise from body-fixed local north); a consumer
converts it into this world frame with :func:`topocentric_to_world_azimuth`. The rigorous
per-cell topocentric horizon that removes that conversion is RM-P1-WORLDS-12 (issue #11).
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astro_mine.worlds._hashing import canonical_meta_bytes

__all__ = [
    "FLAT_HORIZON_DEG",
    "azimuth_bin",
    "curvature_drop_m",
    "horizon_field",
    "horizon_hash",
    "sun_visibility_raster",
    "topocentric_to_world_azimuth",
]

F32 = np.float32

#: A cell with no terrain rising above its local horizontal sees the flat-ground skyline
#: at 0 degrees; only terrain *above* the cell raises the horizon and can block the Sun.
FLAT_HORIZON_DEG = 0.0


def azimuth_bin(azimuth_deg: float, n_azimuth: int) -> int:
    """Bin an azimuth into ``[0, n_azimuth)`` (clockwise from world north).

    Bin ``k`` spans ``[k * 360/n, (k+1) * 360/n)`` degrees; the value is wrapped into
    ``[0, 360)`` first, so any real azimuth resolves.
    """
    if n_azimuth <= 0:
        raise ValueError(f"n_azimuth must be positive, got {n_azimuth}")
    width = 360.0 / n_azimuth
    return int(np.floor((azimuth_deg % 360.0) / width)) % n_azimuth


def curvature_drop_m(distance_m: NDArray[np.float64] | float, body_radius_m: float) -> Any:
    """Skyline drop from planetary curvature at a ground distance: ``d**2 / (2 R)`` (m).

    Terrain a distance ``d`` away sits this much lower relative to the local horizontal
    plane than a flat planet would put it — subtracted from the apparent rise so far
    ridges do not over-block the Sun (a sim-to-real honesty term, conventions.md §1.6).
    """
    return np.square(distance_m) / (2.0 * body_radius_m)


def _shift(arr: NDArray[np.float64], drow: int, dcol: int) -> NDArray[np.float64]:
    """Return ``out`` with ``out[i, j] = arr[i + drow, j + dcol]``; off-grid cells are NaN."""
    height, width = arr.shape
    out = np.full((height, width), np.nan, dtype=np.float64)
    i0, i1 = max(0, -drow), min(height, height - drow)
    j0, j1 = max(0, -dcol), min(width, width - dcol)
    if i1 > i0 and j1 > j0:
        out[i0:i1, j0:j1] = arr[i0 + drow : i1 + drow, j0 + dcol : j1 + dcol]
    return out


def horizon_field(
    elevation: NDArray[np.float64],
    *,
    pixel_size_m: tuple[float, float],
    n_azimuth: int,
    max_radius_m: float,
    body_radius_m: float,
    n_radius_steps: int | None = None,
) -> NDArray[np.float32]:
    """Per-cell, per-azimuth terrain-skyline elevation angle (degrees).

    For every azimuth bin, march the whole grid outward in that world direction and keep
    the running maximum elevation angle to the terrain there, curvature-corrected. The
    march steps in *world* directions via the signed pixel sizes ``pixel_size_m =
    (a, e)`` (``a`` the x/column spacing, ``e`` the y/row spacing — negative for a
    north-up raster), so the result is in the world-azimuth frame regardless of raster
    orientation. Off-grid neighbours contribute nothing, leaving :data:`FLAT_HORIZON_DEG`.

    Returns a ``float32`` array of shape ``(height, width, n_azimuth)``. Cost is
    ``O(n_azimuth * n_radius_steps)`` vectorised array ops — fine at the synthetic test
    grid; a documented precompute on the real DEM (GPU on-demand is P1, RM-P1-WORLDS-10).
    """
    elev = np.asarray(elevation, dtype=np.float64)
    if elev.ndim != 2:
        raise ValueError(f"elevation must be 2-D, got shape {elev.shape}")
    a, e = pixel_size_m
    if a == 0.0 or e == 0.0:
        raise ValueError(f"pixel_size_m components must be non-zero, got {pixel_size_m}")
    base = min(abs(a), abs(e))
    if max_radius_m < base:
        raise ValueError(f"max_radius_m ({max_radius_m}) is smaller than one pixel ({base})")
    if n_radius_steps is None:
        n_radius_steps = max(1, round(max_radius_m / base))
    distances = np.linspace(base, max_radius_m, n_radius_steps)

    height, width = elev.shape
    horizon = np.full((height, width, n_azimuth), FLAT_HORIZON_DEG, dtype=F32)
    for k in range(n_azimuth):
        rad = np.radians(360.0 * k / n_azimuth)
        wx, wy = np.sin(rad), np.cos(rad)  # world unit step, clockwise from +y/north
        running = np.full((height, width), FLAT_HORIZON_DEG, dtype=np.float64)
        for d in distances:
            dcol = round((wx * d) / a)
            drow = round((wy * d) / e)
            if dcol == 0 and drow == 0:
                continue
            rise = _shift(elev, drow, dcol) - elev - curvature_drop_m(d, body_radius_m)
            running = np.fmax(running, np.degrees(np.arctan2(rise, d)))  # NaN (off-grid) skipped
        horizon[:, :, k] = running.astype(F32)
    return horizon


def sun_visibility_raster(
    horizon: NDArray[np.float32], sun_elevation_deg: float, sun_azimuth_world_deg: float
) -> NDArray[np.bool_]:
    """Cells where the Sun clears the terrain skyline: ``sun_el > horizon[:, :, az_bin]``."""
    n_azimuth = horizon.shape[2]
    selected = horizon[:, :, azimuth_bin(sun_azimuth_world_deg, n_azimuth)]
    return sun_elevation_deg > selected


def topocentric_to_world_azimuth(azimuth_topocentric_deg: float, longitude_deg: float) -> float:
    """Convert a body-fixed topocentric azimuth to the world (grid) azimuth.

    For the south-polar stereographic CRS (``lon_0 = 0``) the grid convergence — the angle
    from grid north to true north — equals the body-fixed **longitude**, so a direction at
    ``A`` clockwise from true north is at ``A + longitude`` clockwise from grid north. This
    is exact for the spherical polar stereographic case; the per-cell topocentric horizon
    that removes the approximation is RM-P1-WORLDS-12 (issue #11).
    """
    return (azimuth_topocentric_deg + longitude_deg) % 360.0


def horizon_hash(horizon: NDArray[np.float32], meta: dict[str, Any]) -> str:
    """A deterministic ``sha256:`` digest over the horizon array + canonical metadata.

    ``meta``'s toolchain is provenance and is not covered — see :mod:`astro_mine.worlds._hashing`.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(horizon).tobytes())
    h.update(canonical_meta_bytes(meta))
    return f"sha256:{h.hexdigest()}"


def psr_mask_hash(
    mask: NDArray[np.bool_], void_mask: NDArray[np.bool_], meta: dict[str, Any]
) -> str:
    """A deterministic ``sha256:`` digest over the PSR mask + void mask + canonical metadata.

    The ``mask`` is the SPICE-derived shadow result (cells never sunlit over the sampled window),
    so folding it into the world hash means a kernel/ephemeris change that alters the shipped PSR
    mask changes the world hash — the terrain horizon alone did not pin the PSR-ness the anchor
    scores (RM-P1-WORLDS-15)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(mask, dtype=np.uint8).tobytes())
    h.update(np.ascontiguousarray(void_mask, dtype=np.uint8).tobytes())
    h.update(canonical_meta_bytes(meta))
    return f"sha256:{h.hexdigest()}"
