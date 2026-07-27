"""Fine on-demand ray-cast illumination — the reference + portable CPU fallback (RM-P1-WORLDS-10).

The precomputed horizon map (RM-P0-WORLDS-03) bins the terrain skyline into ``n_azimuth`` azimuth
bins so per-epoch Sun visibility is an O(1) lookup; that binning is its approximation. This module
the **fine on-demand path** worlds.md §8/§11 recommends: per cell and epoch, march the DEM along the
Sun's *exact* (unbinned) world azimuth and keep the running skyline elevation, then light the cell
iff the Sun clears that skyline — the ray-cast reference the GPU backend (``_raycast_gpu``) and the
learned surrogate (``_surrogate``) are measured against, and the CPU fallback that "MUST work"
without a GPU (worlds.md §7, deployment tier 1).

The math is an **engine-neutral array kernel** parameterised on an array module ``xp`` (NumPy here,
CuPy on the device), mirroring ``_horizon.py`` — planetary curvature carried explicitly as a
``d**2 / 2R`` skyline drop, off-grid neighbours contributing nothing (conventions.md §1.6, §5). The
kernel is the single source of truth: ``_raycast_gpu`` re-dispatches *this same* function to CuPy so
the GPU path cannot silently diverge from the CPU reference.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import rasterio.transform
from numpy.typing import NDArray

from astro_mine.core.units import Epoch
from astro_mine.worlds.illumination import IlluminationError, IlluminationModel
from astro_mine.worlds.illumination._horizon import FLAT_HORIZON_DEG

__all__ = [
    "RAYCAST_CPU_BACKEND",
    "RayCastIlluminationModel",
    "raycast_cell_lit",
    "raycast_lit_mask",
    "raycast_skyline_deg",
]

#: The backend selector for the CPU ray-cast field model.
RAYCAST_CPU_BACKEND = "raycast_cpu"


def _march_distances(base: float, max_radius_m: float, n_radius_steps: int) -> list[float]:
    """The ray-march sample distances (m) — identical spacing to ``_horizon.horizon_field``.

    Sharing the sampling means the fine ray-cast and the binned horizon map differ only by the
    azimuth quantisation, never by a different march, so their agreement on unambiguous cells is a
    meaningful check (the RM-P1-WORLDS-10 acceptance tolerance).
    """
    return [float(d) for d in np.linspace(base, max_radius_m, n_radius_steps)]


def _resolve_steps(
    pixel_size_m: tuple[float, float], max_radius_m: float, n_radius_steps: int | None
) -> tuple[float, float, int]:
    a, e = pixel_size_m
    if a == 0.0 or e == 0.0:
        raise ValueError(f"pixel_size_m components must be non-zero, got {pixel_size_m}")
    base = min(abs(a), abs(e))
    if max_radius_m < base:
        raise ValueError(f"max_radius_m ({max_radius_m}) is smaller than one pixel ({base})")
    if n_radius_steps is None:
        n_radius_steps = max(1, round(max_radius_m / base))
    return a, e, n_radius_steps


def _shift(arr: Any, drow: int, dcol: int, xp: Any) -> Any:
    """``out[i, j] = arr[i + drow, j + dcol]`` with off-grid cells NaN (array-module agnostic)."""
    height, width = arr.shape
    out = xp.full((height, width), xp.nan, dtype=arr.dtype)
    i0, i1 = max(0, -drow), min(height, height - drow)
    j0, j1 = max(0, -dcol), min(width, width - dcol)
    if i1 > i0 and j1 > j0:
        out[i0:i1, j0:j1] = arr[i0 + drow : i1 + drow, j0 + dcol : j1 + dcol]
    return out


def raycast_skyline_deg(
    elevation: NDArray[np.float64],
    *,
    pixel_size_m: tuple[float, float],
    azimuth_world_deg: float,
    max_radius_m: float,
    body_radius_m: float,
    n_radius_steps: int | None = None,
    xp: Any = np,
) -> Any:
    """Per-cell terrain-skyline elevation (deg) along one *exact* world azimuth (an ``xp`` array).

    Marches the whole grid outward in the continuous ``azimuth_world_deg`` direction — no bin
    snapping — keeping the running maximum curvature-corrected elevation angle to the terrain there.
    ``xp`` is the array module (``numpy`` or ``cupy``); ``elevation`` must already live on it. Cells
    with no terrain rising above their local horizontal keep :data:`FLAT_HORIZON_DEG`.
    """
    a, e, steps = _resolve_steps(pixel_size_m, max_radius_m, n_radius_steps)
    base = min(abs(a), abs(e))
    rad = math.radians(azimuth_world_deg % 360.0)
    wx, wy = math.sin(rad), math.cos(rad)  # world unit step, clockwise from +y/north
    running = xp.full(elevation.shape, FLAT_HORIZON_DEG, dtype=np.float64)
    for d in _march_distances(base, max_radius_m, steps):
        dcol = round((wx * d) / a)
        drow = round((wy * d) / e)
        if dcol == 0 and drow == 0:
            continue
        drop = (d * d) / (2.0 * body_radius_m)
        rise = _shift(elevation, drow, dcol, xp) - elevation - drop
        running = xp.fmax(running, xp.degrees(xp.arctan2(rise, d)))  # NaN (off-grid) skipped
    return running


def raycast_lit_mask(
    elevation: NDArray[np.float64],
    *,
    pixel_size_m: tuple[float, float],
    sun_elevation_deg: float,
    sun_azimuth_world_deg: float,
    max_radius_m: float,
    body_radius_m: float,
    n_radius_steps: int | None = None,
    xp: Any = np,
) -> Any:
    """Cells where the Sun clears the *exact*-azimuth ray-cast skyline (a boolean ``xp`` array)."""
    skyline = raycast_skyline_deg(
        elevation,
        pixel_size_m=pixel_size_m,
        azimuth_world_deg=sun_azimuth_world_deg,
        max_radius_m=max_radius_m,
        body_radius_m=body_radius_m,
        n_radius_steps=n_radius_steps,
        xp=xp,
    )
    return sun_elevation_deg > skyline


def raycast_cell_lit(
    elevation: NDArray[np.float64],
    row: int,
    col: int,
    *,
    pixel_size_m: tuple[float, float],
    sun_elevation_deg: float,
    sun_azimuth_world_deg: float,
    max_radius_m: float,
    body_radius_m: float,
    n_radius_steps: int | None = None,
) -> bool:
    """Fine single-cell Sun visibility — march only cell ``(row, col)`` toward the true Sun.

    The on-demand per-query primitive: an O(march) skyline in the Sun's exact azimuth for one cell,
    so a swarm point-query costs one ray rather than a full-grid pass.
    """
    a, e, steps = _resolve_steps(pixel_size_m, max_radius_m, n_radius_steps)
    base = min(abs(a), abs(e))
    height, width = elevation.shape
    rad = math.radians(sun_azimuth_world_deg % 360.0)
    wx, wy = math.sin(rad), math.cos(rad)
    e0 = float(elevation[row, col])
    skyline = float(FLAT_HORIZON_DEG)
    for d in _march_distances(base, max_radius_m, steps):
        dcol = round((wx * d) / a)
        drow = round((wy * d) / e)
        if dcol == 0 and drow == 0:
            continue
        r, c = row + drow, col + dcol
        if 0 <= r < height and 0 <= c < width:
            rise = float(elevation[r, c]) - e0 - (d * d) / (2.0 * body_radius_m)
            skyline = max(skyline, math.degrees(math.atan2(rise, d)))
    return bool(sun_elevation_deg > skyline)


class RayCastIlluminationModel(IlluminationModel):
    """Horizon-map illumination with the Sun-visibility path served by fine ray casting.

    Subclasses :class:`~astro_mine.worlds.illumination.IlluminationModel`, so it builds (and keeps)
    the per-azimuth horizon map for the Link line-of-sight product and inherits the full public API;
    only :meth:`illumination_at` and :meth:`illuminated_mask` are overridden to ray-cast the DEM
    against the Sun's exact azimuth (``sun_visible`` / ``psr_mask`` inherit and route through them).
    The active backend is folded into :attr:`illumination_hash`, so selecting it honestly moves the
    hash (RM-P1-WORLDS-15). ``xp``/:meth:`_to_device`/:meth:`_to_host` are the device seam the GPU
    subclass overrides; here they are the identity NumPy path.
    """

    def __init__(self, terrain: Any, *, backend: str = RAYCAST_CPU_BACKEND, **kwargs: Any) -> None:
        super().__init__(terrain, backend=backend, **kwargs)

    # --- device seam (overridden by the CuPy subclass) -----------------------------

    def _xp(self) -> Any:
        """The array module the mask kernel runs on (NumPy on the CPU path)."""
        return np

    def _to_device(self, array: NDArray[np.float64]) -> Any:
        """Move a host array onto the compute device (identity on the CPU path)."""
        return array

    def _to_host(self, mask: Any) -> NDArray[np.bool_]:
        """Return a device mask to the host as a NumPy bool array (identity on the CPU path)."""
        return np.asarray(mask, dtype=np.bool_)

    # --- fine Sun-visibility overrides ---------------------------------------------

    def illumination_at(self, x: float, y: float, epoch: Epoch) -> tuple[bool, float]:
        """``(sun_visible, sun_elevation_deg)`` — the Sun cleared via a single-cell ray march."""
        row, col = rasterio.transform.rowcol(self.transform, x, y)
        row, col = int(row), int(col)
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise IlluminationError(f"({x}, {y}) is outside the terrain grid")
        elevation_deg, world_az = self._sun(x, y, epoch)
        lit = raycast_cell_lit(
            self._filled_elevation,
            row,
            col,
            pixel_size_m=(float(self.transform.a), float(self.transform.e)),
            sun_elevation_deg=elevation_deg,
            sun_azimuth_world_deg=world_az,
            max_radius_m=self.max_radius_m,
            body_radius_m=self.body_radius_m,
        )
        return lit, elevation_deg

    def illuminated_mask(self, epoch: Epoch) -> NDArray[np.bool_]:
        """Fine lit raster: the region-centre Sun ray-cast at its exact (unbinned) azimuth.

        The dominant swarm-scale cost worlds.md §8 flags — dispatched to the device by the GPU
        subclass via :meth:`_to_device`/:meth:`_xp`, run on the same kernel on the CPU here.
        """
        cx, cy = rasterio.transform.xy(self.transform, self.height // 2, self.width // 2)
        elevation_deg, world_az = self._sun(float(cx), float(cy), epoch)
        mask = raycast_lit_mask(
            self._to_device(self._filled_elevation),
            pixel_size_m=(float(self.transform.a), float(self.transform.e)),
            sun_elevation_deg=elevation_deg,
            sun_azimuth_world_deg=world_az,
            max_radius_m=self.max_radius_m,
            body_radius_m=self.body_radius_m,
            xp=self._xp(),
        )
        return self._to_host(mask)
