# SPDX-License-Identifier: Apache-2.0
"""Per-cell topocentric horizon maps — pure NumPy (RM-P1-WORLDS-12).

The rigorous fidelity upgrade to the RM-P0-WORLDS-03 grid-azimuth horizon. Instead of
computing the skyline in the projected **grid** azimuth and reconciling it with the SPICE
body-fixed Sun azimuth through a south-polar-stereographic **grid-convergence** correction
(``topocentric_to_world_azimuth``, exact only for the spherical ``lon_0=0`` polar case),
this module works directly in each cell's **local topocentric frame**: it lifts every cell to
its 3-D body-fixed position and computes the true elevation/azimuth of every neighbouring
cell as seen from it. The horizon is therefore binned by *topocentric* azimuth, so a
consumer indexes it with the SPICE topocentric Sun azimuth **directly** — no projection
correction — and the result generalizes to any CRS or body (the Mars pack, RM-P1-WORLDS-11).

Engine-neutral array kernels (no rasterio, no SPICE), mirroring ``_horizon.py``: the
projected→geographic step that produces the ``(lon, lat)`` grid lives in
:class:`~astro_mine.worlds.illumination.IlluminationModel`; here everything is pure geometry
over body-fixed Cartesian metres. Planetary curvature is *intrinsic* to the 3-D geometry (a
distant cell sits below the local horizontal on its own), so there is no separate ``d**2/2R``
approximation to carry — that flat-skyline term was itself an artefact of the 2.5-D grid
model this replaces.

**Azimuth convention:** clockwise from local topocentric north, in ``[0, 360)`` — identical
to :func:`~astro_mine.worlds.provider._geometry.topocentric_elevation_azimuth` and the SPICE
``sun_geometry`` azimuth, so the two compose without a frame change.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astro_mine.worlds._hashing import canonical_meta_bytes
from astro_mine.worlds.illumination._horizon import FLAT_HORIZON_DEG

__all__ = [
    "body_fixed_positions",
    "horizon_frame_delta",
    "topocentric_elevation_azimuth_grid",
    "topocentric_horizon_field",
    "topocentric_horizon_hash",
]

F32 = np.float32
F64 = np.float64


def body_fixed_positions(
    longitude_deg: NDArray[np.float64],
    latitude_deg: NDArray[np.float64],
    elevation_m: NDArray[np.float64],
    body_radius_m: float,
) -> NDArray[np.float64]:
    """Body-fixed Cartesian positions (m) for a grid of geographic cells.

    Places each cell on the body sphere at ``body_radius_m + elevation_m`` and its
    ``(longitude, latitude)`` — the spherical datum the CRS's ``+R`` reference radius
    defines. Returns an ``(H, W, 3)`` array in the body-fixed frame the topocentric
    geometry is evaluated in. Voids (NaN elevation) propagate as NaN positions and are
    skipped downstream.
    """
    if not (longitude_deg.shape == latitude_deg.shape == elevation_m.shape):
        raise ValueError(
            "longitude, latitude, and elevation grids must share a shape; got "
            f"{longitude_deg.shape}, {latitude_deg.shape}, {elevation_m.shape}"
        )
    lon = np.radians(longitude_deg)
    lat = np.radians(latitude_deg)
    r = body_radius_m + elevation_m
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.stack((x, y, z), axis=-1).astype(F64)


def topocentric_elevation_azimuth_grid(
    observer: NDArray[np.float64], target: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Vectorised topocentric elevation/azimuth of ``target`` seen from ``observer``.

    Both are ``(H, W, 3)`` body-fixed Cartesian grids. Returns ``(elevation_deg,
    azimuth_deg)`` per cell — elevation above the cell's local horizontal (from its radial
    ``up``), azimuth clockwise from local north in ``[0, 360)`` — the exact vectorised
    analogue of :func:`~astro_mine.worlds.provider._geometry.topocentric_elevation_azimuth`.
    Cells where the target coincides with the observer, or the observer is on the spin axis
    (a pole, where azimuth is undefined), yield NaN so the caller skips them.
    """
    delta = target - observer
    rng = np.linalg.norm(delta, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        direction = delta / rng[..., None]
        up = observer / np.linalg.norm(observer, axis=-1, keepdims=True)
        elevation = np.degrees(np.arcsin(np.clip(np.sum(direction * up, axis=-1), -1.0, 1.0)))
        # east = k x up (k = spin axis); north = up x east. |east| -> 0 on the spin axis.
        east = np.stack((-up[..., 1], up[..., 0], np.zeros_like(up[..., 0])), axis=-1)
        east_norm = np.linalg.norm(east, axis=-1)
        east = east / east_norm[..., None]
        north = np.cross(up, east)
        azimuth = (
            np.degrees(
                np.arctan2(np.sum(direction * east, axis=-1), np.sum(direction * north, axis=-1))
            )
            % 360.0
        )
    invalid = (rng == 0.0) | (east_norm < 1e-9) | np.isnan(rng)
    elevation = np.where(invalid, np.nan, elevation)
    azimuth = np.where(invalid, np.nan, azimuth)
    return elevation, azimuth


def _shift(arr: NDArray[np.float64], drow: int, dcol: int) -> NDArray[np.float64]:
    """``out[i, j] = arr[i + drow, j + dcol, :]`` with off-grid cells NaN (``(H, W, 3)``)."""
    height, width = arr.shape[:2]
    out = np.full_like(arr, np.nan)
    i0, i1 = max(0, -drow), min(height, height - drow)
    j0, j1 = max(0, -dcol), min(width, width - dcol)
    if i1 > i0 and j1 > j0:
        out[i0:i1, j0:j1] = arr[i0 + drow : i1 + drow, j0 + dcol : j1 + dcol]
    return out


def topocentric_horizon_field(
    positions: NDArray[np.float64],
    *,
    pixel_size_m: tuple[float, float],
    n_azimuth: int,
    max_radius_m: float,
) -> NDArray[np.float32]:
    """Per-cell, per-topocentric-azimuth terrain-skyline elevation angle (degrees).

    For every neighbouring cell within ``max_radius_m`` (enumerated as integer pixel
    offsets from the signed pixel sizes ``pixel_size_m = (a, e)``), compute the true
    topocentric elevation/azimuth of that neighbour from each cell and keep, per cell and
    per azimuth bin, the running maximum elevation angle. Cells with no terrain rising above
    their local horizontal keep :data:`~astro_mine.worlds.illumination._horizon.FLAT_HORIZON_DEG`.

    Curvature is intrinsic (a far neighbour sits below the local horizontal by construction),
    so — unlike the grid ``horizon_field`` — there is no separate curvature-drop term and no
    projection-specific azimuth correction. Returns a ``float32`` ``(H, W, n_azimuth)`` array.
    Cost is ``O(pixels-within-radius)`` vectorised passes — fine on the synthetic test grid;
    a documented precompute on the real DEM (GPU on-demand is RM-P1-WORLDS-10).
    """
    if positions.ndim != 3 or positions.shape[2] != 3:
        raise ValueError(f"positions must be (H, W, 3), got shape {positions.shape}")
    if n_azimuth <= 0:
        raise ValueError(f"n_azimuth must be positive, got {n_azimuth}")
    a, e = pixel_size_m
    if a == 0.0 or e == 0.0:
        raise ValueError(f"pixel_size_m components must be non-zero, got {pixel_size_m}")
    base = min(abs(a), abs(e))
    if max_radius_m < base:
        raise ValueError(f"max_radius_m ({max_radius_m}) is smaller than one pixel ({base})")

    height, width = positions.shape[:2]
    n_row = int(max_radius_m / abs(e))
    n_col = int(max_radius_m / abs(a))
    horizon = np.full((height, width, n_azimuth), FLAT_HORIZON_DEG, dtype=F32)
    flat = horizon.reshape(height * width, n_azimuth)
    cell_index = np.arange(height * width)
    width_deg = 360.0 / n_azimuth

    for drow in range(-n_row, n_row + 1):
        for dcol in range(-n_col, n_col + 1):
            if drow == 0 and dcol == 0:
                continue
            if np.hypot(drow * e, dcol * a) > max_radius_m:
                continue
            neighbour = _shift(positions, drow, dcol)
            elevation, azimuth = topocentric_elevation_azimuth_grid(positions, neighbour)
            valid = ~np.isnan(elevation) & ~np.isnan(azimuth)
            bins = np.where(valid, np.floor((azimuth % 360.0) / width_deg), 0.0)
            bins = np.clip(bins, 0, n_azimuth - 1).astype(np.intp)
            contribution = np.where(valid, elevation, -np.inf).astype(F32)
            np.maximum.at(flat, (cell_index, bins.ravel()), contribution.ravel())
    return horizon


def topocentric_horizon_hash(horizon: NDArray[np.float32], meta: dict[str, Any]) -> str:
    """A deterministic ``sha256:`` digest over the topocentric horizon + canonical metadata.

    ``meta``'s toolchain is provenance and is not covered — see :mod:`astro_mine.worlds._hashing`.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(horizon).tobytes())
    h.update(canonical_meta_bytes(meta))
    return f"sha256:{h.hexdigest()}"


def horizon_frame_delta(
    grid_horizon: NDArray[np.float32],
    topocentric_horizon: NDArray[np.float32],
    latitude_deg: NDArray[np.float64],
) -> dict[str, float]:
    """Quantify the P0→P1 horizon-frame delta as an explicit error budget.

    The two maps bin the same skyline by different azimuth frames (grid vs topocentric), so
    the physically meaningful discrepancy is the **maximum-over-azimuth** horizon-elevation
    difference at each cell — how differently the two frames would gate a low Sun. Returns
    ``max_abs_deg``/``mean_abs_deg`` over all cells and ``max_abs_deg_high_lat`` restricted to
    the cells furthest from the pole (top decile of ``|latitude|`` distance-from-±90°), where
    the grid-convergence approximation is worst. Both arrays must share a shape.
    """
    if grid_horizon.shape != topocentric_horizon.shape:
        raise ValueError(
            f"horizon shapes differ: {grid_horizon.shape} vs {topocentric_horizon.shape}"
        )
    per_cell = np.abs(
        grid_horizon.max(axis=2).astype(F64) - topocentric_horizon.max(axis=2).astype(F64)
    )
    from_pole = 90.0 - np.abs(latitude_deg)
    if from_pole.size:
        threshold = float(np.quantile(from_pole, 0.9))
        high_lat = per_cell[from_pole >= threshold]
    else:  # pragma: no cover - a grid always has cells
        high_lat = per_cell
    return {
        "max_abs_deg": float(per_cell.max()),
        "mean_abs_deg": float(per_cell.mean()),
        "max_abs_deg_high_lat": float(high_lat.max()) if high_lat.size else 0.0,
    }
