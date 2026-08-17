# SPDX-License-Identifier: Apache-2.0
"""Derived terrain layers + content hashing — pure NumPy kernels (RM-P0-WORLDS-01).

Slope, aspect, roughness, and the vertical-uncertainty / void treatment, plus the
deterministic terrain hash. These are engine-neutral array kernels (no rasterio, no IO)
so they are cheap to unit-test and fully type-checked. Uncertainty is first-class: voids
are flagged and their derived values inflated, never silently interpolated away
(conventions.md §1.6, §5).
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
from numpy.typing import NDArray

from astro_mine.worlds._hashing import canonical_meta_bytes

__all__ = [
    "fill_voids",
    "normal_from_slope_aspect",
    "roughness",
    "slope_aspect",
    "terrain_hash",
    "vertical_uncertainty",
]

F32 = np.float32


def fill_voids(elev: NDArray[np.float32], void: NDArray[np.bool_]) -> NDArray[np.float32]:
    """Return a copy of ``elev`` with void cells filled by the valid-cell median.

    The fill exists only so the derived layers are defined everywhere; the ``void`` mask
    travels alongside to flag where those values are not trustworthy. An all-void tile
    fills with zero. This is an *explicit, flagged* fill — never a silent interpolation.
    """
    filled = np.array(elev, dtype=F32, copy=True)
    valid = ~void
    fill_value = float(np.median(elev[valid])) if bool(valid.any()) else 0.0
    filled[void] = fill_value
    return filled


def slope_aspect(
    elev: NDArray[np.float32], resolution_m: float
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Slope (degrees from horizontal) and aspect (degrees clockwise from +y/north).

    Aspect is the compass direction of steepest descent in [0, 360); flat cells resolve
    to 0. Gradients use the pixel spacing ``resolution_m`` so both are in real units.
    """
    dz_dy, dz_dx = np.gradient(elev.astype(np.float64), resolution_m)
    slope = np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))
    aspect = np.degrees(np.arctan2(dz_dy, -dz_dx)) % 360.0
    return slope.astype(F32), aspect.astype(F32)


def roughness(elev: NDArray[np.float32]) -> NDArray[np.float32]:
    """Local surface roughness: the standard deviation of elevation in a 3x3 window (m)."""
    base = elev.astype(np.float64)
    padded = np.pad(base, 1, mode="edge")
    rows, cols = base.shape
    acc = np.zeros_like(base)
    acc_sq = np.zeros_like(base)
    for dy in (0, 1, 2):
        for dx in (0, 1, 2):
            window = padded[dy : dy + rows, dx : dx + cols]
            acc += window
            acc_sq += window * window
    n = 9.0
    variance = np.clip(acc_sq / n - (acc / n) ** 2, 0.0, None)
    return np.sqrt(variance).astype(F32)


def vertical_uncertainty(
    void: NDArray[np.bool_], baseline_m: float, void_factor: float
) -> NDArray[np.float32]:
    """A vertical-uncertainty field: ``baseline_m`` everywhere, inflated at void cells.

    Carried as a companion layer so downstream sim-to-real claims stay honest — the DEM's
    error and the extra uncertainty of void-filled cells travel with the elevation.
    """
    field = np.full(void.shape, baseline_m, dtype=F32)
    field[void] = np.float32(baseline_m * void_factor)
    return field


def normal_from_slope_aspect(slope_deg: float, aspect_deg: float) -> tuple[float, float, float]:
    """Outward surface normal (unit vector) from slope and aspect, in the grid frame."""
    s = np.radians(slope_deg)
    a = np.radians(aspect_deg)
    nx = float(np.sin(s) * np.sin(a))
    ny = float(np.sin(s) * np.cos(a))
    nz = float(np.cos(s))
    return (nx, ny, nz)


def terrain_hash(layers: dict[str, NDArray[Any]], meta: dict[str, Any]) -> str:
    """A deterministic ``sha256:`` digest over the layer arrays + canonical metadata.

    Reproducible from the same inputs (RM-P0-WORLDS-01 acceptance): the same DEM ingested from two
    clean checkouts hashes equal. ``meta``'s toolchain is provenance and is **not** covered — a
    toolchain that changes the output changes the layer bytes, which this hash already sees; see
    :mod:`astro_mine.worlds._hashing`.
    """
    h = hashlib.sha256()
    for name in sorted(layers):
        h.update(name.encode("utf-8"))
        h.update(np.ascontiguousarray(layers[name]).tobytes())
    h.update(canonical_meta_bytes(meta))
    return f"sha256:{h.hexdigest()}"
