# SPDX-License-Identifier: Apache-2.0
"""Provider geometry kernels — pure, IO-free (RM-P0-WORLDS-06).

Coordinate-independent helpers the Environment-API world provider composes: a first-cut
illumination-derived surface temperature (a documented stand-in until the RM-P0-WORLDS-04
thermal model lands), and the topocentric elevation/azimuth of an arbitrary target seen from
an observer — the angle the horizon-map line-of-sight thresholds.

**Gravity moved.** ``point_mass_gravity`` / ``gravity_j2`` used to be *implemented* here, with a
single hard-coded J2 term; they now live in :mod:`astro_mine.worlds.gravity` — the module
worlds.md §3's map reserves for "point-mass + spherical-harmonic gravity field evaluation" — so the
lunar and Martian packs share one zonal kernel of selectable degree instead of duplicating it. They
are re-exported here unchanged, because that is the name every caller (and the provider) knows them
by; the lunar GM likewise now comes from the published GRAIL field rather than a bare constant.

Pure math over plain 3-tuples (Core's :data:`~astro_mine.core.world.Vector`); no rasterio,
SPICE, or IO (those live in the provider). Body-fixed vectors are SI metres relative to the
body centre; angles are degrees.
"""

from __future__ import annotations

import math

from astro_mine.core.world import Vector
from astro_mine.worlds.gravity import GRGM1200A_GM_M3_S2, gravity_j2, point_mass_gravity

__all__ = [
    "BOND_ALBEDO",
    "EMISSIVITY",
    "MOON_GM_M3_S2",
    "SHADOW_FLOOR_K",
    "SOLAR_CONSTANT_W_M2",
    "STEFAN_BOLTZMANN_W_M2_K4",
    "add_scaled",
    "cross",
    "dot",
    "equilibrium_temperature",
    "gravity_j2",
    "norm",
    "point_mass_gravity",
    "solar_flux",
    "topocentric_elevation_azimuth",
    "unit",
]

#: Lunar gravitational parameter GM (m³·s⁻²); surface g = GM/R² ≈ 1.62 m·s⁻². Sourced from the
#: GRAIL **GRGM1200A** field (:mod:`astro_mine.worlds.gravity`) so that GM, the reference radius,
#: and the zonal harmonics all come from **one** published solution — the previous bare constant
#: (4.9028695e12) had no citation and disagreed with every GRAIL field at the 1e-5 level.
MOON_GM_M3_S2 = GRGM1200A_GM_M3_S2

#: Total solar irradiance at ~1 AU (W·m⁻²) — the Moon is treated as 1 AU from the Sun.
SOLAR_CONSTANT_W_M2 = 1361.0

#: Lunar Bond albedo (fraction of incident flux reflected).
BOND_ALBEDO = 0.12

#: Surface emissivity for the radiative-equilibrium temperature stand-in.
EMISSIVITY = 0.95

#: Stefan-Boltzmann constant (W·m⁻²·K⁻⁴).
STEFAN_BOLTZMANN_W_M2_K4 = 5.670374419e-8

#: Shadow/PSR temperature floor (K) — the deep-shadow value (~tens of K, scenario §5).
SHADOW_FLOOR_K = 40.0


# --- small vector helpers (plain 3-tuples; no numpy) -------------------------------


def dot(a: Vector, b: Vector) -> float:
    """Dot product of two 3-vectors."""
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a: Vector, b: Vector) -> Vector:
    """Cross product ``a x b``."""
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm(a: Vector) -> float:
    """Euclidean length of a 3-vector."""
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def unit(a: Vector) -> Vector | None:
    """Unit vector along ``a``, or ``None`` for the zero vector (direction undefined)."""
    n = norm(a)
    if n == 0.0:
        return None
    return (a[0] / n, a[1] / n, a[2] / n)


def add_scaled(origin: Vector, direction: Vector, t: float) -> Vector:
    """``origin + t · direction`` — a point ``t`` along a ray."""
    return (
        origin[0] + direction[0] * t,
        origin[1] + direction[1] * t,
        origin[2] + direction[2] * t,
    )


# --- physical first-cuts -----------------------------------------------------------


def solar_flux(
    elevation_deg: float, *, lit: bool, solar_constant_w_m2: float = SOLAR_CONSTANT_W_M2
) -> float:
    """Incident solar flux (W·m⁻²) on the local horizontal at a surface point.

    ``solar_constant · sin(elevation)`` when the Sun is above the terrain horizon (``lit``)
    and geometrically up, else 0. ``solar_constant_w_m2`` defaults to the ~1 AU lunar value; a
    body pack passes its own (Mars ~586 W·m⁻² at 1.52 AU, RM-P1-WORLDS-11). Binary lit/shadow —
    finite-solar-disc penumbra is deferred (worlds.md §11 open question).
    """
    if not lit:
        return 0.0
    s = math.sin(math.radians(elevation_deg))
    return solar_constant_w_m2 * s if s > 0.0 else 0.0


def equilibrium_temperature(
    flux_w_m2: float,
    *,
    bond_albedo: float = BOND_ALBEDO,
    emissivity: float = EMISSIVITY,
    shadow_floor_k: float = SHADOW_FLOOR_K,
) -> float:
    """First-cut radiative-equilibrium surface temperature (K) from incident ``flux``.

    ``T = ((1-A)*flux / (eps*sigma))^(1/4)`` for sunlit flux, falling to ``shadow_floor_k`` in
    shadow. The albedo/emissivity/floor default to the lunar values; a body pack passes its own
    (RM-P1-WORLDS-11). No thermal inertia or diurnal lag — the RM-P1-WORLDS-13
    illumination-driven curve is the high-fidelity replacement.
    """
    if flux_w_m2 <= 0.0:
        return shadow_floor_k
    absorbed = (1.0 - bond_albedo) * flux_w_m2
    return float((absorbed / (emissivity * STEFAN_BOLTZMANN_W_M2_K4)) ** 0.25)


def topocentric_elevation_azimuth(observer: Vector, target: Vector) -> tuple[float, float]:
    """Elevation above the local horizontal and azimuth of ``target`` from ``observer``.

    Both are body-fixed vectors from the body centre (m). Returns ``(elevation_deg,
    azimuth_deg)`` with elevation in [-90, 90] above the local horizontal plane and azimuth
    clockwise from local north in [0, 360) — the same convention as the SPICE topocentric
    geometry, so the azimuth feeds ``topocentric_to_world_azimuth`` directly. Elevation is
    geometric (spherical local vertical); the terrain-horizon comparison is the provider's
    line-of-sight. Azimuth is 0 on the spin axis (a pole), where it is undefined.

    Raises :class:`ValueError` if ``target`` coincides with ``observer`` or ``observer`` is
    the body centre (the local vertical is then undefined).
    """
    delta = (target[0] - observer[0], target[1] - observer[1], target[2] - observer[2])
    rng = norm(delta)
    if rng == 0.0:
        raise ValueError("target coincides with observer; geometry is undefined")
    direction = (delta[0] / rng, delta[1] / rng, delta[2] / rng)

    up = unit(observer)
    if up is None:
        raise ValueError("observer is the body centre; the local vertical is undefined")
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, dot(direction, up)))))

    east = cross((0.0, 0.0, 1.0), up)
    east_norm = norm(east)
    if east_norm < 1e-9:  # observer on the spin axis (a pole) — azimuth is undefined
        return elevation, 0.0
    east = (east[0] / east_norm, east[1] / east_norm, east[2] / east_norm)
    north = cross(up, east)
    azimuth = math.degrees(math.atan2(dot(direction, east), dot(direction, north))) % 360.0
    return elevation, azimuth
