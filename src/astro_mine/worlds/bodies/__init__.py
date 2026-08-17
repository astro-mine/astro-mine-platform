# SPDX-License-Identifier: Apache-2.0
"""Celestial-body packs — a new world is a package, not a core change (RM-P1-WORLDS-11).

A :class:`BodyPack` bundles everything body-specific that the (otherwise body-agnostic) Worlds
machinery needs to serve a world: the NAIF body + body-fixed/topocentric frames, the reference
radius/geoid, a gravity model (point-mass + J2 zonal harmonic), radiative/thermal constants,
the diurnal period, default thermophysical classes, an atmospheric-dust field model, and the
body's geographic + default projected CRS. :data:`MOON_PACK` reproduces the Phase-0 lunar
constants **exactly** (so the anchor is byte-identical), and :data:`MARS_PACK` adds Mars as a
plugin — validating charter §10.2's "support a new world = a package, not a Worlds/Core change".

Everything here is built from Core value types (``ReferenceFrame``/``PlanetaryCRS`` over plain
body-name strings) and the parametric geometry/thermal helpers — **no** Core narrow-waist change.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from astro_mine.core.units import MOON, FrameClass, PlanetaryCRS, ReferenceFrame
from astro_mine.core.world import Vector
from astro_mine.spice import Site
from astro_mine.worlds.crs import (
    LUNAR_SOUTH_POLAR_STEREOGRAPHIC,
    MOON_RADIUS_M,
    lunar_geographic_proj4,
)
from astro_mine.worlds.gravity import (
    MARS_GM_M3_S2,
    MARS_GRAVITY,
    MARS_J2,
    MOON_GRAVITY,
    GravityModel,
)
from astro_mine.worlds.provider._geometry import (
    BOND_ALBEDO,
    EMISSIVITY,
    SHADOW_FLOOR_K,
    SOLAR_CONSTANT_W_M2,
)
from astro_mine.worlds.thermal import TERRAIN_CLASSES, ThermalClass
from astro_mine.worlds.thermal._solver import LUNAR_DIURNAL_PERIOD_S

__all__ = [
    "MARS",
    "MARS_BODY_FIXED",
    "MARS_GM_M3_S2",
    "MARS_J2",
    "MARS_PACK",
    "MARS_RADIUS_M",
    "MARS_SOLAR_CONSTANT_W_M2",
    "MARS_SURFACE_FRAME",
    "MOON_PACK",
    "BodyPack",
    "DustModel",
    "mars_equidistant_cylindrical",
    "mars_geographic_proj4",
]

# --- Mars body constants ---------------------------------------------------------

#: NAIF body name (a plain string at the Core waist — no Core Body type change).
MARS = "MARS"
#: Mars volumetric mean radius (m) — the datum the Mars `+R` planetary CRS uses. The *gravity*
#: reference radius is a different number (3396.0 km) and lives with the field, in
#: :mod:`astro_mine.worlds.gravity`.
MARS_RADIUS_M = 3_389_500.0
#: Solar constant at Mars (~1.524 AU): 1361 / 1.524² ≈ 586 W·m⁻².
MARS_SOLAR_CONSTANT_W_M2 = 586.2
#: Martian sol (s) — the diurnal forcing period.
MARS_SOL_S = 88_775.0

MARS_BODY_FIXED = ReferenceFrame(name="IAU_MARS", frame_class=FrameClass.BODY_FIXED, center=MARS)
MARS_SURFACE_FRAME = ReferenceFrame(
    name="MARS_TOPO", frame_class=FrameClass.TOPOCENTRIC, center=MARS
)
_MOON_SURFACE_FRAME = ReferenceFrame(
    name="MOON_ME_TOPO", frame_class=FrameClass.TOPOCENTRIC, center=MOON
)


def mars_geographic_proj4(reference_radius_m: float = MARS_RADIUS_M) -> str:
    """A Martian body-fixed geographic (lat/lon) PROJ string on the Mars sphere."""
    return f"+proj=longlat +R={reference_radius_m:.1f} +no_defs"


def mars_equidistant_cylindrical(reference_radius_m: float = MARS_RADIUS_M) -> PlanetaryCRS:
    """The default global Mars projected CRS — equidistant cylindrical (MOLA-style)."""
    return PlanetaryCRS(
        body=MARS,
        body_fixed_frame=MARS_BODY_FIXED.name,
        reference_radius_m=reference_radius_m,
        projection=(
            f"+proj=eqc +lat_ts=0 +lon_0=0 +x_0=0 +y_0=0 "
            f"+R={reference_radius_m:.1f} +units=m +no_defs"
        ),
    )


@dataclass(frozen=True)
class DustModel:
    """A reduced-order atmospheric-dust environment (a field-model plugin, worlds.md §3).

    Column optical depth with a seasonal modulation, plus a lifting threshold and settling
    rate — the parameters Sim's dust/optical models consume. The Moon carries a near-zero
    (electrostatic-only) instance; Mars a real dusty atmosphere with seasonal storms.
    """

    name: str
    base_optical_depth: float
    seasonal_amplitude: float
    lifting_threshold_ms: float
    deposition_rate_kg_m2_s: float

    def optical_depth(self, season_phase: float) -> float:
        """Column optical depth τ at a seasonal phase in [0, 1) (peak modulation at 0.25)."""
        return self.base_optical_depth * (
            1.0 + self.seasonal_amplitude * math.sin(2.0 * math.pi * season_phase)
        )

    def is_lifting(self, friction_velocity_ms: float) -> bool:
        """Whether surface wind stress exceeds the dust-lifting threshold."""
        return friction_velocity_ms >= self.lifting_threshold_ms


@dataclass(frozen=True)
class BodyPack:
    """Everything body-specific the body-agnostic Worlds machinery needs to serve a world."""

    name: str
    body: str
    body_fixed_frame: ReferenceFrame
    surface_frame: ReferenceFrame
    reference_radius_m: float
    #: The body's published gravity field — GM + its low-order zonal harmonics + the reference
    #: radius *they* are normalized to (which is not :attr:`reference_radius_m`, the CRS datum).
    #: Both packs evaluate it through the one :mod:`astro_mine.worlds.gravity` kernel.
    gravity_model: GravityModel
    solar_constant_w_m2: float
    bond_albedo: float
    emissivity: float
    shadow_floor_k: float
    diurnal_period_s: float
    geographic_proj4: str
    default_crs: PlanetaryCRS
    dust: DustModel
    thermal_classes: dict[str, ThermalClass] = field(default_factory=dict)

    @property
    def gm_m3_s2(self) -> float:
        """The body's gravitational parameter GM (m³·s⁻²), from its gravity field."""
        return self.gravity_model.gm_m3_s2

    @property
    def j2(self) -> float:
        """The body's degree-2 zonal harmonic (oblateness); 0.0 for a pure point-mass field."""
        return self.gravity_model.j2

    def site(self, lat_deg: float, lon_deg: float) -> Site:
        """A body-fixed :class:`~astro_mine.spice.Site` on the body sphere at ``(lat, lon)``."""
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        r = self.reference_radius_m
        position = (
            r * math.cos(lat) * math.cos(lon),
            r * math.cos(lat) * math.sin(lon),
            r * math.sin(lat),
        )
        return Site(body=self.body, position_m=position, frame=self.body_fixed_frame)

    def gravity(self, position: Vector) -> Vector:
        """Local gravity vector at ``position`` in the surface frame — point-mass + zonal harmonics.

        Delegates to the pack's :class:`~astro_mine.worlds.gravity.GravityModel`, so the Moon
        (GRAIL J2/J3/J4) and Mars (J2) run the **same** kernel at whatever degree their published
        field justifies — worlds.md §11's "point-mass + spherical harmonics for the Moon/Mars".
        """
        return self.gravity_model.acceleration(position)


#: The lunar pack. Its gravity is **point-mass + low-order GRAIL zonal harmonics** (J2/J3/J4 from
#: GRGM1200A) — worlds.md §12's Phase-0 MVP line. It previously shipped ``j2 = 0.0`` ("point-mass
#: only"), which left the Moon — the anchor body — the *only* pack without the oblateness term §11
#: recommends; the correction is ~6e-4 of g at the poles the anchor operates at.
MOON_PACK = BodyPack(
    name="moon",
    body=MOON,
    body_fixed_frame=ReferenceFrame(name="MOON_ME", frame_class=FrameClass.BODY_FIXED, center=MOON),
    surface_frame=_MOON_SURFACE_FRAME,
    reference_radius_m=MOON_RADIUS_M,
    gravity_model=MOON_GRAVITY,
    solar_constant_w_m2=SOLAR_CONSTANT_W_M2,
    bond_albedo=BOND_ALBEDO,
    emissivity=EMISSIVITY,
    shadow_floor_k=SHADOW_FLOOR_K,
    diurnal_period_s=LUNAR_DIURNAL_PERIOD_S,
    geographic_proj4=lunar_geographic_proj4(),
    default_crs=LUNAR_SOUTH_POLAR_STEREOGRAPHIC,
    dust=DustModel(
        name="lunar_electrostatic",
        base_optical_depth=0.0,  # no atmospheric column; dust is electrostatic/mechanical
        seasonal_amplitude=0.0,
        lifting_threshold_ms=float("inf"),
        deposition_rate_kg_m2_s=0.0,
    ),
    thermal_classes=dict(TERRAIN_CLASSES),
)


#: Representative Martian thermophysical classes (higher inertia + albedo than the Moon).
MARS_THERMAL_CLASSES: dict[str, ThermalClass] = {
    "mars_equatorial": ThermalClass(
        name="mars_equatorial",
        thermal_inertia_tiu=200.0,
        density_kg_m3=1600.0,
        specific_heat_j_kg_k=700.0,
        albedo=0.25,
        emissivity=0.95,
        peak_sun_elevation_deg=85.0,
        environment_flux_w_m2=2.0,  # thin-atmosphere IR + subsurface floor (~150 K night)
    ),
    "mars_polar": ThermalClass(
        name="mars_polar",
        thermal_inertia_tiu=250.0,
        density_kg_m3=1600.0,
        specific_heat_j_kg_k=700.0,
        albedo=0.30,
        emissivity=0.95,
        peak_sun_elevation_deg=25.0,
        environment_flux_w_m2=2.0,
    ),
}


#: The Mars pack — Mars ships purely as a plugin (RM-P1-WORLDS-11).
MARS_PACK = BodyPack(
    name="mars",
    body=MARS,
    body_fixed_frame=MARS_BODY_FIXED,
    surface_frame=MARS_SURFACE_FRAME,
    reference_radius_m=MARS_RADIUS_M,
    gravity_model=MARS_GRAVITY,
    solar_constant_w_m2=MARS_SOLAR_CONSTANT_W_M2,
    bond_albedo=0.25,
    emissivity=0.95,
    shadow_floor_k=150.0,
    diurnal_period_s=MARS_SOL_S,
    geographic_proj4=mars_geographic_proj4(),
    default_crs=mars_equidistant_cylindrical(),
    dust=DustModel(
        name="mars_atmospheric",
        base_optical_depth=0.5,
        seasonal_amplitude=0.6,  # dust-storm season modulation
        lifting_threshold_ms=2.5,
        deposition_rate_kg_m2_s=1.0e-9,
    ),
    thermal_classes=MARS_THERMAL_CLASSES,
)
