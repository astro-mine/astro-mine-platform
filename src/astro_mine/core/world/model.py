# SPDX-License-Identifier: Apache-2.0
"""World-provider API — result containers (worlds.md §5/§6).

The frozen, in-memory returns of the Core Environment-API world/terrain surface Worlds
implements: a :class:`SurfacePoint` bundling the per-point geometry/illumination/thermal/
regolith query, plus the small value types it carries. Pure data — the constitutive laws
that consume the regolith tuple live in Sim (worlds.md §6); Core owns only the shape.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from astro_mine.core.units import ReferenceFrame

__all__ = [
    "Illumination",
    "IlluminationState",
    "RegolithParams",
    "SurfacePoint",
    "Vector",
]

#: A 3-D vector (position, normal, gravity, ray origin/direction) in a named frame (SI).
Vector = tuple[float, float, float]


class IlluminationState(StrEnum):
    """Solar-illumination state at a surface point (worlds.md §3 ``IlluminationModel``).

    ``LIT`` is full sun, ``SHADOW`` full umbra (the permanently-shadowed-region condition),
    ``PENUMBRA`` the partial-shadow transition. Closed; grows by RFC.
    """

    LIT = "lit"
    PENUMBRA = "penumbra"
    SHADOW = "shadow"


@dataclass(frozen=True, slots=True)
class Illumination:
    """Solar illumination at a surface point: the qualitative state plus the incident solar
    flux (W·m⁻²) used by power/thermal models (worlds.md §3)."""

    state: IlluminationState
    solar_flux_w_m2: float


@dataclass(frozen=True, slots=True)
class RegolithParams:
    """The regolith terramechanics/dust parameter tuple at a surface point (worlds.md §3/§6).

    Pure parameter *data*; the wheel/soil and contact constitutive models that consume it run
    in Sim. Every field is optional so a world that models only a subset still conforms.
    """

    bulk_density_kg_m3: float | None = None
    cohesion_pa: float | None = None
    friction_angle_deg: float | None = None
    bearing_capacity_pa: float | None = None
    thermal_inertia_tiu: float | None = None


@dataclass(frozen=True, slots=True)
class SurfacePoint:
    """The world/terrain query result at a (position, epoch) — the surface of the Core
    Environment API that Worlds owns and Sim consumes (worlds.md §6).

    ``frame`` is the local surface frame at the point (a
    :class:`~astro_mine.core.units.ReferenceFrame`); ``elevation_m`` the ground elevation;
    ``surface_normal`` the outward unit normal; ``gravity`` the local gravity vector
    (m·s⁻²); ``illumination`` the solar state + flux; ``temperature_k`` the surface
    temperature; ``regolith`` the terramechanics parameter handle.
    """

    frame: ReferenceFrame
    elevation_m: float
    surface_normal: Vector
    gravity: Vector
    illumination: Illumination
    temperature_k: float
    regolith: RegolithParams = field(default_factory=RegolithParams)
