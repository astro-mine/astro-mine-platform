# SPDX-License-Identifier: Apache-2.0
"""World-provider API — the Core Environment-API world/terrain surface (worlds.md §5/§6).

A Core-owned thin Protocol for the world/terrain query surface Worlds implements and Sim
consumes: given a position and epoch, return ground elevation, surface normal and frame,
local gravity, illumination/solar flux, surface temperature, and the regolith terramechanics
parameter tuple — plus terrain ray-casting and horizon line-of-sight, the occlusion query
Link uses for visibility (worlds.md §6). Body packs, DEMs, illumination/thermal/regolith
field models, and the geometric kernels are plugins behind it (worlds.md §3); Core owns only
the *shape* — no SPICE/PROJ resolution, geometry, or IO.

Public API:

- the contract — :class:`WorldProvider` (runtime-checkable Protocol);
- the result — :class:`SurfacePoint` and the value types it carries (:class:`Illumination`,
  :class:`IlluminationState`, :class:`RegolithParams`, the :data:`Vector` type);
- conformance — :func:`check_world_provider` and :class:`WorldProviderContractError`.
"""

from __future__ import annotations

from astro_mine.core.world import conformance, model, protocol
from astro_mine.core.world.conformance import (
    WorldProviderContractError,
    check_world_provider,
)
from astro_mine.core.world.model import (
    Illumination,
    IlluminationState,
    RegolithParams,
    SurfacePoint,
    Vector,
)
from astro_mine.core.world.protocol import WorldProvider

__all__ = [
    "Illumination",
    "IlluminationState",
    "RegolithParams",
    "SurfacePoint",
    "Vector",
    "WorldProvider",
    "WorldProviderContractError",
    "check_world_provider",
    "conformance",
    "model",
    "protocol",
]
