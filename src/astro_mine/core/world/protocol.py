# SPDX-License-Identifier: Apache-2.0
"""World-provider API v0.1 — the contract (worlds.md §5/§6).

The Core-owned Environment-API *world/terrain* query surface Worlds implements and Sim
consumes (worlds.md §6): given a position and epoch, return the ground geometry, surface
frame, local gravity, illumination/solar flux, surface temperature, and the regolith
terramechanics parameter tuple at that point — plus terrain ray-casting and horizon
line-of-sight, the occlusion machinery Link queries for inter-agent and Earth visibility
(worlds.md §6). The body packs, DEMs, illumination/thermal/regolith field models, and the
hot geometric kernels are all plugins behind this contract (worlds.md §3); Core owns only
the *shape* — no SPICE/PROJ resolution, no geometry, no IO (those are Worlds').

:func:`~astro_mine.core.world.check_world_provider` is the consumer-driven contract test an
implementor (Worlds) runs in its own CI.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from astro_mine.core.units import Epoch, ReferenceFrame
from astro_mine.core.world.model import SurfacePoint, Vector

__all__ = ["WorldProvider"]


@runtime_checkable
class WorldProvider(Protocol):
    """The Core world/terrain contract Worlds implements (worlds.md §6).

    All spatial queries resolve in :attr:`frame` (the body-fixed query frame, SI metres); an
    absolute ``epoch`` is a :class:`~astro_mine.core.units.Epoch` for the time-varying
    illumination/thermal state. Determinism is contractual: the same (position, epoch) MUST
    yield an identical :class:`~astro_mine.core.world.model.SurfacePoint`.
    """

    @property
    def frame(self) -> ReferenceFrame:
        """The body-fixed reference frame queried positions and rays resolve in."""
        ...

    def sample(self, position: Vector, *, epoch: Epoch | None = None) -> SurfacePoint:
        """Query the surface at ``position`` (and ``epoch``) — geometry, gravity,
        illumination, temperature, and regolith in one :class:`SurfacePoint`."""
        ...

    def ray_intersect(self, origin: Vector, direction: Vector) -> Vector | None:
        """The first terrain intersection of the ray from ``origin`` along ``direction``,
        or ``None`` if it misses — the occlusion primitive Link uses for line-of-sight."""
        ...

    def line_of_sight(
        self, observer: Vector, target: Vector, *, epoch: Epoch | None = None
    ) -> bool:
        """Whether ``target`` is visible from ``observer`` over the horizon (unoccluded by
        terrain) — the horizon-LOS query backing inter-agent and Earth-link visibility."""
        ...
