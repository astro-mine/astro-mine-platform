"""World-provider API contract-test utility (worlds.md §5/§6).

The consumer-driven conformance check a Worlds implementation runs in its own CI to prove it
honors the Core :class:`~astro_mine.core.world.protocol.WorldProvider` contract — analogous
to :func:`astro_mine.core.env.check_environment`. It drives the query surface at a sample
point and asserts the contract: the object satisfies the Protocol, ``frame`` is a
:class:`~astro_mine.core.units.ReferenceFrame`, ``sample`` returns a well-formed
:class:`~astro_mine.core.world.model.SurfacePoint` (frame/elevation/normal/gravity/
illumination/temperature/regolith of the right types), and ``ray_intersect`` / ``line_of_sight``
return the contracted shapes. Raises :class:`WorldProviderContractError` on any violation.
"""

from __future__ import annotations

from astro_mine.core.units import ReferenceFrame
from astro_mine.core.world.model import (
    Illumination,
    IlluminationState,
    RegolithParams,
    SurfacePoint,
    Vector,
)
from astro_mine.core.world.protocol import WorldProvider

__all__ = ["WorldProviderContractError", "check_world_provider"]


class WorldProviderContractError(AssertionError):
    """Raised when a provider violates the Core WorldProvider API contract."""


def _check_vector(value: object, where: str) -> None:
    ok = isinstance(value, tuple) and len(value) == 3 and all(isinstance(c, float) for c in value)
    if not ok:
        raise WorldProviderContractError(f"{where} must be a 3-tuple of floats")


def check_world_provider(provider: WorldProvider, *, position: Vector = (0.0, 0.0, 0.0)) -> None:
    """Assert ``provider`` honors the Core WorldProvider API v0.1 contract.

    Drives ``frame``, ``sample``, ``ray_intersect``, and ``line_of_sight`` at ``position``
    and checks the return shapes — in particular that ``sample`` yields a fully-typed
    :class:`~astro_mine.core.world.model.SurfacePoint`. Returns ``None`` on success.
    """
    if not isinstance(provider, WorldProvider):
        raise WorldProviderContractError(
            "object does not satisfy the WorldProvider protocol (missing frame/sample/"
            "ray_intersect/line_of_sight)"
        )
    if not isinstance(provider.frame, ReferenceFrame):
        raise WorldProviderContractError("frame must be a units.ReferenceFrame")

    point = provider.sample(position)
    if not isinstance(point, SurfacePoint):
        raise WorldProviderContractError(
            f"sample() must return a SurfacePoint, got {type(point).__name__}"
        )
    if not isinstance(point.frame, ReferenceFrame):
        raise WorldProviderContractError("SurfacePoint.frame must be a units.ReferenceFrame")
    if not isinstance(point.elevation_m, float) or not isinstance(point.temperature_k, float):
        raise WorldProviderContractError("elevation_m and temperature_k must be floats")
    _check_vector(point.surface_normal, "SurfacePoint.surface_normal")
    _check_vector(point.gravity, "SurfacePoint.gravity")
    if not isinstance(point.illumination, Illumination) or not isinstance(
        point.illumination.state, IlluminationState
    ):
        raise WorldProviderContractError(
            "SurfacePoint.illumination must carry an IlluminationState"
        )
    if not isinstance(point.regolith, RegolithParams):
        raise WorldProviderContractError("SurfacePoint.regolith must be a RegolithParams")

    hit = provider.ray_intersect(position, (0.0, 0.0, -1.0))
    if hit is not None:
        _check_vector(hit, "ray_intersect() result")
    if not isinstance(provider.line_of_sight(position, (1.0, 0.0, 0.0)), bool):
        raise WorldProviderContractError("line_of_sight() must return a bool")
