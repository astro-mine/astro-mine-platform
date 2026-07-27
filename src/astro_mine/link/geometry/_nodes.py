"""Link nodes: the endpoints of a line-of-sight query.

A node is either fixed to the body surface (its body-fixed position is known up front) or
an ephemeris target whose position is resolved from SPICE at query time.
"""

from __future__ import annotations

from dataclasses import dataclass

from astro_mine.core.world import Vector

__all__ = ["EphemerisNode", "Node", "SurfaceNode"]


@dataclass(frozen=True)
class SurfaceNode:
    """A node fixed to the body surface — a surface agent, rim tower, or body-side antenna.

    ``position_m`` is body-fixed Cartesian metres relative to the body centre, expressed in
    the world provider's :attr:`~astro_mine.core.world.WorldProvider.frame`.
    """

    name: str
    position_m: Vector


@dataclass(frozen=True)
class EphemerisNode:
    """A node whose position is resolved from SPICE at query time — a relay orbiter or Earth.

    ``target`` is the NAIF body name or id the ephemeris provider resolves (e.g. ``"EARTH"``
    or a relay orbiter's SPK id).
    """

    name: str
    target: str


Node = SurfaceNode | EphemerisNode
