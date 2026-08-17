# SPDX-License-Identifier: Apache-2.0
"""Geometric line-of-sight + terrain occlusion (RM-P0-LINK-01).

LOS via SPICE ephemeris geometry (:mod:`astro_mine.spice`) composed with Worlds terrain
occlusion consumed through the Core :class:`~astro_mine.core.world.WorldProvider` contract.
Geometry is ground truth; RF is a layer on top. Degrades loudly on missing
kernels/providers/frames -- never defaults to "connected".

Backlog: RM-P0-LINK-01 -- astro-mine-link#1
"""

from __future__ import annotations

from astro_mine.link.geometry._ephemeris import EphemerisProvider, SpiceEphemeris
from astro_mine.link.geometry._errors import LinkGeometryError
from astro_mine.link.geometry._nodes import EphemerisNode, Node, SurfaceNode
from astro_mine.link.geometry._visibility import LosResult, compute_los

__all__ = [
    "EphemerisNode",
    "EphemerisProvider",
    "LinkGeometryError",
    "LosResult",
    "Node",
    "SpiceEphemeris",
    "SurfaceNode",
    "compute_los",
]
