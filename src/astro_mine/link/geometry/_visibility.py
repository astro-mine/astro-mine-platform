"""Geometric line-of-sight with terrain occlusion (RM-P0-LINK-01).

Composes SPICE ephemeris geometry (where a relay orbiter / Earth is, via
:mod:`astro_mine.spice`) with Worlds terrain occlusion served through the Core
:class:`~astro_mine.core.world.WorldProvider` contract (horizon-map ``line_of_sight``), to
decide whether two nodes can see each other at an epoch. Geometry is ground truth.

**Degrade loudly.** A missing world provider, a missing ephemeris provider for an ephemeris
node, a frame with no centre body, or a missing SPICE kernel all raise rather than assume
connectivity — Link never defaults a link to "connected" (link.md §2.9; this issue's
acceptance criteria).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from astro_mine.core.units import Epoch, ReferenceFrame, require_frame
from astro_mine.core.world import Vector, WorldProvider
from astro_mine.link.geometry._ephemeris import EphemerisProvider
from astro_mine.link.geometry._errors import LinkGeometryError
from astro_mine.link.geometry._nodes import EphemerisNode, Node, SurfaceNode

__all__ = ["LosResult", "compute_los"]


@dataclass(frozen=True)
class LosResult:
    """The line-of-sight verdict for an ordered ``observer -> target`` pair at an epoch.

    Carries the boolean visibility, the slant range (metres), and the frame/epoch the
    result was computed in — provenance a downstream contact plan or mask can record.
    """

    observer: str
    target: str
    visible: bool
    range_m: float
    epoch: Epoch
    frame: ReferenceFrame


def compute_los(
    observer: Node,
    target: Node,
    epoch: Epoch,
    *,
    world: WorldProvider,
    ephemeris: EphemerisProvider | None = None,
) -> LosResult:
    """Whether ``observer`` can see ``target`` at ``epoch`` — geometric LOS + terrain occlusion.

    Resolves each node's body-fixed position — surface nodes are fixed, ephemeris nodes
    (relay orbiter / Earth) are resolved from ``ephemeris`` (SPICE) — then asks the Core
    :class:`~astro_mine.core.world.WorldProvider` for the horizon-aware line of sight.

    Raises :class:`LinkGeometryError` when ``world`` is missing, the frame has no centre
    body, or an ephemeris node is supplied without an ``ephemeris`` provider; SPICE
    kernel/geometry errors propagate unchanged. Never defaults to "connected".
    """
    if world is None:
        raise LinkGeometryError(
            "no world provider — cannot evaluate terrain occlusion; refusing to assume connectivity"
        )
    frame = require_frame(world.frame)
    observer_pos = _resolve(observer, epoch, frame, ephemeris)
    target_pos = _resolve(target, epoch, frame, ephemeris)
    visible = bool(world.line_of_sight(observer_pos, target_pos, epoch=epoch))
    return LosResult(
        observer=observer.name,
        target=target.name,
        visible=visible,
        range_m=math.dist(observer_pos, target_pos),
        epoch=epoch,
        frame=frame,
    )


def _resolve(
    node: Node, epoch: Epoch, frame: ReferenceFrame, ephemeris: EphemerisProvider | None
) -> Vector:
    """The body-fixed position of ``node`` in ``frame`` at ``epoch``."""
    if isinstance(node, SurfaceNode):
        return node.position_m
    if isinstance(node, EphemerisNode):
        if ephemeris is None:
            raise LinkGeometryError(
                f"node {node.name!r} is an ephemeris target ({node.target!r}) but no "
                "ephemeris provider was given"
            )
        return ephemeris.position_body_fixed(node.target, epoch, frame=frame)
    raise LinkGeometryError(  # pragma: no cover - exhaustive over the Node union
        f"unsupported node type: {type(node).__name__}"
    )
