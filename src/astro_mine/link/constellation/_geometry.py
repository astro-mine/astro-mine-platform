"""Space-side (relay↔relay) occultation geometry for the constellation (RM-P1-LINK-10).

Surface-involved links (surface↔surface, surface↔relay, surface↔Earth) are decided by
terrain-occluded LOS through the Core :class:`~astro_mine.core.world.WorldProvider`
contract (reusing :func:`~astro_mine.link.geometry.compute_los`). But a **relay↔relay**
link is high above the terrain — its only occluder is the *central body itself*. This
module supplies that one missing primitive: whether the straight segment between two
orbiter positions clears a sphere of the body's radius centred at the body origin (the
body-fixed frame's centre). Geometry is ground truth (link.md §2.1); like the rest of
Link it never assumes connectivity.
"""

from __future__ import annotations

import math

from astro_mine.core.world import Vector

__all__ = ["body_occulted_los"]


def body_occulted_los(a: Vector, b: Vector, body_radius_m: float) -> bool:
    """Whether the segment ``a→b`` clears a body sphere of ``body_radius_m`` at the origin.

    Both positions are body-fixed Cartesian metres relative to the body centre (the frame's
    centre is the sphere centre). Returns ``True`` when the closest approach of the segment
    to the origin is at or beyond ``body_radius_m`` — i.e. the body does not cut the line of
    sight — and ``False`` when the body occults it (two relays on opposite sides of the
    Moon). A non-positive radius means "no occluding body": always visible.

    Reduced-order: the body is a sphere (the mean radius), the standard Phase-1 relay-relay
    occultation model; an oblate/terrain-accurate limb is a later fidelity tier.
    """
    if body_radius_m <= 0.0:
        return True
    ax, ay, az = a
    bx, by, bz = b
    dx, dy, dz = bx - ax, by - ay, bz - az
    seg_sq = dx * dx + dy * dy + dz * dz
    if seg_sq == 0.0:  # degenerate: coincident endpoints — distance of the point itself
        return math.sqrt(ax * ax + ay * ay + az * az) >= body_radius_m
    # Parameter of the closest point on the *segment* to the origin, clamped to [0, 1].
    t = -(ax * dx + ay * dy + az * dz) / seg_sq
    t = max(0.0, min(1.0, t))
    cx, cy, cz = ax + t * dx, ay + t * dy, az + t * dz
    return math.sqrt(cx * cx + cy * cy + cz * cz) >= body_radius_m
