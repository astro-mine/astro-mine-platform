# SPDX-License-Identifier: Apache-2.0
"""Relay-orbiter contact windows (LINK-02): when a surface asset can see the relay.

Binds LINK-01's terrain-occluded line-of-sight (:func:`~astro_mine.link.geometry.compute_los`)
as the visibility predicate and reduces it over an epoch window. Because the predicate carries
Worlds terrain occlusion, a surface agent inside a PSR genuinely *loses* the relay — the
comms-denial that makes the anchor scenario hard (scenario §5; link.md §12).

Phase-0 baseline: a single relay orbiter. Multi-hop / constellation reachability is P1.
"""

from __future__ import annotations

from astro_mine.core.units import Epoch, EpochWindow
from astro_mine.core.world import WorldProvider
from astro_mine.link.geometry import EphemerisProvider, Node, compute_los
from astro_mine.link.windows._search import search_windows
from astro_mine.link.windows._window import ContactWindow

__all__ = ["relay_contact_windows"]


def relay_contact_windows(
    observer: Node,
    orbiter: Node,
    window: EpochWindow,
    step_s: float,
    *,
    world: WorldProvider,
    ephemeris: EphemerisProvider | None = None,
    refine_s: float | None = None,
) -> list[ContactWindow]:
    """Contact windows for ``observer -> orbiter`` over ``window``, sampled every ``step_s``.

    ``observer`` is typically a surface asset and ``orbiter`` the relay (an
    :class:`~astro_mine.link.geometry.EphemerisNode` resolved from ``ephemeris``), but any
    :class:`~astro_mine.link.geometry.Node` pair works — the same machinery yields
    rover↔rim-tower windows. Visibility is LINK-01's terrain-occluded LOS in ``world``'s
    frame; SPICE/provider errors propagate (the search never assumes connectivity).
    """

    def visible_at(epoch: Epoch) -> bool:
        return compute_los(observer, orbiter, epoch, world=world, ephemeris=ephemeris).visible

    return search_windows(
        (observer.name, orbiter.name), visible_at, window, step_s, refine_s=refine_s
    )
