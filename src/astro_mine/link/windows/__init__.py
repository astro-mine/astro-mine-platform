"""Relay-orbiter and ground-station contact windows (RM-P0-LINK-02).

Reduces the time-varying visibility series into contact intervals over an epoch window: a
generic rise/set :func:`search_windows` (with optional bisection refinement), the Moon-side
:func:`relay_contact_windows` (terrain-occluded LOS to the relay, built on LINK-01), and the
Earth-side :func:`dsn_contact_windows` (a :class:`GroundStation` elevation mask over the shared
:mod:`astro_mine.spice` topocentric geometry). Connectivity is a function of epoch by
construction; intervals are the first-class product (link.md §2.3). Degrades loudly — provider
errors propagate, never a silent "no contact".

Backlog: RM-P0-LINK-02 -- astro-mine-link#2
"""

from __future__ import annotations

from astro_mine.link.windows._errors import LinkWindowError
from astro_mine.link.windows._ground import (
    EARTH_BODY_FIXED,
    EARTH_RADIUS_M,
    GroundStation,
    SpiceTopocentric,
    TopocentricProvider,
    dsn_contact_windows,
)
from astro_mine.link.windows._relay import relay_contact_windows
from astro_mine.link.windows._search import VisibilityPredicate, search_windows
from astro_mine.link.windows._window import ContactWindow

__all__ = [
    "EARTH_BODY_FIXED",
    "EARTH_RADIUS_M",
    "ContactWindow",
    "GroundStation",
    "LinkWindowError",
    "SpiceTopocentric",
    "TopocentricProvider",
    "VisibilityPredicate",
    "dsn_contact_windows",
    "relay_contact_windows",
    "search_windows",
]
