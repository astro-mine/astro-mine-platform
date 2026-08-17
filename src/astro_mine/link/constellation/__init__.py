# SPDX-License-Identifier: Apache-2.0
"""Relay constellation geometry + multi-hop reachability (RM-P1-LINK-10 / -13).

Extends the P0 single-relay MVP to a richer relay constellation: multiple orbiters as
:class:`~astro_mine.link.geometry.EphemerisNode`\\ s with independent ephemerides
(:class:`ConstellationScenario`, :func:`constellation_contact_windows`), central-body
relay↔relay occultation (:func:`body_occulted_los`), **instantaneous multi-hop reachability**
surface → relay → … → Earth (:func:`reachable_route`, :func:`reachability_windows`,
:func:`build_routes`), and content-addressed **ground-station catalogs beyond DSN**
(:class:`GroundStationCatalog`, :func:`default_ground_catalog`). Store-and-forward delivery is
the separate :mod:`~astro_mine.link.network` layer (RM-P1-LINK-11).

Backlog: RM-P1-LINK-10 -- astro-mine-link#17
"""

from __future__ import annotations

from astro_mine.link.constellation._catalog import (
    GroundStationCatalog,
    builtin_catalog,
    default_ground_catalog,
    load_ground_catalog,
)
from astro_mine.link.constellation._errors import LinkConstellationError
from astro_mine.link.constellation._geometry import body_occulted_los
from astro_mine.link.constellation._reach import (
    build_routes,
    ground_node_ids,
    reachability_windows,
    reachable_route,
    route_exists,
)
from astro_mine.link.constellation._scenario import (
    ConstellationScenario,
    constellation_contact_windows,
    contact_nodes,
)

__all__ = [
    "ConstellationScenario",
    "GroundStationCatalog",
    "LinkConstellationError",
    "body_occulted_los",
    "build_routes",
    "builtin_catalog",
    "constellation_contact_windows",
    "contact_nodes",
    "default_ground_catalog",
    "ground_node_ids",
    "load_ground_catalog",
    "reachability_windows",
    "reachable_route",
    "route_exists",
]
