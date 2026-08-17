# SPDX-License-Identifier: Apache-2.0
"""Link products: ContactPlan, ConnectivitySampler, CommsObservationMask (RM-P0-LINK-04).

The content-addressed products Link emits and Sim consumes through the Core Environment
API. :func:`build_contact_plan` reduces LINK-02 contact windows (+ LINK-03 link budgets)
into a Core :class:`~astro_mine.core.messages.ContactPlan`; :class:`ConnectivitySampler`
answers on-demand ``connectivity(epoch)`` from that plan and emits the per-tick, per-agent
:class:`~astro_mine.core.messages.CommsObservationMask` that gates what a policy can see and
exchange. Link defines no new message types — it produces the Core ones (conventions.md §1.1).

Backlog: RM-P0-LINK-04 -- astro-mine-link#4
"""

from __future__ import annotations

from astro_mine.link.products._errors import LinkProductsError
from astro_mine.link.products._ops import (
    LIVE_MISSION_LINK_PREDICTION,
    EarthLinkSchedule,
    EarthLinkWindow,
    earth_link_schedule,
)
from astro_mine.link.products._plan import build_contact_plan
from astro_mine.link.products._sampler import ConnectivitySampler, LinkState
from astro_mine.link.products._series import (
    CubeManifest,
    CubeSlice,
    contact_edge_table,
    emit_time_series,
    read_cube_pair,
    read_cube_time_window,
    write_contact_edge_table,
)

__all__ = [
    "LIVE_MISSION_LINK_PREDICTION",
    "ConnectivitySampler",
    "CubeManifest",
    "CubeSlice",
    "EarthLinkSchedule",
    "EarthLinkWindow",
    "LinkProductsError",
    "LinkState",
    "build_contact_plan",
    "contact_edge_table",
    "earth_link_schedule",
    "emit_time_series",
    "read_cube_pair",
    "read_cube_time_window",
    "write_contact_edge_table",
]
