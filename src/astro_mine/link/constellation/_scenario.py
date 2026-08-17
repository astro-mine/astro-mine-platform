# SPDX-License-Identifier: Apache-2.0
"""The relay constellation: a typed node set → a time-varying, multi-pair contact graph.

Extends the P0 single-relay MVP (RM-P0-LINK-01/02) to a **constellation** (RM-P1-LINK-10):
multiple relay orbiters — each an :class:`~astro_mine.link.geometry.EphemerisNode` with its
own SPK ephemeris — plus surface agents and Earth ground stations, all in one
:class:`ConstellationScenario`. :func:`constellation_contact_windows` computes contact
windows across the **full {surface x relay x ground} node set** over the epoch window, each
pair decided by the right geometry:

- surface↔surface and surface↔relay — terrain-occluded LOS through the Core ``WorldProvider``
  (:func:`~astro_mine.link.geometry.compute_los`); a PSR agent genuinely loses the relay;
- relay↔relay — central-body occultation (:func:`~astro_mine.link.constellation.body_occulted_los`),
  no terrain;
- relay↔ground — the Earth station's elevation mask on the relay (reusing the LINK-02
  topocentric geometry);
- surface↔ground — the station sees the body *and* the surface agent has an unoccluded
  terrain LOS toward Earth (a PSR agent has neither, so it reaches Earth only via a relay
  chain — the property RM-P1-LINK-10 must show).

Everything is a function of epoch by construction (link.md §2.3); the windows are the
first-class product, reduced from the same rise/set search the P0 windows use, and feed the
multi-hop reachability in :mod:`~astro_mine.link.constellation._reach`. Link imports only
Core + ``astro_mine.spice`` — terrain arrives through the injected ``WorldProvider``, never a
dependency on ``astro-mine-worlds`` (link.md §2.2, conventions.md §1.1).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from itertools import combinations

from astro_mine.core.messages import ContactNode
from astro_mine.core.messages.enums import NodeRole
from astro_mine.core.units import EpochWindow, require_frame
from astro_mine.core.world import WorldProvider
from astro_mine.link.constellation._errors import LinkConstellationError
from astro_mine.link.constellation._geometry import body_occulted_los
from astro_mine.link.geometry import EphemerisNode, EphemerisProvider, SurfaceNode, compute_los
from astro_mine.link.windows import (
    ContactWindow,
    GroundStation,
    TopocentricProvider,
    search_windows,
)
from astro_mine.spice import MOON_RADIUS_M

__all__ = [
    "ConstellationScenario",
    "constellation_contact_windows",
    "contact_nodes",
]

#: The free ``ContactNode.kind`` labels Link stamps by node category.
SURFACE_KIND = "surface_agent"
RELAY_KIND = "relay_orbiter"
GROUND_KIND = "ground_station"


#: Progress for the offline plan build. The library only *emits*; a CLI decides whether to
#: show it (``scripts/build_anchor_contact_plan.py`` turns it on).
_LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConstellationScenario:
    """The immutable inputs of a relay-constellation contact computation (link.md §3).

    ``surface`` are body-fixed agents/towers, ``relays`` are orbiters resolved from
    ``ephemeris`` (each an independent SPK target), and ``ground`` are Earth antennas. The
    ``world`` provider supplies terrain occlusion (its ``frame`` is the body-fixed query
    frame), ``ephemeris`` positions the relays/Earth, and ``topocentric`` gives Earth-station
    elevation. ``window``/``step_s``/``refine_s`` drive the rise/set search; ``body_radius_m``
    is the occulting-sphere radius for relay↔relay LOS (defaults to the Moon); ``earth_target``
    is the NAIF body a surface agent's direct-to-Earth path points at.

    Every node name MUST be unique across the three categories (a contact graph keyed by name
    cannot carry a collision); the node set MUST be non-empty. Both are checked in
    :meth:`__post_init__`, loudly (link.md §2.9).
    """

    surface: tuple[SurfaceNode, ...]
    relays: tuple[EphemerisNode, ...]
    ground: tuple[GroundStation, ...]
    world: WorldProvider
    ephemeris: EphemerisProvider
    topocentric: TopocentricProvider
    window: EpochWindow
    step_s: float
    refine_s: float | None = None
    body_radius_m: float = MOON_RADIUS_M
    earth_target: str = "EARTH"
    link_surface_to_ground: bool = True
    _names: frozenset[str] = field(init=False, repr=False, compare=False, default=frozenset())

    def __post_init__(self) -> None:
        names = [n.name for n in self.surface] + [n.name for n in self.relays]
        names += [g.name for g in self.ground]
        if not names:
            raise LinkConstellationError("empty constellation: declare at least one node")
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise LinkConstellationError(
                f"duplicate node name(s) {duplicates} across the constellation; every "
                "surface/relay/ground node needs a unique id"
            )
        object.__setattr__(self, "_names", frozenset(names))

    @property
    def node_names(self) -> frozenset[str]:
        """Every node id in the constellation."""
        return self._names


def contact_nodes(scenario: ConstellationScenario) -> list[ContactNode]:
    """The Core :class:`~astro_mine.core.messages.ContactNode` graph for ``scenario``.

    Surface agents and relay orbiters are ``SPACE`` role (Moon-side); ground stations are
    ``GROUND`` (Earth). The free ``kind`` label lets a consumer tell a relay from a rover
    without parsing the name (a role alone cannot — link.md §3)."""
    nodes = [
        ContactNode(id=n.name, role=NodeRole.SPACE, kind=SURFACE_KIND) for n in scenario.surface
    ]
    nodes += [ContactNode(id=n.name, role=NodeRole.SPACE, kind=RELAY_KIND) for n in scenario.relays]
    nodes += [
        ContactNode(id=g.name, role=NodeRole.GROUND, kind=GROUND_KIND) for g in scenario.ground
    ]
    return nodes


def expected_pair_count(scenario: ConstellationScenario) -> int:
    """How many node pairs :func:`constellation_contact_windows` will search.

    Derived from the node counts, not from running the search — so a caller can size a progress
    report (or an ETA) *before* the first pair is walked. Mirrors the pair enumeration below
    exactly; if one changes, so must the other."""
    n_surface, n_relay, n_ground = len(scenario.surface), len(scenario.relays), len(scenario.ground)
    pairs = (
        n_surface * (n_surface - 1) // 2  # surface <-> surface
        + n_surface * n_relay  # surface <-> relay
        + n_relay * (n_relay - 1) // 2  # relay   <-> relay
        + n_relay * n_ground  # relay   <-> ground
    )
    if scenario.link_surface_to_ground:
        pairs += n_surface * n_ground
    return pairs


def constellation_contact_windows(scenario: ConstellationScenario) -> list[ContactWindow]:
    """Contact windows across the **whole** node set — the time-varying multi-hop graph.

    Iterates every relevant node pair, binds the geometry-appropriate visibility predicate,
    and reduces it with the shared :func:`~astro_mine.link.windows.search_windows`. The union
    of all pairs' intervals is the constellation contact graph the reachability layer walks.
    Provider/kernel errors propagate — a pair is never silently marked connected.

    Progress is logged per pair at ``INFO`` (see :func:`_search`). The anchor's search is a
    30-day mission window stepped at 60 s and refined to 5 s across every pair, which takes ~27 min
    on one core — long enough that a maintainer running it offline needs to see it moving, and cheap
    enough that one log line per pair costs nothing.
    """
    _LOG.info(
        "contact-window search: %d pairs over %s (step %.0fs, refine %s)",
        expected_pair_count(scenario),
        scenario.window,
        scenario.step_s,
        f"{scenario.refine_s:.0f}s" if scenario.refine_s else "none",
    )
    windows: list[ContactWindow] = []
    windows += _surface_surface(scenario)
    windows += _surface_relay(scenario)
    windows += _relay_relay(scenario)
    windows += _relay_ground(scenario)
    if scenario.link_surface_to_ground:
        windows += _surface_ground(scenario)
    return windows


def _search(
    scenario: ConstellationScenario, pair: tuple[str, str], visible_at: object
) -> list[ContactWindow]:
    started = time.monotonic()
    found = search_windows(
        pair,
        visible_at,  # type: ignore[arg-type]
        scenario.window,
        scenario.step_s,
        refine_s=scenario.refine_s,
    )
    # Wall-clock, purely to report progress: it never reaches the plan, so the search stays a pure
    # function of (kernels, terrain, nodes, window, config) and the plan digest stays deterministic.
    _LOG.info(
        "  %s <-> %s: %d interval(s) in %.1fs",
        pair[0],
        pair[1],
        len(found),
        time.monotonic() - started,
    )
    return found


def _surface_surface(scenario: ConstellationScenario) -> list[ContactWindow]:
    out: list[ContactWindow] = []
    for a, b in combinations(scenario.surface, 2):
        out += _search(
            scenario,
            (a.name, b.name),
            lambda epoch, a=a, b=b: (
                compute_los(a, b, epoch, world=scenario.world, ephemeris=scenario.ephemeris).visible
            ),
        )
    return out


def _surface_relay(scenario: ConstellationScenario) -> list[ContactWindow]:
    out: list[ContactWindow] = []
    for a in scenario.surface:
        for r in scenario.relays:
            out += _search(
                scenario,
                (a.name, r.name),
                lambda epoch, a=a, r=r: (
                    compute_los(
                        a, r, epoch, world=scenario.world, ephemeris=scenario.ephemeris
                    ).visible
                ),
            )
    return out


def _relay_relay(scenario: ConstellationScenario) -> list[ContactWindow]:
    frame = require_frame(scenario.world.frame)
    out: list[ContactWindow] = []
    for a, b in combinations(scenario.relays, 2):
        out += _search(
            scenario,
            (a.name, b.name),
            lambda epoch, a=a, b=b: body_occulted_los(
                scenario.ephemeris.position_body_fixed(a.target, epoch, frame=frame),
                scenario.ephemeris.position_body_fixed(b.target, epoch, frame=frame),
                scenario.body_radius_m,
            ),
        )
    return out


def _relay_ground(scenario: ConstellationScenario) -> list[ContactWindow]:
    out: list[ContactWindow] = []
    for r in scenario.relays:
        for g in scenario.ground:
            out += _search(
                scenario,
                (r.name, g.name),
                lambda epoch, r=r, g=g: (
                    scenario.topocentric.elevation_deg(r.target, g.site, epoch)
                    >= g.min_elevation_deg
                ),
            )
    return out


def _surface_ground(scenario: ConstellationScenario) -> list[ContactWindow]:
    body = require_frame(scenario.world.frame).center
    if body is None:
        raise LinkConstellationError(
            f"world frame {scenario.world.frame.name!r} has no centre body; cannot evaluate "
            "the station's elevation of the body for a surface-to-ground link"
        )
    out: list[ContactWindow] = []
    earth = EphemerisNode(name=f"__{scenario.earth_target.lower()}__", target=scenario.earth_target)
    for a in scenario.surface:
        for g in scenario.ground:
            out += _search(
                scenario,
                (a.name, g.name),
                lambda epoch, a=a, g=g: (
                    scenario.topocentric.elevation_deg(body, g.site, epoch) >= g.min_elevation_deg
                    and compute_los(
                        a, earth, epoch, world=scenario.world, ephemeris=scenario.ephemeris
                    ).visible
                ),
            )
    return out
