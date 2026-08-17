"""The anchor scenario's comms model: the lunar polar relay + DSN contact plan.

The pinned comms scenario behind the flagship benchmark — *lunar polar water-ice prospecting*
(``scenarios/1-lunar-polar-ice-prospecting.md``): a swarm of surface agents in and around the
permanently shadowed regions of the Shackleton-de Gerlache ridge, one relay orbiter, and the three
NASA DSN complexes. It is the scenario that makes the benchmark **comms-denied for real** — a rover
on a PSR floor loses line of sight to the relay behind the crater rim *and* has no direct-to-Earth
path, so it reaches Earth only through a relay pass (``LUNAR-TR-003``; link.md §12 "P0 exit").

**What Link owns here, and what it does not.** This module owns the *scenario*: the node set, the
antennas, the DSN complexes, the epoch window, and the fidelity config — pinned data, hashed into a
:class:`~astro_mine.link.cache.CacheKey` (link.md §5). It does **not** own the geometry inputs.
Terrain occlusion arrives through the Core :class:`~astro_mine.core.world.WorldProvider` contract
(an injected [Worlds](worlds.md) provider — never an ``astro-mine-worlds`` import, conventions.md
§1.1) and orbits/frames arrive through :mod:`astro_mine.spice` (RFC-0002). Both are **required**:
:func:`anchor_scenario` takes them as arguments, and there is no default — a comms model must never
silently assume "connected" because a provider was missing (link.md §2, §9).

The resulting :class:`~astro_mine.core.messages.ContactPlan` is content-addressed
(:func:`~astro_mine.link.cache.plan_digest`) and published to Hub as a ``comms_model`` artifact via
:mod:`astro_mine.link.registry` — the digest a Bench ``ScenarioSpec`` pins.

Backlog: RM-P0-LINK-04 -- astro-mine-link#25
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from astro_mine.core.messages import ContactPlan
from astro_mine.core.sadf.enums import CommsBand, NodeRole
from astro_mine.core.sadf.model import Comms
from astro_mine.core.units import Epoch, EpochWindow, TimeScale
from astro_mine.core.world import Vector, WorldProvider
from astro_mine.link.budget import LinkBudget, compute_link_budget
from astro_mine.link.constellation import (
    ConstellationScenario,
    builtin_catalog,
    constellation_contact_windows,
    contact_nodes,
)
from astro_mine.link.geometry import EphemerisNode, EphemerisProvider, SurfaceNode
from astro_mine.link.products import build_contact_plan
from astro_mine.link.windows import GroundStation, TopocentricProvider
from astro_mine.spice import MOON_RADIUS_M

__all__ = [
    "ANCHOR_ARTIFACT_NAME",
    "ANCHOR_ARTIFACT_VERSION",
    "ANCHOR_EPOCH_WINDOW",
    "ANCHOR_REFINE_S",
    "ANCHOR_RELAY",
    "ANCHOR_RELAY_TARGET",
    "ANCHOR_SCENARIO_ID",
    "ANCHOR_STEP_S",
    "ANCHOR_SURFACE_SITES",
    "AnchorSurfaceSite",
    "anchor_budgets",
    "anchor_config",
    "anchor_earth_terminals",
    "anchor_ground_stations",
    "anchor_node_ids",
    "anchor_radios",
    "anchor_scenario",
    "anchor_surface_nodes",
    "build_anchor_contact_plan",
]

#: The Bench scenario this comms model belongs to (``bench/zoo/lunar_polar_ice_prospecting_v1``).
ANCHOR_SCENARIO_ID = "lunar-polar-ice-prospecting-v1"

#: The Hub artifact identity of the anchor comms model — the stable content **id** a Bench
#: ``ContentPins.link`` entry needs alongside the digest (a bare ``ContactPlan`` has no id field,
#: which is exactly why the anchor's ``link`` pin was ``null`` until now).
ANCHOR_ARTIFACT_NAME = "lunar-polar-relay-dsn"
#: **0.3.0** extends the pinned epoch window from 24 h to the full mission horizon the Bench anchor
#: runs (30 days, ``43_200 x 60 s``; see :data:`ANCHOR_EPOCH_WINDOW`). The 24 h window is a
#: well-chosen *representative* rotation, but Sim masks every epoch beyond the plan as no-contact,
#: so ``comms_robustness`` over the 30-day episode was silently diluted by the ~29 un-modelled days
#: -- the same shape as the two defects below, and the last one left (astro-mine-link#34).
#:
#: **0.2.0** renamed the swarm's contact-graph nodes to the Fleet SADF asset ids
#: (the SADF ``identity.id``) that 0.1.0 already *claimed* they were while emitting other names
#: (``prospecting-rover``). That mismatch was load-bearing: Sim binds a contact node to an agent by
#: exact id match, so the published plan's nodes intersected none of Sim's agents, no observation
#: was ever masked, and the anchor's ``comms_robustness`` scored *not applicable* for a reason no
#: one could see (astro-mine-sim#53).
#:
#: Digests in a registry are immutable, so each fix is a new version rather than a re-publish; Bench
#: re-pins the artifact. The convention, stated once: **a contact node that is a fleet asset is
#: named by that asset's SADF ``identity.id``.** Nodes that are not fleet assets -- the DSN ground
#: stations -- keep their catalogue ids, and bind to no agent, which is correct: they are not robots
#: in the swarm.
ANCHOR_ARTIFACT_VERSION = "0.3.0"

#: The pinned epoch window: the **30-day mission** the Bench anchor runs, from 2030-01-01T00:00:00
#: TDB (``43_200 x 60 s = 2_592_000 s``, matching the anchor episode's ``max_sim_seconds``). The
#: window spans the mission it is pinned into so ``comms_robustness`` scores connectivity over the
#: whole episode: Sim masks any epoch past the plan as no-contact, so a window shorter than the
#: episode silently reads un-modelled time as denied (astro-mine-link#34).
#:
#: The first 24 h is the *representative* sub-window the geometry was chosen around -- one Earth
#: rotation, so all three DSN complexes take their turn, and ~12 relay periods (~2.6 h each), so
#: every surface node sees several passes and several blackouts; the mission is ~30 such rotations.
#: Pinned as TDB seconds past J2000 so the *scenario* is describable without a leap-second kernel;
#: computing it still requires the kernels (see :func:`anchor_scenario`).
ANCHOR_EPOCH_WINDOW = EpochWindow(
    start=Epoch(tdb_seconds=946_728_000.0, scale=TimeScale.TDB),
    end=Epoch(tdb_seconds=949_320_000.0, scale=TimeScale.TDB),
)

#: The rise/set search grid: a 60 s coarse step bisected to 5 s. A lunar relay pass is tens of
#: minutes, so 60 s cannot skip one; 5 s is far inside the RM-P0-LINK-05 oracle budget (20 s).
ANCHOR_STEP_S = 60.0
ANCHOR_REFINE_S = 5.0

#: The notional relay orbiter's ephemeris target. It is **not** a NAIF-catalogued body: the anchor
#: pins a spacecraft SPK id, and whoever computes the plan furnishes the matching SPK — the same way
#: the RM-P0-LINK-05 oracle regression consumes a GMAT-exported SPK. Link never propagates an orbit
#: itself (link.md §2.2); it resolves this target through the injected ``EphemerisProvider``.
ANCHOR_RELAY_TARGET = "-90001"
ANCHOR_RELAY = EphemerisNode(name="relay-orbiter", target=ANCHOR_RELAY_TARGET)


@dataclass(frozen=True, slots=True)
class AnchorSurfaceSite:
    """A pinned surface agent of the anchor swarm: where it sits, and how high its antenna is.

    ``lat_deg``/``lon_deg`` are planetocentric selenographic coordinates in the world provider's
    body-fixed frame; ``elevation_m`` is the site's terrain height above the mean lunar radius and
    ``antenna_height_m`` the mast above it. ``psr`` records whether the site sits on a permanently
    shadowed floor — the nodes whose Earth link the crater rim is expected to deny.
    """

    name: str
    lat_deg: float
    lon_deg: float
    elevation_m: float
    antenna_height_m: float
    psr: bool
    description: str

    def position_m(self) -> Vector:
        """The body-fixed Cartesian position (metres) of the site's antenna phase centre."""
        radius = MOON_RADIUS_M + self.elevation_m + self.antenna_height_m
        lat, lon = math.radians(self.lat_deg), math.radians(self.lon_deg)
        return (
            radius * math.cos(lat) * math.cos(lon),
            radius * math.cos(lat) * math.sin(lon),
            radius * math.sin(lat),
        )


#: The anchor swarm's comms nodes. Ids **are** the Fleet SADF asset ids the Bench anchor pins
#: (the SADF ``identity.id``), so a Sim agent and a contact-plan node name the same robot -- see
#: the node-id vocabulary note on :data:`ANCHOR_ARTIFACT_VERSION` above. The two
#: PSR sites are on the Shackleton crater floor (deep shadow, no Earth line of sight); the two lit
#: sites are on the Shackleton-de Gerlache connecting ridge, where the ISRU plant and the haul
#: route live (scenarios/1 §3-§5).
ANCHOR_SURFACE_SITES: tuple[AnchorSurfaceSite, ...] = (
    AnchorSurfaceSite(
        name="prospecting-rover",
        lat_deg=-89.90,
        lon_deg=0.0,
        elevation_m=-3800.0,
        antenna_height_m=1.5,
        psr=True,
        description="Active-perception scout on the Shackleton PSR floor.",
    ),
    AnchorSurfaceSite(
        name="excavator",
        lat_deg=-89.86,
        lon_deg=90.0,
        elevation_m=-3500.0,
        antenna_height_m=1.5,
        psr=True,
        description="Granular excavation on the PSR floor.",
    ),
    AnchorSurfaceSite(
        name="hauler",
        lat_deg=-89.78,
        lon_deg=135.0,
        elevation_m=-800.0,
        antenna_height_m=1.5,
        psr=False,
        description="Haul route between the PSR floor and the ridge plant.",
    ),
    AnchorSurfaceSite(
        name="isru-plant",
        lat_deg=-89.68,
        lon_deg=204.0,
        elevation_m=1800.0,
        antenna_height_m=4.0,
        psr=False,
        description="Extraction/purification plant on the lit connecting ridge.",
    ),
)


def _radio_label(node_id: str) -> str:
    """A short, human-readable label for a node's radio (``excavator-s-band``).

    Cosmetic only. The *key* a radio is filed under is the node id -- the fully-qualified SADF asset
    id, which is what binds the contact graph to a Sim agent -- but a ``Comms.name`` is a display
    label that never reaches the ContactPlan, so it stays readable rather than repeating the
    namespace."""
    return node_id.rsplit(".", 1)[-1]


def anchor_radios() -> dict[str, Comms]:
    """The pinned SADF radios of the anchor node set, keyed by node id (link.md §3).

    The comms capability blocks Link reads from Fleet SADF (fleet.md §3). Each node carries the
    radio it uses on the **proximity** leg: a surface agent an S-band terminal, the relay a
    higher-EIRP S-band cross-link. A DSN complex carries its X-band aperture — its only radio.
    The space-side **Earth-return** terminals are the separate X-band set in
    :func:`anchor_earth_terminals` (a node's proximity and Earth links are different radios, and the
    parametric budget refuses a cross-band pair).

    Pinned here so the anchor comms model is buildable offline from Link alone — no
    ``astro-mine-fleet`` dependency (conventions.md §1.1); the values mirror the anchor's published
    SADF assets.
    """
    radios = {
        site.name: Comms(
            name=f"{_radio_label(site.name)}-s-band",
            band=CommsBand.S_BAND,
            node_role=NodeRole.SPACE,
            eirp_dbw=12.0,
            gt_db_per_k=-8.0,
            modcod_supported=_SURFACE_MODCODS,
        )
        for site in ANCHOR_SURFACE_SITES
    }
    radios[ANCHOR_RELAY.name] = Comms(
        name="relay-s-band",
        band=CommsBand.S_BAND,
        node_role=NodeRole.SPACE,
        eirp_dbw=28.0,
        gt_db_per_k=2.0,
        modcod_supported=_RELAY_MODCODS,
        relay=True,
    )
    radios.update(_dsn_radios())
    return radios


def anchor_earth_terminals() -> dict[str, Comms]:
    """The space-side **X-band Earth-return** terminals, keyed by node id (plus the DSN apertures).

    The direct-to-Earth leg is a different radio from the proximity leg: the relay carries a
    high-gain X-band Earth terminal, and a surface agent a small direct-to-Earth X-band terminal it
    can only close when it actually sees Earth — which, on a PSR floor, it never does. Keeping the
    two sets apart is what lets the parametric budget stay band-consistent (link.md §3, §11).
    """
    terminals = {
        site.name: Comms(
            name=f"{_radio_label(site.name)}-x-band",
            band=CommsBand.X_BAND,
            node_role=NodeRole.SPACE,
            eirp_dbw=18.0,
            gt_db_per_k=-2.0,
            modcod_supported=_SURFACE_MODCODS,
        )
        for site in ANCHOR_SURFACE_SITES
    }
    terminals[ANCHOR_RELAY.name] = Comms(
        name="relay-x-band",
        band=CommsBand.X_BAND,
        node_role=NodeRole.SPACE,
        eirp_dbw=42.0,
        gt_db_per_k=12.0,
        modcod_supported=_RELAY_MODCODS,
        relay=True,
    )
    terminals.update(_dsn_radios())
    return terminals


def _dsn_radios() -> dict[str, Comms]:
    """A DSN complex's 34 m X-band aperture — its only radio, so it is in both radio sets."""
    return {
        station.name: Comms(
            name=f"{station.name}-x-band",
            band=CommsBand.X_BAND,
            node_role=NodeRole.GROUND,
            eirp_dbw=97.0,
            gt_db_per_k=52.0,
            modcod_supported=_RELAY_MODCODS,
        )
        for station in anchor_ground_stations()
    }


_SURFACE_MODCODS = ["bpsk_r1_2", "qpsk_r1_2", "qpsk_r3_4"]
_RELAY_MODCODS = ["bpsk_r1_2", "qpsk_r1_2", "qpsk_r3_4", "8psk_r3_4"]


def anchor_ground_stations() -> tuple[GroundStation, ...]:
    """The anchor's Earth ground segment: the three NASA DSN complexes (RM-P0-LINK-02)."""
    return builtin_catalog("dsn").stations


def anchor_surface_nodes() -> tuple[SurfaceNode, ...]:
    """The anchor's surface agents, as body-fixed
    :class:`~astro_mine.link.geometry.SurfaceNode`\\ s."""
    return tuple(
        SurfaceNode(name=site.name, position_m=site.position_m()) for site in ANCHOR_SURFACE_SITES
    )


def anchor_node_ids() -> tuple[str, ...]:
    """Every node id of the anchor contact graph, in declaration order (surface, relay, ground)."""
    return (
        *(site.name for site in ANCHOR_SURFACE_SITES),
        ANCHOR_RELAY.name,
        *(station.name for station in anchor_ground_stations()),
    )


def anchor_config() -> dict[str, object]:
    """The fidelity/config block of the anchor comms scenario — the LINK-05 ``config`` cache input.

    Everything that changes the *computation* but is not a node, an epoch, or a file: the search
    grid, the relay target, the link-budget margin, and the pinned representative slant ranges the
    per-pass budgets are evaluated at.
    """
    return {
        "step_s": ANCHOR_STEP_S,
        "refine_s": ANCHOR_REFINE_S,
        "relay_target": ANCHOR_RELAY_TARGET,
        "margin_db": _MARGIN_DB,
        "representative_range_m": dict(_REPRESENTATIVE_RANGE_M),
        "link_surface_to_ground": True,
    }


def anchor_scenario(
    *,
    world: WorldProvider,
    ephemeris: EphemerisProvider,
    topocentric: TopocentricProvider,
    window: EpochWindow | None = None,
) -> ConstellationScenario:
    """The anchor's :class:`~astro_mine.link.constellation.ConstellationScenario`.

    ``world`` supplies terrain occlusion through the Core ``WorldProvider`` contract (the anchor's
    Shackleton-de Gerlache DEM/horizon bundle), ``ephemeris`` resolves the relay SPK and Earth, and
    ``topocentric`` gives DSN station elevation — all through :mod:`astro_mine.spice`. ``window``
    defaults to :data:`ANCHOR_EPOCH_WINDOW`.

    Every provider is required: passing a stub that always sees a target would silently turn the
    comms-denied benchmark into a fully connected one, so there is deliberately no default.
    """
    return ConstellationScenario(
        surface=anchor_surface_nodes(),
        relays=(ANCHOR_RELAY,),
        ground=anchor_ground_stations(),
        world=world,
        ephemeris=ephemeris,
        topocentric=topocentric,
        window=window or ANCHOR_EPOCH_WINDOW,
        step_s=ANCHOR_STEP_S,
        refine_s=ANCHOR_REFINE_S,
        body_radius_m=MOON_RADIUS_M,
        link_surface_to_ground=True,
    )


def build_anchor_contact_plan(scenario: ConstellationScenario) -> ContactPlan:
    """Compute the anchor's :class:`~astro_mine.core.messages.ContactPlan` from ``scenario``.

    Searches the contact windows across the whole {surface x relay x DSN} node set
    (:func:`~astro_mine.link.constellation.constellation_contact_windows`), annotates each node
    pair with its LINK-03 parametric link budget, and reduces both into the Core ContactPlan Sim
    consumes. Deterministic: same kernels + same terrain + same node set + same epoch window + same
    config ⇒ the identical plan, hence the identical :func:`~astro_mine.link.cache.plan_digest`
    (link.md §5; conventions.md §1.5).
    """
    windows = constellation_contact_windows(scenario)
    nodes = contact_nodes(scenario)
    budgets = anchor_budgets({node.id for node in nodes})
    return build_contact_plan(
        nodes,
        windows,
        budgets=budgets,
        epoch_window=scenario.window,
    )


def anchor_budgets(node_ids: Iterable[str]) -> dict[tuple[str, str], LinkBudget]:
    """The per-pair parametric link budgets of the anchor (RM-P0-LINK-03).

    One representative per-pass budget per ordered node pair — the Phase-0 approximation
    :func:`~astro_mine.link.products.build_contact_plan` documents (the dense per-tick
    latency/bandwidth cube is the P1 :func:`~astro_mine.link.products.emit_time_series` product).
    Each pair is evaluated at the pinned representative slant range for its pair class
    (surface↔surface across the polar basin, surface/relay↔relay at the relay's mean altitude, and
    the Moon-Earth leg), so the budget is a function of pinned scenario data alone and cannot drift
    with the geometry backend. An Earth-leg pair is budgeted with the X-band Earth terminals; every
    other pair with the S-band proximity radios (:func:`anchor_radios`).
    """
    proximity = anchor_radios()
    earth = anchor_earth_terminals()
    known = set(node_ids)
    ground = {station.name for station in anchor_ground_stations()}
    budgets: dict[tuple[str, str], LinkBudget] = {}
    for observer in sorted(known):
        for target in sorted(known):
            if observer == target:
                continue
            if observer in ground and target in ground:
                continue  # station-to-station is a terrestrial network, not a space link
            earth_leg = observer in ground or target in ground
            radios = earth if earth_leg else proximity
            budgets[observer, target] = compute_link_budget(
                radios[observer],
                radios[target],
                range_m=_representative_range_m(observer, target, ground),
                margin_db=_MARGIN_DB,
            )
    return budgets


#: The link margin reserved above each mod/cod's required Eb/N0 (link.md §11 parametric default).
_MARGIN_DB = 3.0

#: Representative slant ranges per pair class (metres): the polar-basin surface baseline, the
#: relay's mean slant range from the surface, and the mean Moon-Earth distance. Pinned scenario
#: data, so the per-pass budget is reproducible without re-deriving geometry.
_REPRESENTATIVE_RANGE_M: Mapping[str, float] = {
    "surface_surface": 30.0e3,
    "surface_relay": 2.5e6,
    "earth_link": 3.844e8,
}


def _representative_range_m(observer: str, target: str, ground: set[str]) -> float:
    """The representative slant range for the ``observer -> target`` pair class."""
    if observer in ground or target in ground:
        return _REPRESENTATIVE_RANGE_M["earth_link"]
    if ANCHOR_RELAY.name in (observer, target):
        return _REPRESENTATIVE_RANGE_M["surface_relay"]
    return _REPRESENTATIVE_RANGE_M["surface_surface"]
