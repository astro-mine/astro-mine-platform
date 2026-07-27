"""Bench ``ScenarioSpec`` → Sim ``Scenario`` — the content-pinned bridge (RM-P1-SIM-01).

A Bench :class:`~astro_mine.bench.scenario.ScenarioSpec` declares *what* to run and pins its inputs
**by content hash** — a world, a fleet, resource fields — but it deliberately carries no bytes:
Bench
composes, it is never a second simulator (bench.md §2.2). Resolving those pins into a runnable
:class:`~astro_mine.sim.runtime.Scenario` is Sim's job, and this module is where it happens.

The resolution goes through the RM-P1-SIM-01 :class:`~astro_mine.sim.runtime.ContentResolver`, not
through hand-authored fixtures: every fleet pin is pulled from the content store (supply-chain
verified, fail-closed) and read back as a Core :class:`~astro_mine.core.sadf.model.Asset`; the world
and prospect pins are reconstructed into Core providers by their producers' registered entry-point
factories. So the run is byte-addressed to the exact bundles the scenario pinned, and the resolved
``id -> sha256:`` map rides in the run's provenance.

**Each asset's engine is inferred from the asset itself**, not from a scenario literal: an asset
whose
SADF declares proximity-orbit mobility propagates orbitally, one with wheels drives, one with a
digging tool excavates. The dynamics *parameters* likewise come from the resolved content (Fleet
mass
and geometry; Worlds regolith and gravity) via the RM-P0-SIM-03 builders — so the scenario document
supplies none of the physics.

**Placement is the scenario's** (bench#63). A spec pins where each asset stands, and this module
honours it. It did not always: placement was derived from the scenario hash, which made siting a
function of the *content digest* — so a re-pin moved the swarm, and the physics contradicted the
pinned ``ContactPlan`` computed against a deliberate siting. The hash-derived ring survives only as
the fallback for an asset a scenario does not place.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from astro_mine.core.sadf.enums import ContactElementKind, Regime
from astro_mine.core.units import INERTIAL_J2000, J2000_EPOCH, Epoch
from astro_mine.sim.runtime.content import (
    ContentPin,
    ContentResolver,
    ResolvedAsset,
    ResolvedContent,
    ScenarioContent,
    UnresolvedProvider,
    agent_spec_from_asset,
    dem_granular_dynamics_from_content,
    granular_dynamics_from_content,
    mobility_dynamics_from_content,
    mujoco_dynamics_from_content,
)
from astro_mine.sim.runtime.scenario import (
    MOON_MU_M3_S2,
    MOON_RADIUS_M,
    AgentSpec,
    Dynamics,
    IsruSpec,
    KinematicDynamics,
    OrbitalDynamics,
    Scenario,
    Vec3Spec,
)
from astro_mine.sim.scheduler import FidelityPolicy

if TYPE_CHECKING:
    from astro_mine.bench.scenario import ResolvedScenario, ScoringSpec, SitePlacement
    from astro_mine.core.sadf.model import Asset
    from astro_mine.core.world import WorldProvider
    from astro_mine.sim.comms import ConnectivitySource
    from astro_mine.sim.runtime.content import BundleStore, ProviderFactory

__all__ = [
    "ResolvedRun",
    "dynamics_for_asset",
    "scenario_content_from_spec",
    "sim_scenario_from_spec",
]

#: The default lunar polar relay orbit (m) the anchor's orbiter is placed in when the spec pins no
#: initial state (it pins content, not placement — see the module docstring).
_RELAY_ORBIT_RADIUS_M = 1_837_400.0
#: Surface assets are laid out on a deterministic ring of this radius (m) around the site origin.
_SURFACE_LAYOUT_RADIUS_M = 25.0
#: The default per-asset battery capacity (J) when the pinned SADF declares no storage.
_DEFAULT_BATTERY_J = 5.0e6


class ResolvedRun:
    """Everything a Sim run needs, resolved from a Bench :class:`ResolvedScenario`.

    ``scenario`` is the runnable Sim scenario; ``world_provider`` / ``resource_field`` /
    ``connectivity`` are the live Core providers reconstructed from the pinned world / prospect /
    link bundles (``None`` when no producer factory is installed for their kind — the caller injects
    one then); ``content_hashes`` is the ``id -> sha256:`` map that rides in the run's provenance,
    so the trace is byte-addressed to the content it ran on.

    ``unresolved`` names the pins that resolved by digest but rebuilt no provider (#67) — the
    difference between "the caller will inject one" and "nobody will, and this run is blind"."""

    __slots__ = (
        "belief",
        "connectivity",
        "content_hashes",
        "resource_field",
        "scenario",
        "scoring",
        "unresolved",
        "world_provider",
    )

    def __init__(
        self,
        scenario: Scenario,
        *,
        world_provider: WorldProvider | None,
        resource_field: object | None,
        content_hashes: dict[str, str],
        connectivity: ConnectivitySource | None = None,
        unresolved: tuple[UnresolvedProvider, ...] = (),
        scoring: ScoringSpec | None = None,
        belief: object | None = None,
    ) -> None:
        self.scenario = scenario
        self.world_provider = world_provider
        self.resource_field = resource_field
        self.content_hashes = content_hashes
        self.connectivity = connectivity
        self.unresolved = unresolved
        #: The scenario's pinned scoring parameters (bench#63), carried through so the scorer can
        #: prefer them over its own defaults. ``None`` when the spec pins none.
        self.scoring = scoring
        #: The conditionable belief the prospect pin rebuilt (#66), or ``None`` when no producer
        #: registered a ``prior_recipe`` factory. The belief-quality metrics need it; nothing else
        #: does, so a run without one is degraded, not broken.
        self.belief = belief


def scenario_content_from_spec(resolved: ResolvedScenario) -> ScenarioContent:
    """Map a Bench spec's :class:`ContentPins` onto Sim's :class:`ScenarioContent`.

    The two are the same *shape* by design but are separately owned — Sim cannot import Bench's
    models into its runtime, and Bench never imports Sim (conventions.md §1.1). Each pin's
    ``content_hash`` is the reference a :class:`BundleStore` resolves, so the run is
    content-addressed
    rather than tag-addressed."""
    pins = resolved.spec.content
    return ScenarioContent(
        world=ContentPin(id=pins.world.id, reference=pins.world.content_hash),
        fleet=tuple(ContentPin(id=ref.id, reference=ref.content_hash) for ref in pins.fleet),
        prospect=tuple(ContentPin(id=ref.id, reference=ref.content_hash) for ref in pins.prospect),
        link=(
            ContentPin(id=pins.link.id, reference=pins.link.content_hash)
            if pins.link is not None
            else None
        ),
    )


def _has_contact(asset: Asset, kind: ContactElementKind) -> bool:
    return asset.mobility is not None and any(
        element.kind is kind for element in asset.mobility.contact
    )


def _regimes(asset: Asset) -> set[Regime]:
    return set(asset.mobility.regimes) if asset.mobility is not None else set()


def dynamics_for_asset(
    resolved: ResolvedAsset,
    *,
    world: WorldProvider | None = None,
    position: Vec3Spec = (0.0, 0.0, 0.0),
    contact_tier: bool = False,
    dem_tier: bool = False,
) -> Dynamics:
    """The regime engine this **asset** routes to, inferred from its own SADF declaration.

    Not from a scenario literal: an asset that declares proximity-orbit mobility propagates
    orbitally;
    one with a digging **tool** excavates; one with **wheels** drives; anything else is the
    reference
    kinematic engine. That is the "routing an engine is *configuration*" principle applied at the
    content boundary — the pinned fleet decides which physics it needs.

    The dynamics *parameters* come from the resolved content too (Fleet mass/geometry, Worlds
    regolith/gravity) via the RM-P0-SIM-03 builders, so nothing here is hand-authored.

    ``contact_tier`` selects the articulated MuJoCo wheel-soil tier for a wheeled asset instead of
    the
    reduced-order kinematic one — the fidelity dial, exposed as a flag rather than baked in.

    ``dem_tier`` is the same dial for an *excavator*: it selects the high-fidelity DEM granular
    block instead of the reduced-order one. Without it a TOOL-bearing asset could only ever route to
    ``granular``, so the DEM/surrogate ladder — and therefore the surrogate speedup measurement
    (surrogate.md §8) — was unreachable from a Bench spec (#51)."""
    asset = resolved.asset
    if Regime.PROXIMITY_ORBIT in _regimes(asset):
        # An orbiter. Worlds' contract exposes surface gravity at a point, not the body's GM, so the
        # central-body mu stays the documented reduced-order constant (a Worlds gravity-model
        # extension, not a Sim placeholder).
        return OrbitalDynamics(mu_m3_s2=MOON_MU_M3_S2)
    if _has_contact(asset, ContactElementKind.TOOL):
        # An excavator: a digging tool is what makes it one.
        if dem_tier:
            return dem_granular_dynamics_from_content(resolved, world=world, position=position)
        return granular_dynamics_from_content(resolved, world=world, position=position)
    if _has_contact(asset, ContactElementKind.WHEEL):
        if contact_tier:
            return mujoco_dynamics_from_content(resolved, world=world, position=position)
        return mobility_dynamics_from_content(resolved, world=world, position=position)
    return KinematicDynamics()


def _isru_for_asset(asset: Asset) -> IsruSpec | None:
    """The reduced-order ISRU block for an asset that declares a plant (RM-P1-SIM-02).

    The plant's **throughput** is the asset's (SADF ``payload.isru.throughput_kg_hr``), so the
    extraction rate the run's stored-water metric depends on comes from the pinned content, not from
    a
    scenario literal. The tank's **capacity** likewise (``payload.capacity_kg``)."""
    payload = asset.payload
    if payload is None or payload.isru is None:
        return None
    throughput = payload.isru.throughput_kg_hr
    rate_kg_s = throughput / 3600.0 if throughput else IsruSpec().extraction_rate_kg_s
    return IsruSpec(extraction_rate_kg_s=rate_kg_s, capacity_kg=payload.capacity_kg)


def _body_fixed(lat_rad: float, lon_rad: float, radius_m: float) -> Vec3Spec:
    """Planetocentric spherical -> body-fixed Cartesian metres."""
    return (
        radius_m * math.cos(lat_rad) * math.cos(lon_rad),
        radius_m * math.cos(lat_rad) * math.sin(lon_rad),
        radius_m * math.sin(lat_rad),
    )


def _pinned_site(resolved: ResolvedScenario, asset_id: str) -> SitePlacement | None:
    """The scenario's pinned site for ``asset_id``, if it pins one."""
    placement = resolved.spec.placement
    if placement is None:
        return None
    return next((site for site in placement.sites if site.asset == asset_id), None)


def _site_position(site: SitePlacement, world: WorldProvider | None) -> Vec3Spec:
    """The body-fixed position of a **pinned** site (bench#63).

    The site's ``elevation_m`` is the terrain height above the body's mean radius, so the asset's
    body origin sits at ``MOON_RADIUS_M + elevation``. Deliberately *not* the same arithmetic as
    ``astro_mine.link.anchor.AnchorSurfaceSite.position_m()``, which adds ``antenna_height_m``
    because it wants an antenna phase centre: a rover's body origin is not 1.5 m above the ground,
    and an ISRU plant's is not 4 m. The mast belongs to the asset's own SADF, not to its placement.

    An omitted ``elevation_m`` means *"take it from the pinned terrain"* — the DEM is already pinned
    by hash, so sampling it is as reproducible as restating the number and cannot drift from the
    world the run actually uses. As in the ring layout, ``sample()`` is called with no epoch:
    terrain height is geometry, not illumination.
    """
    lat, lon = math.radians(site.lat_deg), math.radians(site.lon_deg)
    elevation = site.elevation_m
    if elevation is None:
        if world is None:
            elevation = 0.0
        else:
            reference = _body_fixed(lat, lon, MOON_RADIUS_M)
            elevation = float(world.sample(reference).elevation_m)
    return _body_fixed(lat, lon, MOON_RADIUS_M + elevation)


def _layout(
    resolved: ResolvedScenario,
    index: int,
    total: int,
    orbital: bool,
    world: WorldProvider | None = None,
    *,
    asset_id: str | None = None,
) -> Vec3Spec:
    """A deterministic initial position for the asset at ``index``, **body-fixed**.

    **A pinned site wins.** Since bench#63 a ``ScenarioSpec`` can pin placement, and when it does
    the scenario decides where the asset stands — see :func:`_site_position`. Everything below is
    the fallback for an asset the scenario does not place.

    A ``ScenarioSpec`` pins *content*, and historically not *placement* — where an asset started was
    a property of the run, not of the bundle. So the fallback layout is derived from the scenario
    hash: reproducible across machines and clean checkouts (which is what the determinism gate
    needs), but not a hand-authored literal either. Orbiters go to the relay orbit; unplaced surface
    assets are spread on a ring.

    **Why the fallback is not a neutral default.** ``scenario_hash`` derives from ``spec_hash``, so
    a re-pin changes the jitter and *moves the swarm* — taking ``water_mass`` (through the
    resource-field sample under the plant), ``discovery_latency`` and every illumination-dependent
    metric with it, for reasons having nothing to do with the policy. It also contradicts a pinned
    ``ContactPlan``, which [Link](link.md) computes against a deliberate siting: the anchor's comms
    geometry was computed with the plant on a lit ridge at -89.68° while the physics ran it on a
    25 m ring at the pole. That is what pinning placement fixes.

    **The ring is on the body's surface, not around its centre.** That sentence is the fix. Core's
    ``WorldProvider`` resolves queried positions in the **body-fixed** frame (``world/protocol.py``:
    "the body-fixed reference frame queried positions and rays resolve in"), ``AgentSpec`` says the
    same of ``initial_position_m``, and ``Scenario.frame`` defaults to ``MOON_BODY_FIXED``. Orbiters
    already honoured that — ``_RELAY_ORBIT_RADIUS_M`` is a radius **from the body centre**. Surface
    assets did not: they were laid out on a 25 m ring about the **origin**, which in a body-fixed
    frame is 25 m from the Moon's centre, on the equator (``z = 0``).

    Worlds read that literally, as it should. The point projected far outside the south-polar grid,
    so every query fell through to the out-of-grid default — and because ``sample()`` is
    deliberately *total* ("out-of-grid positions … return a well-formed default point rather than
    raising"), nothing raised. What came back instead was:

    * ``gravity`` evaluated 25 m from the body **centre** — ``3.3e24 m/s^2``, wrong by ~24 orders of
      magnitude;
    * ``regolith`` of ``None`` on every field, so Sim silently fell back to its reduced-order
      defaults and the **pinned world's soil never reached the physics** (friction 31 deg, not the
      world's 40 deg);
    * ``elevation`` of 0.

    The reduced-order engines integrate no particle bed, so this stayed invisible. The DEM granular
    tier does: at 3.3e24 m/s^2 the 1200-substep settle explodes, the bed is **NaN before the first
    step**, and both fidelity tiers then produce confident garbage — which is how the surrogate
    speedup measurement came back with a NaN realized error (astro-mine-bench#31).

    So the ring now sits on the surface: the same deterministic ``(x, y)``, at the body-fixed
    radius, snapped to the terrain. Snapped, not merely set to the reference radius, because an
    asset left 1.3 km beneath the ridge it stands on is *below* the terrain — and ``line_of_sight``
    would then be occluded by the ground it is buried in, silently zeroing comms.

    ``sample()`` is called with no epoch, which needs no ephemeris — the terrain height is geometry,
    not illumination.
    """
    site = None if asset_id is None else _pinned_site(resolved, asset_id)
    if site is not None:
        if orbital:
            # A body-fixed lat/lon cannot describe an orbit, and silently ignoring the pin would
            # put the orbiter somewhere the scenario did not ask for while claiming it was placed.
            raise ValueError(
                f"scenario {resolved.scenario_id!r} pins a surface site for {asset_id!r}, but that "
                "asset declares the proximity-orbit regime — a body-fixed site cannot place an "
                "orbiter. Remove the site, or pin an asset that operates on the surface."
            )
        return _site_position(site, world)
    if orbital:
        return (_RELAY_ORBIT_RADIUS_M, 0.0, 0.0)
    digest = hashlib.sha256(f"{resolved.scenario_hash}:{index}".encode()).digest()
    jitter = int.from_bytes(digest[:4], "big") / float(1 << 32)  # deterministic, in [0, 1)
    angle = 2.0 * math.pi * ((index + jitter) / max(total, 1))
    x = _SURFACE_LAYOUT_RADIUS_M * math.cos(angle)
    y = _SURFACE_LAYOUT_RADIUS_M * math.sin(angle)

    # The site is the body-fixed south pole — where the anchor world grid is centred, and the
    # same lunar geometry `_RELAY_ORBIT_RADIUS_M` / `MOON_MU_M3_S2` assume. A scenario wanting a
    # different site has to pin placement, which `ScenarioSpec` does not yet express.
    reference = (x, y, -MOON_RADIUS_M)
    if world is None:
        return reference
    elevation = float(world.sample(reference).elevation_m)
    return (x, y, -(MOON_RADIUS_M + elevation))


def _initial_velocity(orbital: bool) -> Vec3Spec:
    if not orbital:
        return (0.0, 0.0, 0.0)
    return (0.0, math.sqrt(MOON_MU_M3_S2 / _RELAY_ORBIT_RADIUS_M), 0.0)


def _battery_j(asset: Asset) -> float:
    """The asset's usable battery capacity (J) — the sum of its SADF power-storage banks.

    The starting state of charge is the asset's own, so the night-survival margin a run is scored on
    comes from the pinned fleet rather than a scenario literal."""
    if asset.power is None or not asset.power.storage:
        return _DEFAULT_BATTERY_J
    total = sum(bank.capacity_j for bank in asset.power.storage)
    return total if total > 0.0 else _DEFAULT_BATTERY_J


#: How many epochs across the episode the plan-applicability probe samples. The check only has to
#: answer "does this plan ever say anything here?", so a coarse sweep is enough — and it is setup,
#: run once per episode build, not per tick.
_PLAN_PROBE_SAMPLES = 64


def _require_the_plan_applies(
    connectivity: ConnectivitySource | None,
    *,
    start_epoch: Epoch | None,
    dt_s: float,
    horizon_steps: int,
    agent_ids: Sequence[str],
    scenario_id: str | None,
) -> None:
    """Refuse a pinned ContactPlan that is never in contact anywhere in the episode's time span.

    A ContactPlan is a plan **over a window of time**, so whether it applies at all depends on
    *when* the episode runs. Nothing checked that, and the failure was silent in the worst way: the
    Bench ``EpisodeSpec`` carried no start epoch, so this seam left :class:`Scenario` on its default
    ``J2000_EPOCH`` (TDB 0.0) while the anchor's plan covers 24 h at TDB 946 728 000 (2030-01-01).
    Thirty years apart. Every interval was inactive at every tick, ``earth_contact`` was false
    forever, and ``comms_robustness`` scored a confident **0.0** — a number with the shape of a
    result and none of the content (astro-mine-bench#48).

    This is the temporal twin of the node-id guard in ``runtime/episode.py``: there, a plan whose
    nodes name no agent masks nothing and the metric reads *not applicable*; here, a plan whose
    window does not reach the episode masks *everything* and the metric reads a confident zero. Both
    are vocabulary errors — one in the id namespace, one on the clock — and neither is ever a
    legitimate scenario. A caller who wants an unmasked run pins no ``link`` ref at all.

    The probe uses only the :class:`ConnectivitySource` protocol (``nodes`` + ``comms_mask``): the
    protocol deliberately does not expose the plan's window, and Sim must not reach into Link's
    internals to get it.
    """
    if connectivity is None or start_epoch is None:
        return
    bound = [a for a in agent_ids if a in set(connectivity.nodes)]
    if not bound:
        return  # the node-id guard in runtime/episode.py owns this failure, and states it better.

    span_s = dt_s * horizon_steps
    steps = max(1, min(_PLAN_PROBE_SAMPLES, horizon_steps))
    for index in range(steps + 1):
        epoch = Epoch(
            tdb_seconds=start_epoch.tdb_seconds + span_s * index / steps,
            scale=start_epoch.scale,
        )
        for agent_id in bound:
            mask = connectivity.comms_mask(agent_id, epoch)
            # `links` is populated for every peer whether or not it is up — each `PeerLink` carries
            # its own `reachable` flag — so presence proves nothing and reachability is the test.
            # Either kind of contact counts: a peer link (rover to relay) or a gateway one (relay to
            # DSN). Only a plan silent on both, for every bound agent, across the whole span, is the
            # pathology this guards.
            if mask.earth_contact or any(link.reachable for link in mask.links):
                return

    end_s = start_epoch.tdb_seconds + span_s
    raise ValueError(
        "the pinned ContactPlan is never in contact at any point in this episode, so every "
        "observation would be masked as comms-denied and `comms_robustness` would score a "
        "confident 0.0 rather than failing.\n"
        f"  scenario:      {scenario_id}\n"
        f"  episode epochs: TDB {start_epoch.tdb_seconds:.0f} .. {end_s:.0f} "
        f"({horizon_steps} steps x {dt_s:g}s)\n"
        f"  bound agents:   {sorted(bound)}\n"
        "The plan's contact windows lie outside this span. Declare the scenario's "
        "`episode.start_epoch` to an instant the plan actually covers, or pin no `link` ref for an "
        "unmasked run."
    )


def sim_scenario_from_spec(
    resolved: ResolvedScenario,
    *,
    store: BundleStore,
    seed: int,
    provider_factories: dict[str, ProviderFactory] | None = None,
    verify: bool = True,
    horizon_steps: int | None = None,
    dt_s: float | None = None,
    contact_tier: bool = False,
    dem_tier: bool = False,
    tool_speed_mps: float | None = None,
    fidelity: FidelityPolicy | None = None,
) -> ResolvedRun:
    """Resolve a Bench :class:`ResolvedScenario` into a runnable Sim :class:`ResolvedRun`.

    Pulls every pinned bundle through the RM-P1-SIM-01 :class:`ContentResolver` (supply-chain
    verified
    fail-closed), materializes one agent per fleet asset — its sensors, power/thermal budgets, and
    fidelity profiles from the SADF; its engine and dynamics inferred from the asset and the pinned
    world — and stamps the resolved content hashes for the run's provenance.

    ``horizon_steps`` caps the episode below the spec's horizon (the anchor's is 43 200 ticks — a
    full
    lunar month at a 60 s tick, which is a *benchmark* run, not a test run). ``contact_tier`` routes
    wheeled assets to the articulated MuJoCo tier. The Bench episode's ``dt_s`` is derived from the
    spec (``max_sim_seconds / horizon_steps``), so the Sim clock matches the benchmark's cadence.

    ``dem_tier`` routes excavators to the high-fidelity DEM granular block, and ``fidelity`` is the
    :class:`~astro_mine.sim.scheduler.FidelityPolicy` the run's scheduler admits tiers under — a
    pinned tier, or an ``error_budget`` a surrogate must stay inside to be substituted
    (``LUNAR-TR-002``). Together they are the path a Bench-driven run needs to select DEM *or*
    surrogate physics; before this the Scenario was always built with the default policy, so a
    Bench-driven run could never leave the coarsest tier (#51).

    ``dt_s`` overrides the spec-derived tick, and the **granular tiers require it**. A Bench episode
    declares a *mission* cadence — the anchor's is 60 s — but a DEM bed integrates *contact* at its
    stable internal timestep (~0.8 ms), so it sub-steps ``dt_s / dt_internal_s`` times per tick: at
    a 60 s tick that is ~78 000 sub-steps **per step per agent**, which is not a slow run but an
    unrunnable one. A granular tier is therefore driven at a contact-scale tick (the DEM suite uses
    0.05 s) while the reduced-order tiers keep the mission cadence. Sim will not silently pick one
    for the caller: the dial is explicit."""
    content = scenario_content_from_spec(resolved)
    resolver = ContentResolver(store, provider_factories=provider_factories, verify=verify)
    materialized: ResolvedContent = resolver.resolve(content)

    world = materialized.world_provider
    episode = resolved.spec.episode
    horizon = episode.horizon_steps if horizon_steps is None else horizon_steps
    tick_s = (
        (
            episode.max_sim_seconds / episode.horizon_steps
            if episode.max_sim_seconds is not None
            else 1.0
        )
        if dt_s is None
        else dt_s
    )

    pins = resolved.spec.content
    total = len(pins.fleet)
    agents: list[AgentSpec] = []
    for index, ref in enumerate(pins.fleet):
        asset = materialized.assets[ref.id]
        orbital = Regime.PROXIMITY_ORBIT in _regimes(asset.asset)
        position = _layout(resolved, index, total, orbital, world, asset_id=ref.id)
        dynamics = dynamics_for_asset(
            asset,
            world=world,
            position=position,
            contact_tier=contact_tier,
            dem_tier=dem_tier,
        )
        if tool_speed_mps is not None and dynamics.kind == "dem_granular":
            dynamics = dynamics.model_copy(update={"tool_speed_mps": tool_speed_mps})
        spec = agent_spec_from_asset(
            asset,
            agent_id=ref.id,
            dynamics=dynamics,
            initial_position_m=position,
            velocity_mps=_initial_velocity(orbital),
            battery_soc_j=_battery_j(asset.asset),
            frame=INERTIAL_J2000 if orbital else None,
        )
        # The ISRU block is Sim-owned (no SADF field carries the reduced-order process model), but
        # its
        # throughput is the asset's — so a plant's extraction rate is pinned content too.
        isru = _isru_for_asset(asset.asset)
        agents.append(spec if isru is None else spec.model_copy(update={"isru": isru}))

    start_epoch = episode.start_epoch
    _require_the_plan_applies(
        materialized.connectivity,
        start_epoch=start_epoch,
        dt_s=tick_s,
        horizon_steps=horizon,
        agent_ids=[agent.agent_id for agent in agents],
        scenario_id=resolved.scenario_id,
    )

    scenario = Scenario(
        name=resolved.scenario_id,
        agents=tuple(agents),
        seed=seed,
        dt_s=tick_s,
        horizon_steps=horizon,
        fidelity=FidelityPolicy() if fidelity is None else fidelity,
        # The task decides *when* it runs. A spec that declares no epoch keeps Sim's own default —
        # the pre-existing behaviour, and right for a scenario that pins nothing time-dependent.
        start_epoch=J2000_EPOCH if start_epoch is None else start_epoch,
    )
    return ResolvedRun(
        scenario,
        world_provider=world,
        resource_field=materialized.resource_field,
        content_hashes=materialized.content_hashes,
        connectivity=materialized.connectivity,
        unresolved=materialized.unresolved,
        scoring=resolved.spec.scoring,
        belief=materialized.belief,
    )


def materialize_bench_run(scenario_id: str, *, store: BundleStore, seed: int) -> ResolvedRun:
    """Load a Bench ``ScenarioSpec`` by id, resolve its pins, materialize a runnable Sim episode.

    The CLI ``run`` bridge (RM-P0-SIM-11): a thin convenience over :func:`sim_scenario_from_spec`
    that owns the **Bench-side** load + resolve (``astro_mine.bench.zoo``). It lives here, in the
    adapter package, so the Sim CLI reaches Bench *only* through ``astro_mine.sim.bench`` — the base
    runtime never imports Bench, and the one-way direction holds (conventions.md §1.1; bench.md
    §2.2). Raises :class:`KeyError` if ``scenario_id`` is not in the zoo; needs ``astro-mine-bench``
    installed (the ``[bench]`` extra) to import.
    """
    from astro_mine.bench.zoo import load_scenario, resolve_scenario

    resolved = resolve_scenario(load_scenario(scenario_id))
    return sim_scenario_from_spec(resolved, store=store, seed=seed)
