"""Fixtures for the real-Sim anchor integration (issue #14).

Everything here is *real*: `astro_mine.sim`'s own `Simulator` and stepping core, a Core
`ContactPlan` driving Sim's own `ReferenceConnectivitySampler`, and — for the Bench path — content
published to a real local Hub registry and pinned by hash in a Bench `ScenarioSpec`. Nothing is
stubbed; the toy env does not appear.

Two scenario builders, because the two claims need different things:

- :func:`swarm_scenario` builds a Sim `Scenario` directly. That is how a ≥20-agent swarm is
  reachable at all: a Bench scenario's agents come one-per-pinned-fleet-asset, so 24 agents through
  that path would mean publishing 24 distinct bundles to say nothing new. Real physics, real comms,
  no content store.
- :func:`anchor_content` / :func:`anchor_spec` publish a small pinned fleet and build the
  ScenarioSpec that `bench.baseline.run` scores through `SimEpisodeRunner` — the content-addressed
  path, which is the thing worth proving for the scoring claim.

**Power and thermal budgets are not decoration.** Guard's `DefaultSignalResolver` reads
`battery_soc_j` and `temperature_k` off the observation's `StateSample`, and Sim only publishes
them for agents that declare a `PowerBudget`/`ThermalBudget`. An agent without them observes
`temperature_k = None`, the safety signal resolves to NaN, and the TCB fails the tick **closed** —
so the shield replaces every command with a zero-effort hold. The swarm then still emits one action
per agent per tick and would sail straight through a naive "it didn't collapse" assertion while
being, in fact, frozen. Hence :func:`agent_displacements` and the assertions that use it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from astro_mine.bench.scenario import (
    ContentPins,
    ContentRef,
    EpisodeSpec,
    MetricRef,
    ScenarioSpec,
    SeedSet,
)
from astro_mine.core.messages.enums import NodeRole
from astro_mine.core.messages.model import ContactInterval, ContactNode, ContactPlan, Observation
from astro_mine.core.registry import PluginKind, PluginManifest
from astro_mine.core.sadf.enums import ContactElementKind, Regime, SensorKind
from astro_mine.core.sadf.model import (
    Actuator,
    Asset,
    Body,
    ContactElement,
    Identity,
    Inertia,
    Mobility,
    ObservationModel,
    PowerBudget,
    PowerStorage,
    Range,
    ResourceTarget,
    SadfDocument,
    Sensor,
    ThermalBudget,
    Vec3,
)
from astro_mine.hub.client import HubClient
from astro_mine.hub.registry import Blob, Registry
from astro_mine.hub.supply_chain import generate_keypair
from astro_mine.mind.compose import compose
from astro_mine.mind.compose.graph import HierarchyGraph
from astro_mine.mind.reference import load_stack_resource
from astro_mine.mind.registry import TierRegistry
from astro_mine.sim.comms import ReferenceConnectivitySampler
from astro_mine.sim.runtime import AgentSpec, Scenario
from astro_mine.sim.runtime._hub_adapter import HubBundleStore
from astro_mine.sim.sensors import ReferenceResourceField

__all__ = [
    "ANCHOR_STACK",
    "DT_S",
    "GROUND_STATION",
    "SEED",
    "agent_displacements",
    "anchor_content",
    "anchor_graph",
    "anchor_spec",
    "connectivity",
    "contact_plan",
    "keep_out_breaches",
    "swarm_scenario",
]

#: The stack under test: the three-tier hierarchy with the REAL Guard shield and REAL Allocate
#: planner bound by entry point (see the stack spec for why its params are what they are).
ANCHOR_STACK = "lunar_prospecting_anchor.yaml"
SEED = 7
#: The anchor's decision tick. The stack's SafetySpec is compiled at this sample period.
DT_S = 60.0
GROUND_STATION = "dsn-goldstone"

#: The lander keep-out the stack's SafetySpec authors: a 30 m sphere at the origin with a 3 m
#: margin. Agents start on a ring well outside it, so a breach means the shield failed, not that
#: the scenario was rigged.
_KEEP_OUT_RADIUS_M = 30.0
_KEEP_OUT_MARGIN_M = 3.0
_RING_RADIUS_M = 50.0

_SADF_JSON = "application/vnd.astro-mine.sadf+json"
_WORLD_TAR = "application/vnd.astro-mine.world.bundle.v1.tar"
_FIELD_TAR = "application/vnd.astro-mine.resource-field.bundle.v1.tar"
_HYDROGEN = "water_equivalent_hydrogen"


def anchor_graph(*, seed: int = SEED, registry: TierRegistry | None = None) -> HierarchyGraph:
    """Compose the anchor stack. Raises if the [sim] extra is not installed — the sibling plugins
    are discovered from the entry-point group the sibling *packages* register."""
    return compose(
        load_stack_resource(ANCHOR_STACK),
        registry or TierRegistry.from_entry_points(),
        seed=seed,
    )


# --- the swarm scenario (real Sim, no content store) ------------------------------------


def _ring(index: int, total: int, radius: float = _RING_RADIUS_M) -> tuple[float, float, float]:
    """Spread agents evenly on a ring outside the lander keep-out."""
    angle = 2.0 * math.pi * index / total
    return (radius * math.cos(angle), radius * math.sin(angle), 0.0)


def _power() -> PowerBudget:
    return PowerBudget(storage=[PowerStorage(name="bat", capacity_j=5.0e6)], floor_w=5.0)


def _thermal() -> ThermalBudget:
    return ThermalBudget(
        operating_range_k=Range(min=250.0, max=320.0),
        survival_range_k=Range(min=100.0, max=350.0),
    )


def swarm_scenario(
    n_agents: int, *, horizon_steps: int, dt_s: float = DT_S, seed: int = SEED
) -> Scenario:
    """A Sim scenario with ``n_agents`` prospecting rovers on a ring outside the keep-out.

    Each agent declares a power and thermal budget, so Sim evolves and *publishes*
    ``battery_soc_j`` / ``temperature_k`` — the two signals the Guard shield's SafetySpec resolves.
    Without them the shield fails closed on every tick (see the module docstring).
    """
    return Scenario(
        name="mind-anchor-swarm",
        agents=tuple(
            AgentSpec(
                agent_id=agent_id,
                initial_position_m=_ring(index, n_agents),
                battery_soc_j=5.0e6,
                battery_floor_j=1.0e3,
                initial_temperature_k=250.0,
                power=_power(),
                thermal=_thermal(),
            )
            for index, agent_id in enumerate(agent_ids(n_agents))
        ),
        dt_s=dt_s,
        horizon_steps=horizon_steps,
        seed=seed,
    )


def agent_ids(n_agents: int) -> tuple[str, ...]:
    return tuple(f"rover-{i:02d}" for i in range(n_agents))


# --- the comms model (a real Core ContactPlan through Sim's own sampler) -----------------


def contact_plan(
    nodes: Sequence[str], *, blackout_ticks: Sequence[int], horizon_steps: int, dt_s: float = DT_S
) -> ContactPlan:
    """A ContactPlan giving every agent an Earth link EXCEPT across ``blackout_ticks``.

    A blackout is the *absence* of an open contact window — the same way a real PSR interval or a
    closed relay window denies Earth contact. Sim's sampler derives ``earth_contact`` from the graph
    (true iff some reachable peer is a GROUND node), so Mind sees denial through Core's
    ``CommsObservationMask``, exactly as it would from a Link-produced plan.
    """
    dark = set(blackout_ticks)
    end = (horizon_steps + 1) * dt_s
    intervals: list[ContactInterval] = []
    for node in nodes:
        for lo, hi in _lit_spans(dark, horizon_steps):
            intervals.append(
                ContactInterval(
                    node_a=node,
                    node_b=GROUND_STATION,
                    start_tdb_s=lo * dt_s,
                    # The final span runs past the horizon so the last tick is unambiguously lit.
                    end_tdb_s=end if hi > horizon_steps else hi * dt_s,
                    max_rate_bps=2.0e6,
                    min_latency_s=1.3,
                )
            )
    return ContactPlan(
        nodes=[
            ContactNode(id=GROUND_STATION, role=NodeRole.GROUND),
            *(ContactNode(id=node, role=NodeRole.SPACE) for node in nodes),
        ],
        intervals=intervals,
    )


def _lit_spans(dark: set[int], horizon_steps: int) -> list[tuple[int, int]]:
    """The maximal ``[lo, hi)`` tick spans that are NOT dark."""
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for tick in range(horizon_steps + 2):
        if tick in dark:
            if start is not None:
                spans.append((start, tick))
                start = None
        elif start is None:
            start = tick
    if start is not None:
        spans.append((start, horizon_steps + 2))
    return spans


def connectivity(
    nodes: Sequence[str], *, blackout_ticks: Sequence[int], horizon_steps: int, dt_s: float = DT_S
) -> ReferenceConnectivitySampler:
    """Sim's own ConnectivitySource over :func:`contact_plan` — the real masking path."""
    return ReferenceConnectivitySampler(
        contact_plan(nodes, blackout_ticks=blackout_ticks, horizon_steps=horizon_steps, dt_s=dt_s)
    )


# --- assertions that a run actually happened --------------------------------------------


def agent_displacements(
    scenario: Scenario, final_observations: Mapping[str, Observation]
) -> dict[str, float]:
    """How far each agent actually moved, in metres.

    The anti-theatre check. A shield that cannot certify a command substitutes a zero-effort hold,
    which still *emits an action for every agent on every tick* — so action counts alone cannot
    tell a working swarm from a frozen one. Displacement can.
    """
    start = {a.agent_id: a.initial_position_m for a in scenario.agents}
    moved: dict[str, float] = {}
    for agent_id, observation in final_observations.items():
        x0, y0, z0 = start[agent_id]
        p = observation.self_state.pose.translation_m
        moved[agent_id] = math.dist((x0, y0, z0), (p.x, p.y, p.z))
    return moved


def keep_out_breaches(final_observations: Mapping[str, Observation]) -> list[str]:
    """Agents inside the SafetySpec's lander keep-out sphere (a hard-constraint violation)."""
    limit = _KEEP_OUT_RADIUS_M + _KEEP_OUT_MARGIN_M
    breached = []
    for agent_id, observation in final_observations.items():
        p = observation.self_state.pose.translation_m
        if math.dist((0.0, 0.0, 0.0), (p.x, p.y, p.z)) < limit:
            breached.append(agent_id)
    return breached


# --- the pinned content the Bench scoring path resolves ----------------------------------


def _rover_asset(agent_id: str) -> SadfDocument:
    """A prospecting rover: wheels so it drives, a spectrometer so it discovers, and the power +
    thermal budgets whose telemetry the Guard shield's safety signals resolve from."""
    asset = Asset(
        identity=Identity(id=agent_id, name="Scout", version="0.1.0", kind="rover"),
        root_frame="body",
        bodies=[
            Body(
                name="chassis",
                frame="body",
                mass_kg=210.0,
                center_of_mass_m=Vec3(x=0.0, y=0.0, z=0.0),
                inertia_kg_m2=Inertia(ixx=1.0, iyy=1.0, izz=1.0),
            )
        ],
        actuators=[Actuator(name="drive", velocity=0.6, torque_nm=45.0)],
        mobility=Mobility(
            regimes=[Regime.SURFACE],
            contact=[
                ContactElement(
                    kind=ContactElementKind.WHEEL, dimensions_m=Vec3(x=0.1, y=0.1, z=0.28)
                )
            ],
        ),
        sensors=[
            Sensor(
                name="neutron",
                kind=SensorKind.NEUTRON_SPECTROMETER,
                frame="body",
                observation_model=ObservationModel(noise_sigma=0.005),
                resource=ResourceTarget(species=_HYDROGEN, si_unit="mass_fraction"),
            )
        ],
        power=_power(),
        thermal=_thermal(),
    )
    return SadfDocument(sadf_version="0.1", asset=asset)


class _AnchorWorld:
    """A Core WorldProvider for the pinned world: lunar gravity and regolith terramechanics."""

    @property
    def frame(self) -> Any:
        from astro_mine.core.units import MOON_BODY_FIXED

        return MOON_BODY_FIXED

    def sample(self, position: tuple[float, float, float], *, epoch: Any | None = None) -> Any:
        from astro_mine.core.units import MOON_BODY_FIXED
        from astro_mine.core.world import (
            Illumination,
            IlluminationState,
            RegolithParams,
            SurfacePoint,
        )

        return SurfacePoint(
            frame=MOON_BODY_FIXED,
            elevation_m=0.0,
            surface_normal=(0.0, 0.0, 1.0),
            gravity=(0.0, 0.0, -1.62),
            illumination=Illumination(state=IlluminationState.LIT, solar_flux_w_m2=1361.0),
            temperature_k=250.0,
            regolith=RegolithParams(
                bulk_density_kg_m3=1600.0,
                friction_angle_deg=35.0,
                bearing_capacity_pa=5.0e4,
            ),
        )


def _world_factory(manifest: PluginManifest, layers: Mapping[str, bytes]) -> _AnchorWorld:
    return _AnchorWorld()


def _field_factory(manifest: PluginManifest, layers: Mapping[str, bytes]) -> ReferenceResourceField:
    return ReferenceResourceField(species=_HYDROGEN, peak=0.12, length_scale_m=30.0)


def _publish(
    client: HubClient,
    key: bytes,
    *,
    name: str,
    artifact_kind: str,
    manifest_kind: PluginKind,
    core_interface: str,
    layers: list[Blob],
) -> str:
    manifest = PluginManifest(
        name=name,
        version="0.1.0",
        kind=manifest_kind,
        core_interfaces={core_interface: "0.1.0"},
        license="Apache-2.0",
    )
    artifact = client.publish(
        name=name,
        version="0.1.0",
        kind=artifact_kind,
        manifest=manifest,
        layers=layers,
        private_key_pem=key,
    )
    return str(artifact.digest)


def anchor_content(registry_dir: Path, *, n_agents: int) -> dict[str, Any]:
    """Publish the fleet/world/prospect bundles to a real local Hub registry.

    Returns the ``store`` a :class:`SimEpisodeRunner` resolves pins from, the ``digests`` a
    ScenarioSpec pins by, and the provider ``factories`` that reconstruct world/field.
    """
    private_key, public_key = generate_keypair()
    client = HubClient(Registry(registry_dir), trusted_public_key_pem=public_key)
    store = HubBundleStore(client)

    digests: dict[str, str] = {}
    for agent_id in agent_ids(n_agents):
        document = _rover_asset(agent_id)
        digests[agent_id] = _publish(
            client,
            private_key,
            name=agent_id,
            artifact_kind="asset",
            manifest_kind=PluginKind.ASSET,
            core_interface="sadf",
            layers=[Blob(_SADF_JSON, document.model_dump_json().encode("utf-8"))],
        )
    digests["shackleton-de-gerlache"] = _publish(
        client,
        private_key,
        name="shackleton-de-gerlache",
        artifact_kind="world",
        manifest_kind=PluginKind.WORLD_PROVIDER,
        core_interface="world_provider",
        layers=[Blob(_WORLD_TAR, b"world-bundle-tar-bytes")],
    )
    digests["shackleton-water-ice"] = _publish(
        client,
        private_key,
        name="shackleton-water-ice",
        artifact_kind="plugin",
        manifest_kind=PluginKind.RESOURCE_FIELD_BACKEND,
        core_interface="resource_field",
        layers=[Blob(_FIELD_TAR, b"field-bundle-tar-bytes")],
    )
    return {
        "store": store,
        "digests": digests,
        "factories": {
            PluginKind.WORLD_PROVIDER.value: _world_factory,
            PluginKind.RESOURCE_FIELD_BACKEND.value: _field_factory,
        },
    }


def anchor_spec(
    content: dict[str, Any], *, n_agents: int, horizon_steps: int, seeds: tuple[int, ...] = (1001,)
) -> ScenarioSpec:
    """A Bench ScenarioSpec pinning that published content by hash — the anchor's shape.

    Sim derives one agent per pinned fleet asset, and names it by the pin's id — which is why the
    ContactPlan for a Bench-scored run must use these same ids.
    """
    digests = content["digests"]
    return ScenarioSpec(
        scenario_id="mind-anchor-lunar-polar-ice-prospecting",
        name="Mind anchor stack — lunar polar water-ice prospecting",
        core_interface={"env": "0.1.0", "messages": "0.1.0"},
        content=ContentPins(
            world=ContentRef(
                id="shackleton-de-gerlache",
                content_hash=digests["shackleton-de-gerlache"],
            ),
            fleet=tuple(
                ContentRef(id=agent_id, content_hash=digests[agent_id])
                for agent_id in agent_ids(n_agents)
            ),
            prospect=(
                ContentRef(
                    id="shackleton-water-ice",
                    content_hash=digests["shackleton-water-ice"],
                ),
            ),
        ),
        seeds=SeedSet(public=seeds),
        episode=EpisodeSpec(horizon_steps=horizon_steps, max_sim_seconds=horizon_steps * DT_S),
        metrics=(
            MetricRef(name="water_mass"),
            MetricRef(name="energy_per_kg"),
            MetricRef(name="nights_survived"),
            MetricRef(name="comms_robustness"),
        ),
    )
