# SPDX-License-Identifier: Apache-2.0
"""Content-pinned Scenario construction (RM-P1-SIM-01).

Where :mod:`~astro_mine.sim.runtime.scenario` loads a :class:`Scenario` from an inline document —
its dynamics params, sensor suites, and power/thermal budgets typed straight into the file — this
module materializes those *fleet-sourced* fields from **content pinned by hash** and published to
Hub, so a run reproduces byte-for-byte from resolved content rather than hand-authored literals
(sim.md §3, §5; the follow-on to RM-P0-SIM-11).

The narrow waist is the point (conventions.md §1.1). Sim resolves content **only** from a
:class:`BundleStore` — the ``astro-mine-hub`` client satisfies it structurally — by **Core-typed
content-hash references**, and reads it back as **Core** artifacts: a fleet asset is a Core
:class:`~astro_mine.core.sadf.model.SadfDocument`, a world/resource provider is reconstructed by a
**producer-registered entry-point factory** (group ``astro_mine.providers``) that Sim *discovers*
but never imports. So Sim imports **only Core + the Hub client** — never ``astro_mine.worlds`` /
``fleet`` / ``prospect`` / ``bench``.

The mapping from a Bench ``ScenarioSpec``'s pins to the :class:`ScenarioContent` below is
Bench/orchestrator-side (Sim cannot import ``astro_mine.bench``); Sim's input is the resolved
:class:`ContentPin` set. Resolution is deterministic and content-addressed: two clean checkouts
resolving the same pins produce the identical run.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Protocol, cast, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from astro_mine.core.registry import PluginKind, PluginManifest
from astro_mine.core.sadf.enums import ContactElementKind
from astro_mine.core.sadf.model import Asset, ContactElement, SadfDocument
from astro_mine.sim.engines._rover_mjcf import DEFAULT_WHEEL_RADIUS_M
from astro_mine.sim.runtime.scenario import (
    LUNAR_GRAVITY_M_S2,
    LUNAR_REGOLITH_DENSITY_KG_M3,
    AgentSpec,
    DemGranularDynamics,
    Dynamics,
    GranularDynamics,
    KinematicDynamics,
    MjxContactDynamics,
    MobilityDynamics,
    MujocoMobilityDynamics,
    Vec3Spec,
)

if TYPE_CHECKING:
    from astro_mine.core.resource import ResourceField
    from astro_mine.core.sadf.model import FidelityProfile, PowerBudget, Sensor, ThermalBudget
    from astro_mine.core.units import Epoch, ReferenceFrame
    from astro_mine.core.world import WorldProvider
    from astro_mine.sim.comms import ConnectivitySource

__all__ = [
    "PROVIDER_ENTRY_POINT_GROUP",
    "BundleStore",
    "ContentPin",
    "ContentResolver",
    "ProviderFactory",
    "ResolvedAsset",
    "ResolvedContent",
    "ScenarioContent",
    "SiteConditions",
    "UnresolvedProvider",
    "agent_spec_from_asset",
    "asset_drive_limits",
    "asset_mass_kg",
    "asset_tool_geometry",
    "asset_wheel_radius_m",
    "dem_granular_dynamics_from_content",
    "describe_unresolved",
    "granular_dynamics_from_content",
    "mjx_dynamics_from_content",
    "mobility_dynamics_from_content",
    "mujoco_dynamics_from_content",
    "site_conditions",
]

# Reduced-order fallbacks for a dynamics field the pinned content does not declare. They are the
# same
# lunar-anchor values the hand-authored scenario blocks carried, so an un-pinned scenario is
# byte-identical to before; the point of the builders below is that a *pinned* scenario no longer
# needs them.
_DEFAULT_ROVER_MASS_KG = 250.0
_DEFAULT_MAX_SPEED_MPS = 1.0
_DEFAULT_WHEEL_TORQUE_NM = 40.0
_DEFAULT_FRICTION_ANGLE_DEG = 31.0
_DEFAULT_BEARING_CAPACITY_PA = 4.0e4
_DEFAULT_TOOL_WIDTH_M = 0.3
_DEFAULT_TOOL_HEIGHT_M = 0.10
#: The DEM reference bed is sized a few tool-widths across, so the blade has soil either side of it.
_BED_WIDTH_PER_TOOL = 2.0
_MIN_BED_WIDTH_M = 0.6

#: The entry-point group a content producer registers its ``from_bundle`` factory under, keyed by
#: the Core :class:`~astro_mine.core.registry.PluginKind` value (e.g. ``world_provider``). Sim
#: discovers these via :func:`importlib.metadata.entry_points` — it never imports the producer.
PROVIDER_ENTRY_POINT_GROUP = "astro_mine.providers"

#: A producer's reconstruction factory: ``(manifest, layers) -> provider``, where ``layers`` maps an
#: OCI layer's media type to its raw bytes. Returns a live Core provider (a ``WorldProvider`` or a
#: ``ResourceField``). It MUST import only Core (+ its own package) — never ``astro_mine.sim``.
ProviderFactory = Callable[[PluginManifest, Mapping[str, bytes]], object]


class _Pin(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ContentPin(_Pin):
    """A single content reference: a stable ``id`` pinned to a Hub ``reference``.

    ``reference`` is what a :class:`BundleStore` resolves — a bare ``sha256:<hex>`` digest (the
    reproducible form) or a ``name:version`` tag. ``id`` is the stable content identity (a Fleet
    ``identity.id``, a Worlds ``world_id``, a Prospect recipe name) carried into provenance.
    """

    id: str = Field(min_length=1)
    reference: str = Field(min_length=1)


class ScenarioContent(_Pin):
    """The content a scenario pins by hash: a world, its fleet, and optional fields / comms.

    Mirrors the *shape* of a Bench ``ContentPins`` but is Sim-owned (Sim cannot import
    ``astro_mine.bench``). ``fleet`` assets materialize into agents; ``world`` / ``prospect`` /
    ``link`` reconstruct into injected providers when a factory is registered for their kind.
    """

    world: ContentPin | None = None
    fleet: tuple[ContentPin, ...] = ()
    prospect: tuple[ContentPin, ...] = ()
    link: ContentPin | None = None

    @field_validator("fleet")
    @classmethod
    def _unique_ids(cls, fleet: tuple[ContentPin, ...]) -> tuple[ContentPin, ...]:
        ids = [pin.id for pin in fleet]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate fleet content ids: {dupes}")
        return fleet


@runtime_checkable
class BundleStore(Protocol):
    """The read surface Sim needs from a content registry — the ``astro-mine-hub`` client
    (:class:`astro_mine.hub.client.HubClient` + :class:`~astro_mine.hub.registry.Registry`)
    satisfies it structurally, so Sim's core never hard-imports Hub (see
    :mod:`astro_mine.sim.runtime._hub_adapter`)."""

    def resolve_digest(self, reference: str) -> str:
        """Resolve a digest or ``name:version`` reference to its ``sha256:<hex>`` digest."""
        ...

    def pull_manifest(self, reference: str, *, verify: bool = True) -> bytes:
        """Fetch the artifact's config blob — the Core :class:`PluginManifest` JSON — re-verifying
        the supply chain (signature/SLSA/SBOM) unless ``verify`` is ``False`` (fail closed)."""
        ...

    def pull_layers(self, reference: str, *, verify: bool = True) -> dict[str, bytes]:
        """Fetch the artifact's payload layers as ``media_type -> bytes`` — **content-verified**.

        The layers are the ones the *verified* manifest commits to, and each one's bytes are
        re-hashed against its content address before they are returned; a mismatch fails closed
        (hub.md §2.3; conventions.md §9). That per-layer check is **unconditional** — ``verify``
        toggles only the *supply-chain* re-check (signature/SLSA/SBOM), exactly as on
        :meth:`pull_manifest` — so no unverified byte reaches Sim even under ``verify=False``.
        """
        ...


@dataclass(frozen=True, slots=True)
class ResolvedAsset:
    """A fleet asset resolved from Hub: its Core :class:`~astro_mine.core.sadf.model.Asset` plus
    the ``sha256:`` digest it was pinned by (for provenance)."""

    asset: Asset
    content_hash: str


#: ``PluginKind`` value → the producer package whose entry point rebuilds that provider, and what a
#: run loses without it. Sim never imports these packages; it names them so a blind run is
#: diagnosable instead of merely wrong (#67).
_PRODUCER_FOR_KIND: dict[str, tuple[str, str]] = {
    "world_provider": (
        "astro-mine-worlds",
        "no terrain, gravity or illumination — night windows cannot be measured, so "
        "`nights_survived` scores not-applicable",
    ),
    "resource_field_backend": (
        "astro-mine-prospect",
        "no sealed resource field — prospecting sensors render `valid=False`, so "
        "`discovery_latency` never trips and ISRU extraction sees no abundance",
    ),
    "observation_model": (
        "astro-mine-prospect",
        "no producer-supplied sensor likelihood — the reduced-order default is used instead",
    ),
    "comms_model": (
        "astro-mine-link",
        "no contact plan — every observation is unmasked, so `comms_robustness` scores "
        "not-applicable",
    ),
}


@dataclass(frozen=True, slots=True)
class UnresolvedProvider:
    """A pin that resolved **by digest** but rebuilt no live provider (#67).

    The resolver leaves a provider ``None`` when no factory is registered for its kind — a
    deliberate seam, because a caller may inject its own. What was missing is the ability to tell
    *"the caller will inject"* apart from *"nobody will, and this run is blind"*. This is that
    record: the resolver reports what it could not rebuild and lets the caller set policy.
    """

    content_id: str
    kind: str
    #: The producer package that supplies this kind's entry point, e.g. ``astro-mine-worlds``.
    producer: str
    #: What the run loses without it, in the operator's terms (which metric stops scoring).
    consequence: str

    def describe(self) -> str:
        """A single actionable line — the pin, the missing producer, and what it costs."""
        return (
            f"  - {self.content_id!r} ({self.kind}): install {self.producer} — without it, "
            f"{self.consequence}"
        )


def _unresolved(pin: ContentPin, kind: PluginKind) -> UnresolvedProvider:
    producer, consequence = _PRODUCER_FOR_KIND.get(
        kind.value, ("its producer package", "the run proceeds without that input")
    )
    return UnresolvedProvider(
        content_id=pin.id, kind=kind.value, producer=producer, consequence=consequence
    )


def describe_unresolved(unresolved: Sequence[UnresolvedProvider]) -> str:
    """The multi-line operator-facing diagnostic for a blind run (#67).

    Written to be pasted into an issue: it names every pin that resolved by digest but rebuilt
    nothing, the package that supplies it, and the metric that stops scoring as a result. Content
    and code are fetched separately — ``astro-mine bench fetch`` obtains the bundles, but rebuilding
    a world bundle into a ``WorldProvider`` is ``astro-mine-worlds``' job — so a user who followed
    the documented quickstart can have every digest and still no physics.
    """
    lines = [
        f"{len(unresolved)} pinned input(s) resolved by digest but rebuilt no provider, so this "
        "run is blind to them:",
        *[item.describe() for item in unresolved],
        "Content and code ship separately: `astro-mine bench fetch` obtains the bundles; the "
        "producer packages above rebuild them into live providers.",
    ]
    return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class ResolvedContent:
    """A resolved :class:`ScenarioContent`: fleet assets, the reconstructed world/resource/comms
    providers (``None`` when no factory is registered for their kind — the caller injects one
    then), and the ``id -> sha256:`` map of every pinned input that rides in the run provenance.

    ``connectivity`` is the pinned contact plan rebuilt into a live
    :class:`~astro_mine.sim.comms.ConnectivitySource` (RM-P0-SIM-08). Its nodes bind to agents by
    **exact id match**: a contact node that is a fleet asset carries that asset's SADF
    ``identity.id``, which is the same string Sim uses as ``agent_id`` (see
    :class:`~astro_mine.sim.runtime.episode.Simulator`)."""

    assets: dict[str, ResolvedAsset]
    world_provider: WorldProvider | None
    resource_field: ResourceField | None
    content_hashes: dict[str, str]
    connectivity: ConnectivitySource | None = None
    #: The conditionable belief rebuilt from the same prospect pin as ``resource_field`` (#66) —
    #: the sealed field is what sensors *sample*, this is what a posterior is inferred over. Typed
    #: ``object`` here because the shape lives in :mod:`astro_mine.sim.bench._belief`, which is
    #: behind the ``[bench]`` extra; the scoring path narrows it.
    belief: object | None = None
    #: Pins that resolved by digest but rebuilt no provider (#67). Empty on a fully-resolved run.
    #: The resolver records; the caller decides — the CLI prints it, the Bench runner refuses on it,
    #: and the library tier proceeds, because injection is a documented pattern there.
    unresolved: tuple[UnresolvedProvider, ...] = ()


def _discover_factories() -> dict[str, ProviderFactory]:
    """The producer-registered provider factories, keyed by Core plugin-kind value.

    Discovered from installed distributions via the ``astro_mine.providers`` entry-point group — a
    producer (Worlds, Prospect) self-registers, so Sim reconstructs a live provider without
    importing it. Absent producers simply contribute nothing (the provider stays ``None``)."""
    factories: dict[str, ProviderFactory] = {}
    for entry in entry_points(group=PROVIDER_ENTRY_POINT_GROUP):
        factories[entry.name] = cast(ProviderFactory, entry.load())
    return factories


class ContentResolver:
    """Materialize a :class:`ScenarioContent` into a :class:`ResolvedContent` from a
    :class:`BundleStore`.

    Every pin is pulled and (by default) supply-chain-verified fail-closed; a fleet pin is read back
    as a Core :class:`~astro_mine.core.sadf.model.SadfDocument`, a world/prospect pin is
    reconstructed by the entry-point factory registered for its :class:`PluginKind`. Results are
    cached by resolved digest, so re-resolving the same content is free and deterministic.

    ``provider_factories`` overrides entry-point discovery — pass an explicit ``{kind: factory}``
    map to inject providers (hermetic tests, or an orchestrator that binds its own), leaving Sim's
    resolution path identical.
    """

    def __init__(
        self,
        store: BundleStore,
        *,
        provider_factories: Mapping[str, ProviderFactory] | None = None,
        verify: bool = True,
    ) -> None:
        self._store = store
        self._factories: dict[str, ProviderFactory] = (
            dict(provider_factories) if provider_factories is not None else _discover_factories()
        )
        self._verify = verify
        self._asset_cache: dict[str, ResolvedAsset] = {}
        self._provider_cache: dict[tuple[str, str], object] = {}

    def resolve(self, content: ScenarioContent) -> ResolvedContent:
        """Resolve every pin to Core artifacts + a live provider set + the content-hash map."""
        content_hashes: dict[str, str] = {}
        unresolved: list[UnresolvedProvider] = []

        assets: dict[str, ResolvedAsset] = {}
        for pin in content.fleet:
            resolved = self._resolve_asset(pin)
            assets[pin.id] = resolved
            content_hashes[pin.id] = resolved.content_hash

        world_provider: WorldProvider | None = None
        if content.world is not None:
            provider, digest = self._resolve_provider(content.world, PluginKind.WORLD_PROVIDER)
            content_hashes[content.world.id] = digest
            world_provider = cast("WorldProvider | None", provider)
            if provider is None:
                unresolved.append(_unresolved(content.world, PluginKind.WORLD_PROVIDER))

        resource_field: ResourceField | None = None
        belief: object | None = None
        for pin in content.prospect:
            provider, digest = self._resolve_provider(pin, PluginKind.RESOURCE_FIELD_BACKEND)
            content_hashes[pin.id] = digest
            if provider is None:
                unresolved.append(_unresolved(pin, PluginKind.RESOURCE_FIELD_BACKEND))
            # A scenario resolves a single active resource field (the first pinned); additional
            # prospect pins still ride in provenance so the run stays content-addressed to them.
            if resource_field is None and provider is not None:
                resource_field = cast("ResourceField", provider)
            # The same pin also reconstructs the **conditionable belief** the belief-quality
            # metrics need (#66) — a different contract from the sealed field above, from the same
            # bundle. Absence is not reported as unresolved: a scenario that pins no scoring
            # parameters scores those metrics not-applicable either way, and a producer that
            # predates the `prior_recipe` factory is not a broken run.
            if belief is None:
                candidate, _ = self._resolve_provider(pin, PluginKind.PRIOR_RECIPE)
                if candidate is not None:
                    belief = candidate

        connectivity: ConnectivitySource | None = None
        if content.link is not None:
            # The pinned comms model (RM-P0-SIM-08), resolved exactly like the world and prospect
            # pins: the producer's entry-point factory (Link's ``comms_model``) rebuilds a live
            # ConnectivitySource from the bundle's layers, so Sim reconstructs a ContactPlan
            # without importing Link (conventions.md §1.1). Before this it resolved to a digest
            # only — the plan was pinned, hashed into provenance, and then never used, so the
            # anchor's comms model was dead weight and comms_robustness was unscorable (#53).
            provider, digest = self._resolve_provider(content.link, PluginKind.COMMS_MODEL)
            content_hashes[content.link.id] = digest
            connectivity = cast("ConnectivitySource | None", provider)
            if provider is None:
                unresolved.append(_unresolved(content.link, PluginKind.COMMS_MODEL))

        return ResolvedContent(
            assets=assets,
            world_provider=world_provider,
            resource_field=resource_field,
            connectivity=connectivity,
            belief=belief,
            content_hashes=dict(sorted(content_hashes.items())),
            unresolved=tuple(unresolved),
        )

    def _resolve_asset(self, pin: ContentPin) -> ResolvedAsset:
        digest = self._store.resolve_digest(pin.reference)
        cached = self._asset_cache.get(digest)
        if cached is not None:
            return cached
        # Verify the supply chain, then read the SADF document out of the payload layers. The asset
        # is a Core artifact (SadfDocument) — decoding needs no Fleet import (conventions.md §1.1).
        self._store.pull_manifest(pin.reference, verify=self._verify)
        layers = self._store.pull_layers(pin.reference, verify=self._verify)
        asset = _decode_asset(layers, pin.id)
        resolved = ResolvedAsset(asset=asset, content_hash=digest)
        self._asset_cache[digest] = resolved
        return resolved

    def _resolve_provider(self, pin: ContentPin, kind: PluginKind) -> tuple[object, str]:
        digest = self._store.resolve_digest(pin.reference)
        # Keyed by (digest, kind), not digest alone: one pin can legitimately reconstruct more than
        # one kind of provider. A prospect bundle yields both the queryable sealed field
        # (`resource_field_backend`) and the conditionable belief (`prior_recipe`, #66), and under
        # a digest-only key the second lookup would hand back the first provider — the same object,
        # silently, for a different contract.
        cache_key = (digest, kind.value)
        if cache_key in self._provider_cache:
            return self._provider_cache[cache_key], digest
        manifest = PluginManifest.model_validate_json(
            self._store.pull_manifest(pin.reference, verify=self._verify)
        )
        factory = self._factories.get(kind.value)
        if factory is None:
            # No producer factory installed for this kind: resolve the reference (so the hash rides
            # in provenance) but leave the live provider for the caller to inject.
            return None, digest
        layers = self._store.pull_layers(pin.reference, verify=self._verify)
        provider = factory(manifest, layers)
        self._provider_cache[cache_key] = provider
        return provider, digest


def _decode_asset(layers: Mapping[str, bytes], content_id: str) -> Asset:
    """Recover the Core :class:`Asset` from a fleet bundle's layers.

    The asset travels as a canonical SADF **JSON** layer; we identify it by decoding rather than by
    a pinned media-type string, so Sim stays decoupled from Fleet's exact layer vocabulary — any
    layer that validates as a :class:`SadfDocument` is the asset."""
    for data in layers.values():
        try:
            document = SadfDocument.model_validate_json(data)
        except (ValidationError, ValueError):
            continue
        return document.asset
    raise ValueError(f"no SADF document layer in fleet bundle {content_id!r}")


#: The material an excavator digs and a hauler carries. SADF payload slots name what they accept;
#: a slot accepting this is a cargo bin for the purposes of the value chain (#64).
BULK_REGOLITH = "regolith"


def cargo_capacity_kg(asset: Asset) -> float | None:
    """The asset's total regolith-carrying capacity (kg), or ``None`` if it carries none.

    Read from the SADF payload slots the asset actually declares — Fleet's hauler has authored a
    ``cargo_bin`` accepting ``regolith`` since its first revision, and nothing has ever consumed it.
    A slot that accepts regolith but declares no ``max_mass_kg`` is treated as unbounded, which the
    transfer rule then only limits by rate.
    """
    if asset.payload is None:
        return None
    slots = [s for s in asset.payload.slots if BULK_REGOLITH in s.accepts]
    if not slots:
        return None
    if any(s.max_mass_kg is None for s in slots):
        return None
    return float(sum(s.max_mass_kg or 0.0 for s in slots))


def is_digger(asset: Asset) -> bool:
    """Whether this asset excavates — it declares a ``TOOL`` contact element.

    The same test :func:`~astro_mine.sim.bench._scenario.dynamics_for_asset` routes on when it
    sends an asset to a granular engine, so "digs" means exactly "gets a digging engine" (#64).
    """
    return asset.mobility is not None and any(
        element.kind is ContactElementKind.TOOL for element in asset.mobility.contact
    )


def agent_spec_from_asset(
    resolved: ResolvedAsset,
    *,
    agent_id: str,
    dynamics: Dynamics | None = None,
    initial_position_m: Vec3Spec = (0.0, 0.0, 0.0),
    velocity_mps: Vec3Spec = (0.0, 0.0, 0.0),
    mode: str = "idle",
    frame: ReferenceFrame | None = None,
    battery_soc_j: float = 0.0,
    battery_floor_j: float = 0.0,
    initial_temperature_k: float | None = None,
) -> AgentSpec:
    """Build an :class:`AgentSpec` whose *fleet-sourced* fields come from a resolved SADF asset.

    The asset authoritatively supplies the Core-typed fields Sim previously carried as inline
    placeholders (RM-P0-SIM-06/07, RM-P0-FLEET-04/05): the **sensor suite**, the **power/thermal
    budgets**, and the **multi-fidelity profiles**. The engine ``dynamics`` (reduced-order params
    SADF does not carry) and the scenario **placement** (pose, velocity, mode, frame, battery state)
    remain caller-authored — that is where a scenario says *how many* and *where*, which is not a
    property of the pinned content. ``dynamics`` defaults to the reference kinematic engine.
    """
    asset = resolved.asset
    sensors: tuple[Sensor, ...] = tuple(asset.sensors)
    power: PowerBudget | None = asset.power
    thermal: ThermalBudget | None = asset.thermal
    fidelity_profiles: tuple[FidelityProfile, ...] = tuple(asset.fidelity_profiles)
    return AgentSpec(
        agent_id=agent_id,
        initial_position_m=initial_position_m,
        velocity_mps=velocity_mps,
        battery_soc_j=battery_soc_j,
        battery_floor_j=battery_floor_j,
        mode=mode,
        frame=frame,
        dynamics=dynamics if dynamics is not None else KinematicDynamics(),
        fidelity_profiles=fidelity_profiles,
        sensors=sensors,
        power=power,
        thermal=thermal,
        initial_temperature_k=initial_temperature_k,
        # What the asset can carry along the value chain, from the payload slots it declares (#64).
        # Fleet's hauler has authored a `cargo_bin` accepting regolith since its first revision.
        cargo_capacity_kg=cargo_capacity_kg(asset),
    )


# --- dynamics sourced from resolved content (RM-P0-SIM-03) ----------------------------------
# Where :func:`agent_spec_from_asset` materializes the *Core-typed* fleet fields (sensors, budgets,
# fidelity profiles), the builders below close the last gap RM-P0-SIM-03 left open: the **engine
# dynamics parameters** — mass, wheel/tool geometry, regolith terramechanics, gravity — which used
# to
# be hand-authored placeholders on the scenario ("plumbed later"). They are now read from the
# content
# the scenario actually pins: the **asset** side from the resolved Core
# :class:`~astro_mine.core.sadf.model.Asset` (Fleet), the **world** side from the resolved Core
# :class:`~astro_mine.core.world.WorldProvider` (Worlds) sampled at the agent's site.
#
# Sim still imports only Core (conventions.md §1.1): a SADF asset and a WorldProvider are Core
# types,
# so nothing here reaches into ``astro_mine.fleet`` or ``astro_mine.worlds``. Every field falls back
# to its documented reduced-order default when the pinned content does not declare it — a world that
# models no regolith, or an asset with no wheels, degrades to the Phase-0 constant rather than
# failing, and the fallback is always visible in the returned block.


def asset_mass_kg(asset: Asset, *, default: float) -> float:
    """The asset's total mass — the sum of its SADF rigid-body masses (Fleet's authoritative value).

    Falls back to ``default`` for an asset that declares no bodies (a placeholder SADF)."""
    if not asset.bodies:
        return default
    return sum(body.mass_kg for body in asset.bodies)


def _contact_element(asset: Asset, kind: ContactElementKind) -> ContactElement | None:
    """The asset's first declared ground-contact element of ``kind`` (a wheel, a digging tool)."""
    if asset.mobility is None:
        return None
    for element in asset.mobility.contact:
        if element.kind is kind:
            return element
    return None


def asset_wheel_radius_m(asset: Asset, *, default: float) -> float:
    """The wheel radius from the asset's SADF ``wheel`` contact element (``dimensions_m.z`` is the
    wheel's radius by the SADF geometry convention); ``default`` when the asset declares no
    wheel."""
    wheel = _contact_element(asset, ContactElementKind.WHEEL)
    if wheel is None or wheel.dimensions_m is None:
        return default
    return wheel.dimensions_m.z or default


def asset_drive_limits(
    asset: Asset, *, default_speed_mps: float, default_torque_nm: float
) -> tuple[float, float]:
    """The asset's ``(max_speed_mps, wheel_torque_nm)`` from its SADF drive actuators.

    Fleet declares the authoritative actuator limits Guard enforces (fleet.md §9), so the rover's
    top
    speed and wheel torque come from the asset rather than the scenario. Missing limits fall
    back."""
    speeds = [a.velocity for a in asset.actuators if a.velocity is not None]
    torques = [a.torque_nm for a in asset.actuators if a.torque_nm is not None]
    return (
        max(speeds) if speeds else default_speed_mps,
        max(torques) if torques else default_torque_nm,
    )


def asset_tool_geometry(
    asset: Asset, *, default_width_m: float, default_height_m: float
) -> tuple[float, float]:
    """The excavator tool's ``(width_m, height_m)`` from the asset's SADF ``tool`` contact element.

    The DEM granular tier's blade geometry — previously a hand-authored scenario placeholder — is a
    property of the *excavator*, so it belongs to the pinned Fleet asset."""
    tool = _contact_element(asset, ContactElementKind.TOOL)
    if tool is None or tool.dimensions_m is None:
        return default_width_m, default_height_m
    return (tool.dimensions_m.x or default_width_m, tool.dimensions_m.z or default_height_m)


@dataclass(frozen=True, slots=True)
class SiteConditions:
    """The world's terramechanics + gravity at one site — the Worlds side of a dynamics block.

    Resolved by sampling the pinned :class:`~astro_mine.core.world.WorldProvider` at the agent's
    position. Every field carries the world's value when it models one, else the documented
    reduced-order lunar default, so a world that models only a subset still yields a usable
    block."""

    regolith_density_kg_m3: float
    friction_angle_deg: float
    bearing_capacity_pa: float
    gravity_m_s2: float


def site_conditions(
    world: WorldProvider | None,
    position: Vec3Spec,
    *,
    epoch: Epoch | None = None,
) -> SiteConditions:
    """Sample the pinned world's regolith + gravity at ``position`` (the RM-P0-WORLDS-05 handoff).

    With no world pinned, every field is the reduced-order lunar default — identical to the
    hand-authored Phase-0 behaviour, so an un-pinned scenario is unaffected."""
    if world is None:
        return SiteConditions(
            regolith_density_kg_m3=LUNAR_REGOLITH_DENSITY_KG_M3,
            friction_angle_deg=_DEFAULT_FRICTION_ANGLE_DEG,
            bearing_capacity_pa=_DEFAULT_BEARING_CAPACITY_PA,
            gravity_m_s2=LUNAR_GRAVITY_M_S2,
        )
    surface = world.sample(position, epoch=epoch)
    regolith = surface.regolith
    gx, gy, gz = surface.gravity
    magnitude = math.sqrt(gx * gx + gy * gy + gz * gz)
    return SiteConditions(
        regolith_density_kg_m3=(
            regolith.bulk_density_kg_m3
            if regolith.bulk_density_kg_m3 is not None
            else LUNAR_REGOLITH_DENSITY_KG_M3
        ),
        friction_angle_deg=(
            regolith.friction_angle_deg
            if regolith.friction_angle_deg is not None
            else _DEFAULT_FRICTION_ANGLE_DEG
        ),
        bearing_capacity_pa=(
            regolith.bearing_capacity_pa
            if regolith.bearing_capacity_pa is not None
            else _DEFAULT_BEARING_CAPACITY_PA
        ),
        gravity_m_s2=magnitude if magnitude > 0.0 else LUNAR_GRAVITY_M_S2,
    )


def mobility_dynamics_from_content(
    resolved: ResolvedAsset,
    *,
    world: WorldProvider | None = None,
    position: Vec3Spec = (0.0, 0.0, 0.0),
    epoch: Epoch | None = None,
) -> MobilityDynamics:
    """The reduced-order mobility block, sourced from the pinned asset + world.

    ``mass_kg`` and ``max_speed_mps`` come from the SADF asset; the drawbar-pull limit
    ``max_traction_n`` is *derived* — the friction cone of the asset's weight on the pinned
    regolith, ``mu * m * g`` with ``mu = tan(phi)`` — rather than typed in, so the traction limit
    follows the
    terrain the scenario actually pins (worlds.md §6: the constitutive law lives in Sim)."""
    asset = resolved.asset
    site = site_conditions(world, position, epoch=epoch)
    mass_kg = asset_mass_kg(asset, default=_DEFAULT_ROVER_MASS_KG)
    max_speed_mps, _ = asset_drive_limits(
        asset,
        default_speed_mps=_DEFAULT_MAX_SPEED_MPS,
        default_torque_nm=_DEFAULT_WHEEL_TORQUE_NM,
    )
    friction = math.tan(math.radians(site.friction_angle_deg))
    return MobilityDynamics(
        mass_kg=mass_kg,
        max_speed_mps=max_speed_mps,
        max_traction_n=friction * mass_kg * site.gravity_m_s2,
    )


def mujoco_dynamics_from_content(
    resolved: ResolvedAsset,
    *,
    world: WorldProvider | None = None,
    position: Vec3Spec = (0.0, 0.0, 0.0),
    epoch: Epoch | None = None,
) -> MujocoMobilityDynamics:
    """The MuJoCo articulated-contact block, sourced from the pinned asset + world.

    Every physical parameter the contact model needs now has a real provenance: the **chassis mass**
    and **wheel radius** and **actuator limits** from the Fleet SADF asset; the **regolith friction
    angle**, **bearing capacity**, and **gravity** from the Worlds sample at the agent's site. The
    friction cone the rover's traction is limited by is therefore the *pinned world's*, not a
    constant."""
    asset = resolved.asset
    site = site_conditions(world, position, epoch=epoch)
    max_speed_mps, wheel_torque_nm = asset_drive_limits(
        asset,
        default_speed_mps=_DEFAULT_MAX_SPEED_MPS,
        default_torque_nm=_DEFAULT_WHEEL_TORQUE_NM,
    )
    return MujocoMobilityDynamics(
        mass_kg=asset_mass_kg(asset, default=_DEFAULT_ROVER_MASS_KG),
        max_speed_mps=max_speed_mps,
        wheel_radius_m=asset_wheel_radius_m(asset, default=DEFAULT_WHEEL_RADIUS_M),
        wheel_torque_nm=wheel_torque_nm,
        gravity_m_s2=site.gravity_m_s2,
        friction_angle_deg=site.friction_angle_deg,
        bearing_capacity_pa=site.bearing_capacity_pa,
    )


def mjx_dynamics_from_content(
    resolved: ResolvedAsset,
    *,
    world: WorldProvider | None = None,
    position: Vec3Spec = (0.0, 0.0, 0.0),
    epoch: Epoch | None = None,
    batch_size: int = 64,
) -> MjxContactDynamics:
    """The MJX batched-contact block, sourced from the pinned asset + world.

    The same physical machine :func:`mujoco_dynamics_from_content` builds (so the CPU and GPU
    contact
    tiers step an identical rover), plus the batch shape the vectorized rollout is built at."""
    mujoco_block = mujoco_dynamics_from_content(
        resolved, world=world, position=position, epoch=epoch
    )
    return MjxContactDynamics(
        mass_kg=mujoco_block.mass_kg,
        max_speed_mps=mujoco_block.max_speed_mps,
        body_half_extents_m=mujoco_block.body_half_extents_m,
        wheel_radius_m=mujoco_block.wheel_radius_m,
        wheel_width_m=mujoco_block.wheel_width_m,
        wheel_mass_kg=mujoco_block.wheel_mass_kg,
        wheel_torque_nm=mujoco_block.wheel_torque_nm,
        gravity_m_s2=mujoco_block.gravity_m_s2,
        friction_angle_deg=mujoco_block.friction_angle_deg,
        bearing_capacity_pa=mujoco_block.bearing_capacity_pa,
        batch_size=batch_size,
    )


def granular_dynamics_from_content(
    resolved: ResolvedAsset,
    *,
    world: WorldProvider | None = None,
    position: Vec3Spec = (0.0, 0.0, 0.0),
    epoch: Epoch | None = None,
    max_dig_rate_m3_s: float = 0.01,
) -> GranularDynamics:
    """The reduced-order granular block, with its **regolith density** from the pinned world.

    ``regolith_density_kg_m3`` — the field that turns excavated volume into excavated *mass*, and so
    directly drives Bench's ``water_mass`` metric — is now the world's, not a scenario constant."""
    site = site_conditions(world, position, epoch=epoch)
    return GranularDynamics(
        regolith_density_kg_m3=site.regolith_density_kg_m3,
        max_dig_rate_m3_s=max_dig_rate_m3_s,
    )


def dem_granular_dynamics_from_content(
    resolved: ResolvedAsset,
    *,
    world: WorldProvider | None = None,
    position: Vec3Spec = (0.0, 0.0, 0.0),
    epoch: Epoch | None = None,
) -> DemGranularDynamics:
    """The high-fidelity DEM granular block, sourced from the pinned asset + world.

    Closes all three of the "plumbed later" gaps the DEM tier shipped with: the **terramechanics**
    (density, friction) from the Worlds ``RegolithParams``; the **gravity** from the Worlds gravity
    model; and the **tool geometry** (blade width/height) from the Fleet excavator's SADF ``tool``
    contact element. The DEM *numerics* (particle count/radius, contact stiffness, bed width) stay
    on the block — they size the reference bed, which is a solver choice, not a property of the
    world or
    the asset."""
    site = site_conditions(world, position, epoch=epoch)
    tool_width_m, tool_height_m = asset_tool_geometry(
        resolved.asset,
        default_width_m=_DEFAULT_TOOL_WIDTH_M,
        default_height_m=_DEFAULT_TOOL_HEIGHT_M,
    )
    return DemGranularDynamics(
        regolith_density_kg_m3=site.regolith_density_kg_m3,
        friction_coeff=math.tan(math.radians(site.friction_angle_deg)),
        gravity_m_s2=site.gravity_m_s2,
        bed_width_m=max(tool_width_m * _BED_WIDTH_PER_TOOL, _MIN_BED_WIDTH_M),
        tool_height_m=tool_height_m,
    )
