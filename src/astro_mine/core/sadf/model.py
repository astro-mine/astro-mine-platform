"""SADF v0.1 — typed Pydantic models (the in-memory asset representation).

These models are the typed view of a SADF document that components import and build
against (Fleet authors, Sim instantiates, Mind/Allocate/Guard reason over). The
**canonical** schema is the hand-authored JSON Schema shipped in-package at
``astro_mine/core/sadf/schema/sadf.schema.json``; these models mirror it. In
RM-P0-CORE-07 they will be *generated* from that schema
with ``datamodel-code-generator`` — until then they are hand-maintained and a
consistency test (``tests/test_sadf_consistency.py``) asserts the two agree.

Design note: these models are **purely structural** (no cross-reference or dual-use
semantic validation). Such rules — gated capability tags, ``root_frame`` resolution —
exceed JSON Schema's expressiveness and live in :mod:`astro_mine.core.sadf.loader`,
so the model stays behaviourally identical to the canonical JSON Schema.

All quantities are SI; every spatial value resolves in an explicitly named frame
(conventions.md §5; SADF principle "frame- and unit-explicit").
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.core import compat
from astro_mine.core.sadf.enums import (
    CapabilityTag,
    CommsBand,
    CommsProtocol,
    ContactElementKind,
    DeterminismClass,
    EarthInterfaceMode,
    FidelityTier,
    FlightStack,
    GeometryFormat,
    GeometryRole,
    JointType,
    NodeRole,
    PowerSourceKind,
    PropellantType,
    PropulsionKind,
    Regime,
    ReturnKind,
    SensorKind,
    SurrogatePhysicsDomain,
)

__all__ = [
    "Actuator",
    "Anchoring",
    "Antenna",
    "Asset",
    "Body",
    "Comms",
    "ContactElement",
    "FidelityProfile",
    "Frame",
    "GeometryRef",
    "Identity",
    "Inertia",
    "Interfaces",
    "Isru",
    "Joint",
    "JointLimits",
    "Mobility",
    "ModeLoad",
    "ObservationModel",
    "PayloadSlot",
    "PayloadSpec",
    "PowerBudget",
    "PowerSource",
    "PowerStorage",
    "Propellant",
    "Propulsion",
    "PropulsionSystem",
    "Provenance",
    "Quat",
    "Range",
    "ResourceTarget",
    "ReturnSpec",
    "SadfDocument",
    "Sensor",
    "Stage",
    "SubAssembly",
    "SurrogateProfile",
    "ThermalBudget",
    "Transform",
    "Vec3",
]

SADF_VERSION = "0.1"


class _Model(BaseModel):
    """Base for every SADF model: reject unknown/typo'd fields loudly."""

    model_config = ConfigDict(extra="forbid")


# --- value types -----------------------------------------------------------------


class Vec3(_Model):
    x: float
    y: float
    z: float


class Quat(_Model):
    """Unit quaternion, scalar-last (x, y, z, w)."""

    x: float
    y: float
    z: float
    w: float


class Transform(_Model):
    """A rigid transform of a frame relative to its parent (SI metres)."""

    translation_m: Vec3
    rotation_quat_xyzw: Quat


class Inertia(_Model):
    """Inertia tensor about the centre of mass (kg·m²), body frame."""

    ixx: float
    iyy: float
    izz: float
    ixy: float = 0.0
    ixz: float = 0.0
    iyz: float = 0.0


class Range(_Model):
    """A closed numeric interval [min, max] in the field's SI unit."""

    min: float
    max: float


# --- structure: frames, geometry, kinematics -------------------------------------


class Frame(_Model):
    """A named local reference frame. ``parent`` is null for the root frame. Body
    frames may carry a SPICE body id; full frame *types* arrive in RM-P0-CORE-06."""

    name: str
    parent: str | None = None
    transform: Transform | None = None
    spice_body_id: str | None = None


class GeometryRef(_Model):
    """Reference to external geometry (never embedded). USD for Sim, glTF for View."""

    role: GeometryRole
    format: GeometryFormat
    uri: str
    frame: str
    lod: int = 0


class Body(_Model):
    """A rigid body's mass properties (engine-neutral)."""

    name: str
    frame: str
    mass_kg: float
    center_of_mass_m: Vec3
    inertia_kg_m2: Inertia


class JointLimits(_Model):
    position_rad: Range | None = None
    velocity_rad_s: float | None = None
    effort_nm: float | None = None


class Joint(_Model):
    """A kinematic joint connecting two bodies (engine-neutral)."""

    name: str
    type: JointType
    parent_body: str
    child_body: str
    axis: Vec3 | None = None
    limits: JointLimits | None = None


class Actuator(_Model):
    """An actuator driving a joint, with limits and power draw (authoritative limit
    data Guard enforces; fleet.md §9)."""

    name: str
    target_joint: str | None = None
    torque_nm: float | None = None
    force_n: float | None = None
    velocity: float | None = None
    power_draw_w: float | None = None


# --- power & thermal -------------------------------------------------------------


class PowerSource(_Model):
    name: str
    kind: PowerSourceKind
    nominal_power_w: float


class PowerStorage(_Model):
    name: str
    capacity_j: float
    max_charge_w: float | None = None
    max_discharge_w: float | None = None


class ModeLoad(_Model):
    """Power draw for a named operating mode (Allocate scheduling input)."""

    mode: str
    power_w: float


class PowerBudget(_Model):
    sources: list[PowerSource] = Field(default_factory=list)
    storage: list[PowerStorage] = Field(default_factory=list)
    floor_w: float | None = None  # hard power floor Guard enforces (guard.md §3)
    loads_by_mode: list[ModeLoad] = Field(default_factory=list)


class ThermalBudget(_Model):
    operating_range_k: Range
    survival_range_k: Range | None = None  # survival floor for lunar night / cruise
    dissipation_w: float | None = None
    radiator_area_m2: float | None = None
    heater_power_w: float | None = None
    surface_coupling: bool = False  # couples to Worlds surface-temperature field


# --- sensors & comms -------------------------------------------------------------


class ObservationModel(_Model):
    """Asset-intrinsic sensor likelihood, shared by Sim's forward model and
    Prospect's belief update (prospect.md §3) so the two stay consistent."""

    noise_sigma: float | None = None
    footprint_m2: float | None = None
    depth_response_m: float | None = None
    range_m: float | None = None
    fov_deg: float | None = None


class ResourceTarget(_Model):
    """The resource species and SI unit a prospecting sensor observes
    (prospect.md §5, e.g. species='water_equivalent_hydrogen', si_unit='mass_fraction')."""

    species: str
    si_unit: str


class Sensor(_Model):
    name: str
    kind: SensorKind
    frame: str
    pose: Transform | None = None
    observation_model: ObservationModel | None = None
    resource: ResourceTarget | None = None


class Antenna(_Model):
    gain_dbi: float | None = None
    gain_pattern: str | None = None  # reference to a pattern artifact, if any
    boresight_frame: str | None = None
    pointing_accuracy_deg: float | None = None


class Comms(_Model):
    """An asset's radio (link.md §3/§5). Link computes path loss/SNR/windows; the
    asset declares only its intrinsic radio capability."""

    name: str
    band: CommsBand
    node_role: NodeRole = NodeRole.SPACE
    antenna: Antenna | None = None
    eirp_dbw: float | None = None
    gt_db_per_k: float | None = None
    tx_power_w: float | None = None
    min_rate_bps: float | None = None
    max_rate_bps: float | None = None
    modcod_supported: list[str] = Field(default_factory=list)
    protocols: list[CommsProtocol] = Field(default_factory=list)
    relay: bool = False


# --- mobility & contact ----------------------------------------------------------


class ContactElement(_Model):
    """Asset-side ground-contact element (worlds.md §6/§11). Constitutive physics
    lives in Sim; this declares geometry and limits only."""

    kind: ContactElementKind
    dimensions_m: Vec3 | None = None
    footprint_m2: float | None = None
    max_ground_pressure_pa: float | None = None
    max_slope_deg: float | None = None


class Anchoring(_Model):
    """Microgravity anchoring/contact force envelope (scenario 2 §3; worlds.md §11)."""

    max_force_n: float


class Mobility(_Model):
    regimes: list[Regime] = Field(default_factory=list)
    contact: list[ContactElement] = Field(default_factory=list)
    anchoring: Anchoring | None = None


# --- propulsion / return / payload (RFC-0001) ------------------------------------


class Propellant(_Model):
    type: PropellantType
    mass_kg: float


class PropulsionSystem(_Model):
    kind: PropulsionKind
    thrust_n: float | None = None
    isp_s: float | None = None
    propellant: Propellant | None = None
    power_w: float | None = None  # for electric/low-thrust systems


class Stage(_Model):
    dry_kg: float
    propellant_kg: float


class Propulsion(_Model):
    """Descriptive propulsion budget (RFC-0001; mission-model.md §2.1). NOT
    executable guidance — see the ``operational_targeting`` gate."""

    systems: list[PropulsionSystem] = Field(default_factory=list)
    delta_v_budget_mps: float | None = None
    staging: list[Stage] = Field(default_factory=list)


class ReturnSpec(_Model):
    """Return / Earth-interface capability. ``earth_interface`` is a delivery event
    with mass/Δv accounting, not a guided-EDL spec (RFC-0001 §6)."""

    capability: ReturnKind = ReturnKind.NONE
    payload_capacity_kg: float | None = None
    earth_interface: EarthInterfaceMode = EarthInterfaceMode.NONE


class Isru(_Model):
    """ISRU plant throughput (processes are declared via ``isru.*`` capability tags)."""

    throughput_kg_hr: float | None = None
    plant_power_w: float | None = None


class PayloadSlot(_Model):
    name: str
    frame: str
    accepts: list[str] = Field(default_factory=list)
    max_mass_kg: float | None = None


class PayloadSpec(_Model):
    slots: list[PayloadSlot] = Field(default_factory=list)
    isru: Isru | None = None
    capacity_kg: float | None = None


class SubAssembly(_Model):
    """A nested asset reference (carrier-and-daughters, descent stage)."""

    ref: str
    mount_frame: str
    transform: Transform | None = None


# --- fidelity, interfaces, provenance --------------------------------------------


class SurrogateProfile(_Model):
    physics_domain: SurrogatePhysicsDomain
    trust_region: str | None = None  # opaque ref to the surrogate's trust-region spec


class FidelityProfile(_Model):
    """A fidelity tier under the one asset identity (fleet.md; sim.md). The surrogate
    tier names the physics domain it substitutes (surrogate.md §1)."""

    tier: FidelityTier
    determinism_class: DeterminismClass = DeterminismClass.TOLERANCE
    detail: str | None = None  # which bodies/geometry subset this tier instantiates
    surrogate: SurrogateProfile | None = None


class Interfaces(_Model):
    """Observation/action interface descriptors for policy↔asset↔world compatibility
    resolution (hub.md §11; learn.md §3). May be derived from sensors/actuators."""

    observation_space: str | None = None
    action_space: str | None = None


class Provenance(_Model):
    """Reproducibility provenance (conventions.md §5). Signatures/SBOM attach to the
    published artifact as OCI referrers (hub.md §9), not inline here."""

    input_hashes: list[str] = Field(default_factory=list)
    code_version: str | None = None
    toolchain_version: str | None = None
    env_lockfile: str | None = None
    seed: int | None = None


# --- identity & top level --------------------------------------------------------


class Identity(_Model):
    """Asset identity. ``kind`` is a free, Fleet-namespaced taxonomy label (advisory);
    the closed, Core-owned negotiation vocabulary is ``Asset.capabilities``
    (mission-model.md §3: grow by capability, not type)."""

    id: str
    name: str
    version: str
    kind: str
    description: str | None = None
    labels: dict[str, str] = Field(default_factory=dict)


class Asset(_Model):
    identity: Identity
    capabilities: list[CapabilityTag] = Field(default_factory=list)
    # Maps a Core interface name (``sadf``, ``messages``, …) to the SemVer this asset is
    # authored against — the input to the registry's version negotiation, matching the
    # plugin manifest's ``core_interfaces`` and :data:`compat.CORE_INTERFACE_VERSIONS`.
    core_interface_versions: dict[str, str] = Field(default_factory=dict)
    frames: list[Frame] = Field(default_factory=list)
    root_frame: str
    geometry: list[GeometryRef] = Field(default_factory=list)
    bodies: list[Body] = Field(default_factory=list)
    joints: list[Joint] = Field(default_factory=list)
    actuators: list[Actuator] = Field(default_factory=list)
    power: PowerBudget | None = None
    thermal: ThermalBudget | None = None
    sensors: list[Sensor] = Field(default_factory=list)
    comms: list[Comms] = Field(default_factory=list)
    mobility: Mobility | None = None
    propulsion: Propulsion | None = None
    # 'return' is a Python keyword: stored as ``return_``, authored/serialized as
    # ``return`` via alias.
    return_: ReturnSpec | None = Field(default=None, alias="return")
    payload: PayloadSpec | None = None
    subassemblies: list[SubAssembly] = Field(default_factory=list)
    fidelity_profiles: list[FidelityProfile] = Field(default_factory=list)
    flight_stacks: list[FlightStack] = Field(default_factory=list)
    interfaces: Interfaces | None = None
    provenance: Provenance | None = None

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    def assert_core_compatible(self, provided: Mapping[str, str] | None = None) -> None:
        """Assert this asset's :attr:`core_interface_versions` are satisfied by this Core.

        The bridge from a SADF asset to interface-version negotiation: delegates to
        :func:`astro_mine.core.compat.assert_core_compatible` (the same rule the plugin
        registry applies to a manifest's ``core_interfaces``). ``provided`` overrides the
        Core interface versions negotiated against (defaults to this build's
        :data:`~astro_mine.core.compat.CORE_INTERFACE_VERSIONS`). Raises
        :class:`~astro_mine.core.compat.IncompatibleCoreInterface` on any mismatch."""
        compat.assert_core_compatible(self.core_interface_versions, provided=provided)


class SadfDocument(_Model):
    """Top-level SADF document. ``sadf_version`` pins the schema minor (RM-P0-CORE-01)."""

    sadf_version: Literal["0.1"]
    asset: Asset
