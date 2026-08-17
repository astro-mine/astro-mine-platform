# SPDX-License-Identifier: Apache-2.0
"""SADF v0.1 — closed vocabularies (Core-owned).

These enums are the *closed* part of the narrow waist: the capability-tag
vocabulary and the supporting kind enums that SADF authors choose from. They are
deliberately small and grow only by RFC (conventions.md §3; mission-model.md §3
"SADF grows by capability declaration, not type explosion"). Adding a member is an
append-only change; members are never removed or repurposed.

The capability-tag vocabulary serves four masters at once (mind/allocate/learn/
guard): task↔asset matching, action-space bounding, per-agent space-dict keys, and
the dual-use export-control gate. ``OPERATIONAL_TARGETING`` is reserved and gated:
open assets MUST NOT declare it (see :data:`GATED_CAPABILITY_TAGS`).
"""

from __future__ import annotations

from enum import StrEnum

__all__ = [
    "GATED_CAPABILITY_TAGS",
    "CapabilityTag",
    "CommsBand",
    "CommsProtocol",
    "ContactElementKind",
    "DeterminismClass",
    "EarthInterfaceMode",
    "FidelityTier",
    "FlightStack",
    "GeometryFormat",
    "GeometryRole",
    "JointType",
    "NodeRole",
    "PowerSourceKind",
    "PropellantType",
    "PropulsionKind",
    "Regime",
    "ReturnKind",
    "SensorKind",
    "SurrogatePhysicsDomain",
]


class CapabilityTag(StrEnum):
    """Closed, Core-owned autonomy/capability vocabulary (dotted namespaces).

    Declared by an asset and negotiated against by Allocate/Mind/Learn; also the
    substrate for export-control gating. Spans both flagship scenarios and all
    roadmap phases (RFC-0001 propulsion/return/anchoring tags included).
    """

    # mobility
    MOBILITY_WHEELED = "mobility.wheeled"
    MOBILITY_TRACKED = "mobility.tracked"
    MOBILITY_LEGGED = "mobility.legged"
    MOBILITY_HOP = "mobility.hop"
    MOBILITY_ORBITER = "mobility.orbiter"
    MOBILITY_ROCKET = "mobility.rocket"
    # propulsion (the structured budget lives in the propulsion block; these tags
    # advertise the capability for negotiation/gating)
    PROPULSION_CHEMICAL = "propulsion.chemical"
    PROPULSION_CRYO_CHEMICAL = "propulsion.cryo_chemical"
    PROPULSION_ELECTRIC_ION = "propulsion.electric_ion"
    PROPULSION_COLD_GAS = "propulsion.cold_gas"
    PROPULSION_SOLAR_SAIL = "propulsion.solar_sail"
    # excavation (ISRU mining, scenario 1)
    EXCAVATION_BUCKET = "excavation.bucket"
    EXCAVATION_AUGER = "excavation.auger"
    EXCAVATION_SCOOP = "excavation.scoop"
    EXCAVATION_DRILL = "excavation.drill"
    # sample collection (sample return, scenario 2)
    SAMPLE_COLLECTION_DRILL = "sample_collection.drill"
    SAMPLE_COLLECTION_SCOOP = "sample_collection.scoop"
    SAMPLE_COLLECTION_AUGER = "sample_collection.auger"
    SAMPLE_COLLECTION_PNEUMATIC = "sample_collection.pneumatic"
    SAMPLE_COLLECTION_CAPTURE_BAG = "sample_collection.capture_bag"
    # manipulation
    MANIPULATION_ARM = "manipulation.arm"
    MANIPULATION_GRIPPER = "manipulation.gripper"
    # prospecting (resource-sensing instruments)
    PROSPECTING_NEUTRON = "prospecting.neutron"
    PROSPECTING_NIR = "prospecting.nir"
    PROSPECTING_GPR = "prospecting.gpr"
    PROSPECTING_MASS_SPEC = "prospecting.mass_spec"
    PROSPECTING_DRILL_ASSAY = "prospecting.drill_assay"
    # general sensing
    SENSING_IMAGING = "sensing.imaging"
    SENSING_RANGING = "sensing.ranging"
    SENSING_LIDAR = "sensing.lidar"
    SENSING_IMU = "sensing.imu"
    SENSING_ODOMETRY = "sensing.odometry"
    SENSING_CONTACT = "sensing.contact"
    SENSING_ALTIMETRY = "sensing.altimetry"
    SENSING_THERMAL = "sensing.thermal"
    # comms
    COMMS_RELAY = "comms.relay"
    COMMS_DIRECT_TO_EARTH = "comms.direct_to_earth"
    COMMS_DSN = "comms.dsn"
    COMMS_DTN = "comms.dtn"
    # ISRU processes
    ISRU_THERMAL_EXTRACTION = "isru.thermal_extraction"
    ISRU_ELECTROLYSIS = "isru.electrolysis"
    ISRU_PURIFICATION = "isru.purification"
    ISRU_STORAGE = "isru.storage"
    # power
    POWER_GENERATION = "power.generation"
    POWER_STORAGE = "power.storage"
    POWER_DISTRIBUTION = "power.distribution"
    # anchoring / microgravity contact (scenario 2)
    ANCHORING_ANCHOR = "anchoring.anchor"
    ANCHORING_HARPOON = "anchoring.harpoon"
    ANCHORING_TOUCH_AND_GO = "anchoring.touch_and_go"
    ANCHORING_CAPTURE_BAG = "anchoring.capture_bag"
    # return / delivery (RFC-0001)
    RETURN_SAMPLE_CANISTER = "return.sample_canister"
    RETURN_BULK_HAULER = "return.bulk_hauler"
    EARTH_INTERFACE_BALLISTIC_CAPSULE = "earth_interface.ballistic_capsule"
    # structural / logistics capabilities
    STAGING = "staging"
    REUSABLE = "reusable"
    CARRIER_DISPENSER = "carrier.dispenser"
    # RESERVED + GATED dual-use tag. Open assets MUST NOT declare it; converting a
    # descriptive trajectory into executable maneuver guidance crosses this line and
    # is partitioned out of the commons (RFC-0001 §6; conventions.md §12).
    OPERATIONAL_TARGETING = "operational_targeting"
    # RESERVED + GATED. Reading the sealed ground-truth resource field is Sim's privilege
    # alone — the swarm must never see it (prospect.md §9 ground-truth isolation; the
    # ResourceField surface deliberately exposes no ground-truth accessor). An open-commons
    # asset or plugin MUST NOT declare it.
    GROUND_TRUTH_ACCESS = "ground_truth_access"
    # RESERVED + GATED. High-fidelity link prediction tied to a *live* mission is
    # operational availability intelligence (link.md §9; RFC-0003 / RM-P1-LINK-13). The
    # open commons predicts contacts from public ephemerides + parametric antenna models;
    # a plugin that couples prediction to real-asset scheduling MUST NOT be openly served.
    COMMS_LIVE_MISSION_LINK_PREDICTION = "comms.live_mission_link_prediction"


#: Capability tags that are reserved/gated: an open-commons SADF asset MUST NOT
#: declare them. The loader rejects any asset that does (see ``loader.validate_sadf``).
GATED_CAPABILITY_TAGS: frozenset[CapabilityTag] = frozenset(
    {
        CapabilityTag.OPERATIONAL_TARGETING,
        CapabilityTag.GROUND_TRUTH_ACCESS,
        CapabilityTag.COMMS_LIVE_MISSION_LINK_PREDICTION,
    }
)


class Regime(StrEnum):
    """The closed mission-regime enum (RFC-0001 R1, mission-model.md §1.2).

    Owned by the Mission/Phase/Regime schema; reused here so an asset can declare,
    in ``mobility.regimes``, which regimes it is built to operate in.
    """

    LAUNCH_ASCENT = "launch_ascent"
    INTERPLANETARY_TRANSIT = "interplanetary_transit"
    PROXIMITY_ORBIT = "proximity_orbit"
    SURFACE = "surface"
    ASCENT_RETURN = "ascent_return"
    EARTH_INTERFACE = "earth_interface"


class PropulsionKind(StrEnum):
    """Propulsion-system kind (sizing.md §3; mission-model.md §2.1)."""

    CHEMICAL_BIPROP = "chemical_biprop"
    CHEMICAL_MONOPROP = "chemical_monoprop"
    CRYO_CHEMICAL = "cryo_chemical"
    ELECTRIC_ION = "electric_ion"
    COLD_GAS = "cold_gas"
    SOLAR_SAIL = "solar_sail"


class PropellantType(StrEnum):
    """Propellant type (scenario 2 §6; sizing.md §3)."""

    MMH_NTO = "mmh_nto"
    LOX_LH2 = "lox_lh2"
    LOX_LCH4 = "lox_lch4"
    HYDRAZINE = "hydrazine"
    XENON = "xenon"
    KRYPTON = "krypton"
    COLD_GAS_N2 = "cold_gas_n2"
    OTHER = "other"


class PowerSourceKind(StrEnum):
    """Power-source kind (fleet.md §11; sim.md §3 powertherm)."""

    SOLAR = "solar"
    RTG = "rtg"
    RHU = "rhu"
    FUEL_CELL = "fuel_cell"
    EXTERNAL = "external"


class SensorKind(StrEnum):
    """Sensor kind (sim.md §1/§3; prospect.md §3)."""

    IMAGING = "imaging"
    LIDAR = "lidar"
    RANGEFINDER = "rangefinder"
    IMU = "imu"
    ODOMETRY = "odometry"
    CONTACT = "contact"
    NEUTRON_SPECTROMETER = "neutron_spectrometer"
    NIR_SPECTROMETER = "nir_spectrometer"
    GPR = "gpr"
    MASS_SPECTROMETER = "mass_spectrometer"
    DRILL_ASSAY = "drill_assay"
    ALTIMETER = "altimeter"
    THERMAL_SENSOR = "thermal_sensor"
    COMMS_LINK_STATE = "comms_link_state"
    # ISRU stored-mass gauge: reports cumulative extracted/stored resource (e.g. water,
    # kg) so a run's productivity is observable for Bench (RFC-0003; RM-P1-SIM-02).
    RESOURCE_STORAGE = "resource_storage"


class CommsBand(StrEnum):
    """RF / optical band (link.md §3/§5)."""

    UHF = "uhf"
    S_BAND = "s_band"
    X_BAND = "x_band"
    KA_BAND = "ka_band"
    OPTICAL = "optical"


class CommsProtocol(StrEnum):
    """Comms protocol an asset's radio speaks (link.md §4; bridge.md §4)."""

    ROS2_DDS = "ros2_dds"
    CCSDS_SPP = "ccsds_spp"
    CCSDS_TC_TM = "ccsds_tc_tm"
    CFDP = "cfdp"
    DTN_BPV7 = "dtn_bpv7"


class NodeRole(StrEnum):
    """Comms node role in the link topology (link.md §6)."""

    SPACE = "space"
    GROUND = "ground"


class JointType(StrEnum):
    """Kinematic joint type (fleet.md §10/§11; robotics-standard, engine-neutral)."""

    FIXED = "fixed"
    REVOLUTE = "revolute"
    CONTINUOUS = "continuous"
    PRISMATIC = "prismatic"


class ContactElementKind(StrEnum):
    """Asset-side ground-contact element (worlds.md §6/§11). The constitutive law
    lives in Sim; SADF declares only the asset's contact geometry/limits."""

    WHEEL = "wheel"
    TRACK = "track"
    LEG = "leg"
    TOOL = "tool"
    ANCHOR = "anchor"
    HARPOON = "harpoon"


class GeometryRole(StrEnum):
    """Geometry reference role (fleet.md §3)."""

    VISUAL = "visual"
    COLLISION = "collision"


class GeometryFormat(StrEnum):
    """Geometry interchange format (conventions.md §3; fleet.md §3/§11)."""

    USD = "usd"
    GLTF = "gltf"


class FidelityTier(StrEnum):
    """Multi-fidelity profile tier under one asset identity (fleet.md §"Key
    abstractions"; sim.md §"Architecture principles"). The surrogate tier maps to a
    declared :class:`SurrogatePhysicsDomain`."""

    MASSMODEL = "massmodel"
    KINEMATIC = "kinematic"
    ARTICULATED = "articulated"
    SURROGATE = "surrogate"


class SurrogatePhysicsDomain(StrEnum):
    """Physics domain a surrogate fidelity tier substitutes (surrogate.md §1/§11)."""

    GRANULAR_EXCAVATION = "granular_excavation"
    TERRAMECHANICS = "terramechanics"
    MANIPULATION_CONTACT = "manipulation_contact"
    THERMAL = "thermal"
    MICROGRAVITY_CONTACT = "microgravity_contact"


class DeterminismClass(StrEnum):
    """Determinism class for an asset's dynamics at a fidelity tier (sim.md §11)."""

    BIT_EXACT = "bit_exact"
    TOLERANCE = "tolerance"


class ReturnKind(StrEnum):
    """Return capability (RFC-0001; scenario 2 §6)."""

    SAMPLE_CANISTER = "sample_canister"
    BULK_HAULER = "bulk_hauler"
    NONE = "none"


class EarthInterfaceMode(StrEnum):
    """Earth-interface delivery mode — a delivery/recovery *event* with mass/Δv
    accounting, NOT a guided-EDL spec (RFC-0001 §6; mission-model.md §4)."""

    BALLISTIC_CAPSULE = "ballistic_capsule"
    NONE = "none"


class FlightStack(StrEnum):
    """Flight-software / robotics stack an asset speaks, for Bridge adapter
    selection (bridge.md §3). Sensitive stacks are capability-gated at load."""

    SIM = "sim"
    ROS2 = "ros2"
    CFS = "cfs"
    FPRIME = "fprime"
    CCSDS = "ccsds"
    DSN = "dsn"
