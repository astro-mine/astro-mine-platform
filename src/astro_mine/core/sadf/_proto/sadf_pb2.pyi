from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Vec3(_message.Message):
    __slots__ = ("x", "y", "z")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    z: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ...) -> None: ...

class Quat(_message.Message):
    __slots__ = ("x", "y", "z", "w")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    W_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    z: float
    w: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ..., w: _Optional[float] = ...) -> None: ...

class Transform(_message.Message):
    __slots__ = ("translation_m", "rotation_quat_xyzw")
    TRANSLATION_M_FIELD_NUMBER: _ClassVar[int]
    ROTATION_QUAT_XYZW_FIELD_NUMBER: _ClassVar[int]
    translation_m: Vec3
    rotation_quat_xyzw: Quat
    def __init__(self, translation_m: _Optional[_Union[Vec3, _Mapping]] = ..., rotation_quat_xyzw: _Optional[_Union[Quat, _Mapping]] = ...) -> None: ...

class Inertia(_message.Message):
    __slots__ = ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")
    IXX_FIELD_NUMBER: _ClassVar[int]
    IYY_FIELD_NUMBER: _ClassVar[int]
    IZZ_FIELD_NUMBER: _ClassVar[int]
    IXY_FIELD_NUMBER: _ClassVar[int]
    IXZ_FIELD_NUMBER: _ClassVar[int]
    IYZ_FIELD_NUMBER: _ClassVar[int]
    ixx: float
    iyy: float
    izz: float
    ixy: float
    ixz: float
    iyz: float
    def __init__(self, ixx: _Optional[float] = ..., iyy: _Optional[float] = ..., izz: _Optional[float] = ..., ixy: _Optional[float] = ..., ixz: _Optional[float] = ..., iyz: _Optional[float] = ...) -> None: ...

class Range(_message.Message):
    __slots__ = ("min", "max")
    MIN_FIELD_NUMBER: _ClassVar[int]
    MAX_FIELD_NUMBER: _ClassVar[int]
    min: float
    max: float
    def __init__(self, min: _Optional[float] = ..., max: _Optional[float] = ...) -> None: ...

class Frame(_message.Message):
    __slots__ = ("name", "parent", "transform", "spice_body_id")
    NAME_FIELD_NUMBER: _ClassVar[int]
    PARENT_FIELD_NUMBER: _ClassVar[int]
    TRANSFORM_FIELD_NUMBER: _ClassVar[int]
    SPICE_BODY_ID_FIELD_NUMBER: _ClassVar[int]
    name: str
    parent: str
    transform: Transform
    spice_body_id: str
    def __init__(self, name: _Optional[str] = ..., parent: _Optional[str] = ..., transform: _Optional[_Union[Transform, _Mapping]] = ..., spice_body_id: _Optional[str] = ...) -> None: ...

class GeometryRef(_message.Message):
    __slots__ = ("role", "format", "uri", "frame", "lod")
    ROLE_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    URI_FIELD_NUMBER: _ClassVar[int]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    LOD_FIELD_NUMBER: _ClassVar[int]
    role: str
    format: str
    uri: str
    frame: str
    lod: int
    def __init__(self, role: _Optional[str] = ..., format: _Optional[str] = ..., uri: _Optional[str] = ..., frame: _Optional[str] = ..., lod: _Optional[int] = ...) -> None: ...

class Body(_message.Message):
    __slots__ = ("name", "frame", "mass_kg", "center_of_mass_m", "inertia_kg_m2")
    NAME_FIELD_NUMBER: _ClassVar[int]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    MASS_KG_FIELD_NUMBER: _ClassVar[int]
    CENTER_OF_MASS_M_FIELD_NUMBER: _ClassVar[int]
    INERTIA_KG_M2_FIELD_NUMBER: _ClassVar[int]
    name: str
    frame: str
    mass_kg: float
    center_of_mass_m: Vec3
    inertia_kg_m2: Inertia
    def __init__(self, name: _Optional[str] = ..., frame: _Optional[str] = ..., mass_kg: _Optional[float] = ..., center_of_mass_m: _Optional[_Union[Vec3, _Mapping]] = ..., inertia_kg_m2: _Optional[_Union[Inertia, _Mapping]] = ...) -> None: ...

class JointLimits(_message.Message):
    __slots__ = ("position_rad", "velocity_rad_s", "effort_nm")
    POSITION_RAD_FIELD_NUMBER: _ClassVar[int]
    VELOCITY_RAD_S_FIELD_NUMBER: _ClassVar[int]
    EFFORT_NM_FIELD_NUMBER: _ClassVar[int]
    position_rad: Range
    velocity_rad_s: float
    effort_nm: float
    def __init__(self, position_rad: _Optional[_Union[Range, _Mapping]] = ..., velocity_rad_s: _Optional[float] = ..., effort_nm: _Optional[float] = ...) -> None: ...

class Joint(_message.Message):
    __slots__ = ("name", "type", "parent_body", "child_body", "axis", "limits")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TYPE_FIELD_NUMBER: _ClassVar[int]
    PARENT_BODY_FIELD_NUMBER: _ClassVar[int]
    CHILD_BODY_FIELD_NUMBER: _ClassVar[int]
    AXIS_FIELD_NUMBER: _ClassVar[int]
    LIMITS_FIELD_NUMBER: _ClassVar[int]
    name: str
    type: str
    parent_body: str
    child_body: str
    axis: Vec3
    limits: JointLimits
    def __init__(self, name: _Optional[str] = ..., type: _Optional[str] = ..., parent_body: _Optional[str] = ..., child_body: _Optional[str] = ..., axis: _Optional[_Union[Vec3, _Mapping]] = ..., limits: _Optional[_Union[JointLimits, _Mapping]] = ...) -> None: ...

class Actuator(_message.Message):
    __slots__ = ("name", "target_joint", "torque_nm", "force_n", "velocity", "power_draw_w")
    NAME_FIELD_NUMBER: _ClassVar[int]
    TARGET_JOINT_FIELD_NUMBER: _ClassVar[int]
    TORQUE_NM_FIELD_NUMBER: _ClassVar[int]
    FORCE_N_FIELD_NUMBER: _ClassVar[int]
    VELOCITY_FIELD_NUMBER: _ClassVar[int]
    POWER_DRAW_W_FIELD_NUMBER: _ClassVar[int]
    name: str
    target_joint: str
    torque_nm: float
    force_n: float
    velocity: float
    power_draw_w: float
    def __init__(self, name: _Optional[str] = ..., target_joint: _Optional[str] = ..., torque_nm: _Optional[float] = ..., force_n: _Optional[float] = ..., velocity: _Optional[float] = ..., power_draw_w: _Optional[float] = ...) -> None: ...

class PowerSource(_message.Message):
    __slots__ = ("name", "kind", "nominal_power_w")
    NAME_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    NOMINAL_POWER_W_FIELD_NUMBER: _ClassVar[int]
    name: str
    kind: str
    nominal_power_w: float
    def __init__(self, name: _Optional[str] = ..., kind: _Optional[str] = ..., nominal_power_w: _Optional[float] = ...) -> None: ...

class PowerStorage(_message.Message):
    __slots__ = ("name", "capacity_j", "max_charge_w", "max_discharge_w")
    NAME_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_J_FIELD_NUMBER: _ClassVar[int]
    MAX_CHARGE_W_FIELD_NUMBER: _ClassVar[int]
    MAX_DISCHARGE_W_FIELD_NUMBER: _ClassVar[int]
    name: str
    capacity_j: float
    max_charge_w: float
    max_discharge_w: float
    def __init__(self, name: _Optional[str] = ..., capacity_j: _Optional[float] = ..., max_charge_w: _Optional[float] = ..., max_discharge_w: _Optional[float] = ...) -> None: ...

class ModeLoad(_message.Message):
    __slots__ = ("mode", "power_w")
    MODE_FIELD_NUMBER: _ClassVar[int]
    POWER_W_FIELD_NUMBER: _ClassVar[int]
    mode: str
    power_w: float
    def __init__(self, mode: _Optional[str] = ..., power_w: _Optional[float] = ...) -> None: ...

class PowerBudget(_message.Message):
    __slots__ = ("sources", "storage", "floor_w", "loads_by_mode")
    SOURCES_FIELD_NUMBER: _ClassVar[int]
    STORAGE_FIELD_NUMBER: _ClassVar[int]
    FLOOR_W_FIELD_NUMBER: _ClassVar[int]
    LOADS_BY_MODE_FIELD_NUMBER: _ClassVar[int]
    sources: _containers.RepeatedCompositeFieldContainer[PowerSource]
    storage: _containers.RepeatedCompositeFieldContainer[PowerStorage]
    floor_w: float
    loads_by_mode: _containers.RepeatedCompositeFieldContainer[ModeLoad]
    def __init__(self, sources: _Optional[_Iterable[_Union[PowerSource, _Mapping]]] = ..., storage: _Optional[_Iterable[_Union[PowerStorage, _Mapping]]] = ..., floor_w: _Optional[float] = ..., loads_by_mode: _Optional[_Iterable[_Union[ModeLoad, _Mapping]]] = ...) -> None: ...

class ThermalBudget(_message.Message):
    __slots__ = ("operating_range_k", "survival_range_k", "dissipation_w", "radiator_area_m2", "heater_power_w", "surface_coupling")
    OPERATING_RANGE_K_FIELD_NUMBER: _ClassVar[int]
    SURVIVAL_RANGE_K_FIELD_NUMBER: _ClassVar[int]
    DISSIPATION_W_FIELD_NUMBER: _ClassVar[int]
    RADIATOR_AREA_M2_FIELD_NUMBER: _ClassVar[int]
    HEATER_POWER_W_FIELD_NUMBER: _ClassVar[int]
    SURFACE_COUPLING_FIELD_NUMBER: _ClassVar[int]
    operating_range_k: Range
    survival_range_k: Range
    dissipation_w: float
    radiator_area_m2: float
    heater_power_w: float
    surface_coupling: bool
    def __init__(self, operating_range_k: _Optional[_Union[Range, _Mapping]] = ..., survival_range_k: _Optional[_Union[Range, _Mapping]] = ..., dissipation_w: _Optional[float] = ..., radiator_area_m2: _Optional[float] = ..., heater_power_w: _Optional[float] = ..., surface_coupling: _Optional[bool] = ...) -> None: ...

class ObservationModel(_message.Message):
    __slots__ = ("noise_sigma", "footprint_m2", "depth_response_m", "range_m", "fov_deg")
    NOISE_SIGMA_FIELD_NUMBER: _ClassVar[int]
    FOOTPRINT_M2_FIELD_NUMBER: _ClassVar[int]
    DEPTH_RESPONSE_M_FIELD_NUMBER: _ClassVar[int]
    RANGE_M_FIELD_NUMBER: _ClassVar[int]
    FOV_DEG_FIELD_NUMBER: _ClassVar[int]
    noise_sigma: float
    footprint_m2: float
    depth_response_m: float
    range_m: float
    fov_deg: float
    def __init__(self, noise_sigma: _Optional[float] = ..., footprint_m2: _Optional[float] = ..., depth_response_m: _Optional[float] = ..., range_m: _Optional[float] = ..., fov_deg: _Optional[float] = ...) -> None: ...

class ResourceTarget(_message.Message):
    __slots__ = ("species", "si_unit")
    SPECIES_FIELD_NUMBER: _ClassVar[int]
    SI_UNIT_FIELD_NUMBER: _ClassVar[int]
    species: str
    si_unit: str
    def __init__(self, species: _Optional[str] = ..., si_unit: _Optional[str] = ...) -> None: ...

class Sensor(_message.Message):
    __slots__ = ("name", "kind", "frame", "pose", "observation_model", "resource")
    NAME_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    POSE_FIELD_NUMBER: _ClassVar[int]
    OBSERVATION_MODEL_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_FIELD_NUMBER: _ClassVar[int]
    name: str
    kind: str
    frame: str
    pose: Transform
    observation_model: ObservationModel
    resource: ResourceTarget
    def __init__(self, name: _Optional[str] = ..., kind: _Optional[str] = ..., frame: _Optional[str] = ..., pose: _Optional[_Union[Transform, _Mapping]] = ..., observation_model: _Optional[_Union[ObservationModel, _Mapping]] = ..., resource: _Optional[_Union[ResourceTarget, _Mapping]] = ...) -> None: ...

class Antenna(_message.Message):
    __slots__ = ("gain_dbi", "gain_pattern", "boresight_frame", "pointing_accuracy_deg")
    GAIN_DBI_FIELD_NUMBER: _ClassVar[int]
    GAIN_PATTERN_FIELD_NUMBER: _ClassVar[int]
    BORESIGHT_FRAME_FIELD_NUMBER: _ClassVar[int]
    POINTING_ACCURACY_DEG_FIELD_NUMBER: _ClassVar[int]
    gain_dbi: float
    gain_pattern: str
    boresight_frame: str
    pointing_accuracy_deg: float
    def __init__(self, gain_dbi: _Optional[float] = ..., gain_pattern: _Optional[str] = ..., boresight_frame: _Optional[str] = ..., pointing_accuracy_deg: _Optional[float] = ...) -> None: ...

class Comms(_message.Message):
    __slots__ = ("name", "band", "node_role", "antenna", "eirp_dbw", "gt_db_per_k", "tx_power_w", "min_rate_bps", "max_rate_bps", "modcod_supported", "protocols", "relay")
    NAME_FIELD_NUMBER: _ClassVar[int]
    BAND_FIELD_NUMBER: _ClassVar[int]
    NODE_ROLE_FIELD_NUMBER: _ClassVar[int]
    ANTENNA_FIELD_NUMBER: _ClassVar[int]
    EIRP_DBW_FIELD_NUMBER: _ClassVar[int]
    GT_DB_PER_K_FIELD_NUMBER: _ClassVar[int]
    TX_POWER_W_FIELD_NUMBER: _ClassVar[int]
    MIN_RATE_BPS_FIELD_NUMBER: _ClassVar[int]
    MAX_RATE_BPS_FIELD_NUMBER: _ClassVar[int]
    MODCOD_SUPPORTED_FIELD_NUMBER: _ClassVar[int]
    PROTOCOLS_FIELD_NUMBER: _ClassVar[int]
    RELAY_FIELD_NUMBER: _ClassVar[int]
    name: str
    band: str
    node_role: str
    antenna: Antenna
    eirp_dbw: float
    gt_db_per_k: float
    tx_power_w: float
    min_rate_bps: float
    max_rate_bps: float
    modcod_supported: _containers.RepeatedScalarFieldContainer[str]
    protocols: _containers.RepeatedScalarFieldContainer[str]
    relay: bool
    def __init__(self, name: _Optional[str] = ..., band: _Optional[str] = ..., node_role: _Optional[str] = ..., antenna: _Optional[_Union[Antenna, _Mapping]] = ..., eirp_dbw: _Optional[float] = ..., gt_db_per_k: _Optional[float] = ..., tx_power_w: _Optional[float] = ..., min_rate_bps: _Optional[float] = ..., max_rate_bps: _Optional[float] = ..., modcod_supported: _Optional[_Iterable[str]] = ..., protocols: _Optional[_Iterable[str]] = ..., relay: _Optional[bool] = ...) -> None: ...

class ContactElement(_message.Message):
    __slots__ = ("kind", "dimensions_m", "footprint_m2", "max_ground_pressure_pa", "max_slope_deg")
    KIND_FIELD_NUMBER: _ClassVar[int]
    DIMENSIONS_M_FIELD_NUMBER: _ClassVar[int]
    FOOTPRINT_M2_FIELD_NUMBER: _ClassVar[int]
    MAX_GROUND_PRESSURE_PA_FIELD_NUMBER: _ClassVar[int]
    MAX_SLOPE_DEG_FIELD_NUMBER: _ClassVar[int]
    kind: str
    dimensions_m: Vec3
    footprint_m2: float
    max_ground_pressure_pa: float
    max_slope_deg: float
    def __init__(self, kind: _Optional[str] = ..., dimensions_m: _Optional[_Union[Vec3, _Mapping]] = ..., footprint_m2: _Optional[float] = ..., max_ground_pressure_pa: _Optional[float] = ..., max_slope_deg: _Optional[float] = ...) -> None: ...

class Anchoring(_message.Message):
    __slots__ = ("max_force_n",)
    MAX_FORCE_N_FIELD_NUMBER: _ClassVar[int]
    max_force_n: float
    def __init__(self, max_force_n: _Optional[float] = ...) -> None: ...

class Mobility(_message.Message):
    __slots__ = ("regimes", "contact", "anchoring")
    REGIMES_FIELD_NUMBER: _ClassVar[int]
    CONTACT_FIELD_NUMBER: _ClassVar[int]
    ANCHORING_FIELD_NUMBER: _ClassVar[int]
    regimes: _containers.RepeatedScalarFieldContainer[str]
    contact: _containers.RepeatedCompositeFieldContainer[ContactElement]
    anchoring: Anchoring
    def __init__(self, regimes: _Optional[_Iterable[str]] = ..., contact: _Optional[_Iterable[_Union[ContactElement, _Mapping]]] = ..., anchoring: _Optional[_Union[Anchoring, _Mapping]] = ...) -> None: ...

class Propellant(_message.Message):
    __slots__ = ("type", "mass_kg")
    TYPE_FIELD_NUMBER: _ClassVar[int]
    MASS_KG_FIELD_NUMBER: _ClassVar[int]
    type: str
    mass_kg: float
    def __init__(self, type: _Optional[str] = ..., mass_kg: _Optional[float] = ...) -> None: ...

class PropulsionSystem(_message.Message):
    __slots__ = ("kind", "thrust_n", "isp_s", "propellant", "power_w")
    KIND_FIELD_NUMBER: _ClassVar[int]
    THRUST_N_FIELD_NUMBER: _ClassVar[int]
    ISP_S_FIELD_NUMBER: _ClassVar[int]
    PROPELLANT_FIELD_NUMBER: _ClassVar[int]
    POWER_W_FIELD_NUMBER: _ClassVar[int]
    kind: str
    thrust_n: float
    isp_s: float
    propellant: Propellant
    power_w: float
    def __init__(self, kind: _Optional[str] = ..., thrust_n: _Optional[float] = ..., isp_s: _Optional[float] = ..., propellant: _Optional[_Union[Propellant, _Mapping]] = ..., power_w: _Optional[float] = ...) -> None: ...

class Stage(_message.Message):
    __slots__ = ("dry_kg", "propellant_kg")
    DRY_KG_FIELD_NUMBER: _ClassVar[int]
    PROPELLANT_KG_FIELD_NUMBER: _ClassVar[int]
    dry_kg: float
    propellant_kg: float
    def __init__(self, dry_kg: _Optional[float] = ..., propellant_kg: _Optional[float] = ...) -> None: ...

class Propulsion(_message.Message):
    __slots__ = ("systems", "delta_v_budget_mps", "staging")
    SYSTEMS_FIELD_NUMBER: _ClassVar[int]
    DELTA_V_BUDGET_MPS_FIELD_NUMBER: _ClassVar[int]
    STAGING_FIELD_NUMBER: _ClassVar[int]
    systems: _containers.RepeatedCompositeFieldContainer[PropulsionSystem]
    delta_v_budget_mps: float
    staging: _containers.RepeatedCompositeFieldContainer[Stage]
    def __init__(self, systems: _Optional[_Iterable[_Union[PropulsionSystem, _Mapping]]] = ..., delta_v_budget_mps: _Optional[float] = ..., staging: _Optional[_Iterable[_Union[Stage, _Mapping]]] = ...) -> None: ...

class ReturnSpec(_message.Message):
    __slots__ = ("capability", "payload_capacity_kg", "earth_interface")
    CAPABILITY_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_CAPACITY_KG_FIELD_NUMBER: _ClassVar[int]
    EARTH_INTERFACE_FIELD_NUMBER: _ClassVar[int]
    capability: str
    payload_capacity_kg: float
    earth_interface: str
    def __init__(self, capability: _Optional[str] = ..., payload_capacity_kg: _Optional[float] = ..., earth_interface: _Optional[str] = ...) -> None: ...

class Isru(_message.Message):
    __slots__ = ("throughput_kg_hr", "plant_power_w")
    THROUGHPUT_KG_HR_FIELD_NUMBER: _ClassVar[int]
    PLANT_POWER_W_FIELD_NUMBER: _ClassVar[int]
    throughput_kg_hr: float
    plant_power_w: float
    def __init__(self, throughput_kg_hr: _Optional[float] = ..., plant_power_w: _Optional[float] = ...) -> None: ...

class PayloadSlot(_message.Message):
    __slots__ = ("name", "frame", "accepts", "max_mass_kg")
    NAME_FIELD_NUMBER: _ClassVar[int]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    ACCEPTS_FIELD_NUMBER: _ClassVar[int]
    MAX_MASS_KG_FIELD_NUMBER: _ClassVar[int]
    name: str
    frame: str
    accepts: _containers.RepeatedScalarFieldContainer[str]
    max_mass_kg: float
    def __init__(self, name: _Optional[str] = ..., frame: _Optional[str] = ..., accepts: _Optional[_Iterable[str]] = ..., max_mass_kg: _Optional[float] = ...) -> None: ...

class PayloadSpec(_message.Message):
    __slots__ = ("slots", "isru", "capacity_kg")
    SLOTS_FIELD_NUMBER: _ClassVar[int]
    ISRU_FIELD_NUMBER: _ClassVar[int]
    CAPACITY_KG_FIELD_NUMBER: _ClassVar[int]
    slots: _containers.RepeatedCompositeFieldContainer[PayloadSlot]
    isru: Isru
    capacity_kg: float
    def __init__(self, slots: _Optional[_Iterable[_Union[PayloadSlot, _Mapping]]] = ..., isru: _Optional[_Union[Isru, _Mapping]] = ..., capacity_kg: _Optional[float] = ...) -> None: ...

class SubAssembly(_message.Message):
    __slots__ = ("ref", "mount_frame", "transform")
    REF_FIELD_NUMBER: _ClassVar[int]
    MOUNT_FRAME_FIELD_NUMBER: _ClassVar[int]
    TRANSFORM_FIELD_NUMBER: _ClassVar[int]
    ref: str
    mount_frame: str
    transform: Transform
    def __init__(self, ref: _Optional[str] = ..., mount_frame: _Optional[str] = ..., transform: _Optional[_Union[Transform, _Mapping]] = ...) -> None: ...

class SurrogateProfile(_message.Message):
    __slots__ = ("physics_domain", "trust_region")
    PHYSICS_DOMAIN_FIELD_NUMBER: _ClassVar[int]
    TRUST_REGION_FIELD_NUMBER: _ClassVar[int]
    physics_domain: str
    trust_region: str
    def __init__(self, physics_domain: _Optional[str] = ..., trust_region: _Optional[str] = ...) -> None: ...

class FidelityProfile(_message.Message):
    __slots__ = ("tier", "determinism_class", "detail", "surrogate")
    TIER_FIELD_NUMBER: _ClassVar[int]
    DETERMINISM_CLASS_FIELD_NUMBER: _ClassVar[int]
    DETAIL_FIELD_NUMBER: _ClassVar[int]
    SURROGATE_FIELD_NUMBER: _ClassVar[int]
    tier: str
    determinism_class: str
    detail: str
    surrogate: SurrogateProfile
    def __init__(self, tier: _Optional[str] = ..., determinism_class: _Optional[str] = ..., detail: _Optional[str] = ..., surrogate: _Optional[_Union[SurrogateProfile, _Mapping]] = ...) -> None: ...

class Interfaces(_message.Message):
    __slots__ = ("observation_space", "action_space")
    OBSERVATION_SPACE_FIELD_NUMBER: _ClassVar[int]
    ACTION_SPACE_FIELD_NUMBER: _ClassVar[int]
    observation_space: str
    action_space: str
    def __init__(self, observation_space: _Optional[str] = ..., action_space: _Optional[str] = ...) -> None: ...

class Provenance(_message.Message):
    __slots__ = ("input_hashes", "code_version", "toolchain_version", "env_lockfile", "seed")
    INPUT_HASHES_FIELD_NUMBER: _ClassVar[int]
    CODE_VERSION_FIELD_NUMBER: _ClassVar[int]
    TOOLCHAIN_VERSION_FIELD_NUMBER: _ClassVar[int]
    ENV_LOCKFILE_FIELD_NUMBER: _ClassVar[int]
    SEED_FIELD_NUMBER: _ClassVar[int]
    input_hashes: _containers.RepeatedScalarFieldContainer[str]
    code_version: str
    toolchain_version: str
    env_lockfile: str
    seed: int
    def __init__(self, input_hashes: _Optional[_Iterable[str]] = ..., code_version: _Optional[str] = ..., toolchain_version: _Optional[str] = ..., env_lockfile: _Optional[str] = ..., seed: _Optional[int] = ...) -> None: ...

class Identity(_message.Message):
    __slots__ = ("id", "name", "version", "kind", "description", "labels")
    class LabelsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    VERSION_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    version: str
    kind: str
    description: str
    labels: _containers.ScalarMap[str, str]
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., version: _Optional[str] = ..., kind: _Optional[str] = ..., description: _Optional[str] = ..., labels: _Optional[_Mapping[str, str]] = ...) -> None: ...

class Asset(_message.Message):
    __slots__ = ("identity", "capabilities", "core_interface_versions", "frames", "root_frame", "geometry", "bodies", "joints", "actuators", "power", "thermal", "sensors", "comms", "mobility", "propulsion", "return_spec", "payload", "subassemblies", "fidelity_profiles", "flight_stacks", "interfaces", "provenance")
    class CoreInterfaceVersionsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    IDENTITY_FIELD_NUMBER: _ClassVar[int]
    CAPABILITIES_FIELD_NUMBER: _ClassVar[int]
    CORE_INTERFACE_VERSIONS_FIELD_NUMBER: _ClassVar[int]
    FRAMES_FIELD_NUMBER: _ClassVar[int]
    ROOT_FRAME_FIELD_NUMBER: _ClassVar[int]
    GEOMETRY_FIELD_NUMBER: _ClassVar[int]
    BODIES_FIELD_NUMBER: _ClassVar[int]
    JOINTS_FIELD_NUMBER: _ClassVar[int]
    ACTUATORS_FIELD_NUMBER: _ClassVar[int]
    POWER_FIELD_NUMBER: _ClassVar[int]
    THERMAL_FIELD_NUMBER: _ClassVar[int]
    SENSORS_FIELD_NUMBER: _ClassVar[int]
    COMMS_FIELD_NUMBER: _ClassVar[int]
    MOBILITY_FIELD_NUMBER: _ClassVar[int]
    PROPULSION_FIELD_NUMBER: _ClassVar[int]
    RETURN_SPEC_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    SUBASSEMBLIES_FIELD_NUMBER: _ClassVar[int]
    FIDELITY_PROFILES_FIELD_NUMBER: _ClassVar[int]
    FLIGHT_STACKS_FIELD_NUMBER: _ClassVar[int]
    INTERFACES_FIELD_NUMBER: _ClassVar[int]
    PROVENANCE_FIELD_NUMBER: _ClassVar[int]
    identity: Identity
    capabilities: _containers.RepeatedScalarFieldContainer[str]
    core_interface_versions: _containers.ScalarMap[str, str]
    frames: _containers.RepeatedCompositeFieldContainer[Frame]
    root_frame: str
    geometry: _containers.RepeatedCompositeFieldContainer[GeometryRef]
    bodies: _containers.RepeatedCompositeFieldContainer[Body]
    joints: _containers.RepeatedCompositeFieldContainer[Joint]
    actuators: _containers.RepeatedCompositeFieldContainer[Actuator]
    power: PowerBudget
    thermal: ThermalBudget
    sensors: _containers.RepeatedCompositeFieldContainer[Sensor]
    comms: _containers.RepeatedCompositeFieldContainer[Comms]
    mobility: Mobility
    propulsion: Propulsion
    return_spec: ReturnSpec
    payload: PayloadSpec
    subassemblies: _containers.RepeatedCompositeFieldContainer[SubAssembly]
    fidelity_profiles: _containers.RepeatedCompositeFieldContainer[FidelityProfile]
    flight_stacks: _containers.RepeatedScalarFieldContainer[str]
    interfaces: Interfaces
    provenance: Provenance
    def __init__(self, identity: _Optional[_Union[Identity, _Mapping]] = ..., capabilities: _Optional[_Iterable[str]] = ..., core_interface_versions: _Optional[_Mapping[str, str]] = ..., frames: _Optional[_Iterable[_Union[Frame, _Mapping]]] = ..., root_frame: _Optional[str] = ..., geometry: _Optional[_Iterable[_Union[GeometryRef, _Mapping]]] = ..., bodies: _Optional[_Iterable[_Union[Body, _Mapping]]] = ..., joints: _Optional[_Iterable[_Union[Joint, _Mapping]]] = ..., actuators: _Optional[_Iterable[_Union[Actuator, _Mapping]]] = ..., power: _Optional[_Union[PowerBudget, _Mapping]] = ..., thermal: _Optional[_Union[ThermalBudget, _Mapping]] = ..., sensors: _Optional[_Iterable[_Union[Sensor, _Mapping]]] = ..., comms: _Optional[_Iterable[_Union[Comms, _Mapping]]] = ..., mobility: _Optional[_Union[Mobility, _Mapping]] = ..., propulsion: _Optional[_Union[Propulsion, _Mapping]] = ..., return_spec: _Optional[_Union[ReturnSpec, _Mapping]] = ..., payload: _Optional[_Union[PayloadSpec, _Mapping]] = ..., subassemblies: _Optional[_Iterable[_Union[SubAssembly, _Mapping]]] = ..., fidelity_profiles: _Optional[_Iterable[_Union[FidelityProfile, _Mapping]]] = ..., flight_stacks: _Optional[_Iterable[str]] = ..., interfaces: _Optional[_Union[Interfaces, _Mapping]] = ..., provenance: _Optional[_Union[Provenance, _Mapping]] = ...) -> None: ...

class SadfDocument(_message.Message):
    __slots__ = ("sadf_version", "asset")
    SADF_VERSION_FIELD_NUMBER: _ClassVar[int]
    ASSET_FIELD_NUMBER: _ClassVar[int]
    sadf_version: str
    asset: Asset
    def __init__(self, sadf_version: _Optional[str] = ..., asset: _Optional[_Union[Asset, _Mapping]] = ...) -> None: ...
