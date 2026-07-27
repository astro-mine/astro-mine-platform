from astro_mine.core.units._proto import units_pb2 as _units_pb2
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

class Volume(_message.Message):
    __slots__ = ("frame", "center_m", "dimensions_m", "frame_ref")
    FRAME_FIELD_NUMBER: _ClassVar[int]
    CENTER_M_FIELD_NUMBER: _ClassVar[int]
    DIMENSIONS_M_FIELD_NUMBER: _ClassVar[int]
    FRAME_REF_FIELD_NUMBER: _ClassVar[int]
    frame: str
    center_m: Vec3
    dimensions_m: Vec3
    frame_ref: _units_pb2.ReferenceFrame
    def __init__(self, frame: _Optional[str] = ..., center_m: _Optional[_Union[Vec3, _Mapping]] = ..., dimensions_m: _Optional[_Union[Vec3, _Mapping]] = ..., frame_ref: _Optional[_Union[_units_pb2.ReferenceFrame, _Mapping]] = ...) -> None: ...

class ActuatorCommand(_message.Message):
    __slots__ = ("target", "control_mode", "setpoint", "unit", "stiffness", "damping", "feedforward", "trajectory_ref")
    TARGET_FIELD_NUMBER: _ClassVar[int]
    CONTROL_MODE_FIELD_NUMBER: _ClassVar[int]
    SETPOINT_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    STIFFNESS_FIELD_NUMBER: _ClassVar[int]
    DAMPING_FIELD_NUMBER: _ClassVar[int]
    FEEDFORWARD_FIELD_NUMBER: _ClassVar[int]
    TRAJECTORY_REF_FIELD_NUMBER: _ClassVar[int]
    target: str
    control_mode: str
    setpoint: _containers.RepeatedScalarFieldContainer[float]
    unit: str
    stiffness: float
    damping: float
    feedforward: _containers.RepeatedScalarFieldContainer[float]
    trajectory_ref: str
    def __init__(self, target: _Optional[str] = ..., control_mode: _Optional[str] = ..., setpoint: _Optional[_Iterable[float]] = ..., unit: _Optional[str] = ..., stiffness: _Optional[float] = ..., damping: _Optional[float] = ..., feedforward: _Optional[_Iterable[float]] = ..., trajectory_ref: _Optional[str] = ...) -> None: ...

class ModeCommand(_message.Message):
    __slots__ = ("mode", "params")
    class ParamsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    MODE_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    mode: str
    params: _containers.ScalarMap[str, str]
    def __init__(self, mode: _Optional[str] = ..., params: _Optional[_Mapping[str, str]] = ...) -> None: ...

class GotoTask(_message.Message):
    __slots__ = ("target_frame", "target_pose", "position_tolerance_m", "heading_tolerance_rad", "max_speed_mps", "target_frame_ref")
    TARGET_FRAME_FIELD_NUMBER: _ClassVar[int]
    TARGET_POSE_FIELD_NUMBER: _ClassVar[int]
    POSITION_TOLERANCE_M_FIELD_NUMBER: _ClassVar[int]
    HEADING_TOLERANCE_RAD_FIELD_NUMBER: _ClassVar[int]
    MAX_SPEED_MPS_FIELD_NUMBER: _ClassVar[int]
    TARGET_FRAME_REF_FIELD_NUMBER: _ClassVar[int]
    target_frame: str
    target_pose: Transform
    position_tolerance_m: float
    heading_tolerance_rad: float
    max_speed_mps: float
    target_frame_ref: _units_pb2.ReferenceFrame
    def __init__(self, target_frame: _Optional[str] = ..., target_pose: _Optional[_Union[Transform, _Mapping]] = ..., position_tolerance_m: _Optional[float] = ..., heading_tolerance_rad: _Optional[float] = ..., max_speed_mps: _Optional[float] = ..., target_frame_ref: _Optional[_Union[_units_pb2.ReferenceFrame, _Mapping]] = ...) -> None: ...

class SampleTask(_message.Message):
    __slots__ = ("site_frame", "target_point_m", "method", "depth_m", "sample_mass_kg", "site_frame_ref")
    SITE_FRAME_FIELD_NUMBER: _ClassVar[int]
    TARGET_POINT_M_FIELD_NUMBER: _ClassVar[int]
    METHOD_FIELD_NUMBER: _ClassVar[int]
    DEPTH_M_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_MASS_KG_FIELD_NUMBER: _ClassVar[int]
    SITE_FRAME_REF_FIELD_NUMBER: _ClassVar[int]
    site_frame: str
    target_point_m: Vec3
    method: str
    depth_m: float
    sample_mass_kg: float
    site_frame_ref: _units_pb2.ReferenceFrame
    def __init__(self, site_frame: _Optional[str] = ..., target_point_m: _Optional[_Union[Vec3, _Mapping]] = ..., method: _Optional[str] = ..., depth_m: _Optional[float] = ..., sample_mass_kg: _Optional[float] = ..., site_frame_ref: _Optional[_Union[_units_pb2.ReferenceFrame, _Mapping]] = ...) -> None: ...

class ExcavateTask(_message.Message):
    __slots__ = ("region", "tool", "pattern", "target_volume_m3")
    REGION_FIELD_NUMBER: _ClassVar[int]
    TOOL_FIELD_NUMBER: _ClassVar[int]
    PATTERN_FIELD_NUMBER: _ClassVar[int]
    TARGET_VOLUME_M3_FIELD_NUMBER: _ClassVar[int]
    region: Volume
    tool: str
    pattern: str
    target_volume_m3: float
    def __init__(self, region: _Optional[_Union[Volume, _Mapping]] = ..., tool: _Optional[str] = ..., pattern: _Optional[str] = ..., target_volume_m3: _Optional[float] = ...) -> None: ...

class HaulTask(_message.Message):
    __slots__ = ("from_frame", "to_frame", "payload_kg", "resource_species", "from_frame_ref", "to_frame_ref")
    FROM_FRAME_FIELD_NUMBER: _ClassVar[int]
    TO_FRAME_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_KG_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_SPECIES_FIELD_NUMBER: _ClassVar[int]
    FROM_FRAME_REF_FIELD_NUMBER: _ClassVar[int]
    TO_FRAME_REF_FIELD_NUMBER: _ClassVar[int]
    from_frame: str
    to_frame: str
    payload_kg: float
    resource_species: str
    from_frame_ref: _units_pb2.ReferenceFrame
    to_frame_ref: _units_pb2.ReferenceFrame
    def __init__(self, from_frame: _Optional[str] = ..., to_frame: _Optional[str] = ..., payload_kg: _Optional[float] = ..., resource_species: _Optional[str] = ..., from_frame_ref: _Optional[_Union[_units_pb2.ReferenceFrame, _Mapping]] = ..., to_frame_ref: _Optional[_Union[_units_pb2.ReferenceFrame, _Mapping]] = ...) -> None: ...

class DockTask(_message.Message):
    __slots__ = ("target_asset_id", "port_name", "approach_frame", "approach_frame_ref")
    TARGET_ASSET_ID_FIELD_NUMBER: _ClassVar[int]
    PORT_NAME_FIELD_NUMBER: _ClassVar[int]
    APPROACH_FRAME_FIELD_NUMBER: _ClassVar[int]
    APPROACH_FRAME_REF_FIELD_NUMBER: _ClassVar[int]
    target_asset_id: str
    port_name: str
    approach_frame: str
    approach_frame_ref: _units_pb2.ReferenceFrame
    def __init__(self, target_asset_id: _Optional[str] = ..., port_name: _Optional[str] = ..., approach_frame: _Optional[str] = ..., approach_frame_ref: _Optional[_Union[_units_pb2.ReferenceFrame, _Mapping]] = ...) -> None: ...

class HopTask(_message.Message):
    __slots__ = ("launch_frame", "target_point_m", "range_m", "apoapsis_m", "launch_frame_ref")
    LAUNCH_FRAME_FIELD_NUMBER: _ClassVar[int]
    TARGET_POINT_M_FIELD_NUMBER: _ClassVar[int]
    RANGE_M_FIELD_NUMBER: _ClassVar[int]
    APOAPSIS_M_FIELD_NUMBER: _ClassVar[int]
    LAUNCH_FRAME_REF_FIELD_NUMBER: _ClassVar[int]
    launch_frame: str
    target_point_m: Vec3
    range_m: float
    apoapsis_m: float
    launch_frame_ref: _units_pb2.ReferenceFrame
    def __init__(self, launch_frame: _Optional[str] = ..., target_point_m: _Optional[_Union[Vec3, _Mapping]] = ..., range_m: _Optional[float] = ..., apoapsis_m: _Optional[float] = ..., launch_frame_ref: _Optional[_Union[_units_pb2.ReferenceFrame, _Mapping]] = ...) -> None: ...

class ChargeTask(_message.Message):
    __slots__ = ("source", "target_soc_j", "max_power_w")
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    TARGET_SOC_J_FIELD_NUMBER: _ClassVar[int]
    MAX_POWER_W_FIELD_NUMBER: _ClassVar[int]
    source: str
    target_soc_j: float
    max_power_w: float
    def __init__(self, source: _Optional[str] = ..., target_soc_j: _Optional[float] = ..., max_power_w: _Optional[float] = ...) -> None: ...

class ProspectTask(_message.Message):
    __slots__ = ("region", "sensor_kinds", "info_gain_target")
    REGION_FIELD_NUMBER: _ClassVar[int]
    SENSOR_KINDS_FIELD_NUMBER: _ClassVar[int]
    INFO_GAIN_TARGET_FIELD_NUMBER: _ClassVar[int]
    region: Volume
    sensor_kinds: _containers.RepeatedScalarFieldContainer[str]
    info_gain_target: float
    def __init__(self, region: _Optional[_Union[Volume, _Mapping]] = ..., sensor_kinds: _Optional[_Iterable[str]] = ..., info_gain_target: _Optional[float] = ...) -> None: ...

class TaskDirective(_message.Message):
    __slots__ = ("task_kind", "priority", "deadline_s", "goto", "sample", "excavate", "haul", "dock", "hop", "charge", "prospect", "directive", "params")
    class ParamsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    TASK_KIND_FIELD_NUMBER: _ClassVar[int]
    PRIORITY_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_S_FIELD_NUMBER: _ClassVar[int]
    GOTO_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_FIELD_NUMBER: _ClassVar[int]
    EXCAVATE_FIELD_NUMBER: _ClassVar[int]
    HAUL_FIELD_NUMBER: _ClassVar[int]
    DOCK_FIELD_NUMBER: _ClassVar[int]
    HOP_FIELD_NUMBER: _ClassVar[int]
    CHARGE_FIELD_NUMBER: _ClassVar[int]
    PROSPECT_FIELD_NUMBER: _ClassVar[int]
    DIRECTIVE_FIELD_NUMBER: _ClassVar[int]
    PARAMS_FIELD_NUMBER: _ClassVar[int]
    task_kind: str
    priority: int
    deadline_s: float
    goto: GotoTask
    sample: SampleTask
    excavate: ExcavateTask
    haul: HaulTask
    dock: DockTask
    hop: HopTask
    charge: ChargeTask
    prospect: ProspectTask
    directive: str
    params: _containers.ScalarMap[str, str]
    def __init__(self, task_kind: _Optional[str] = ..., priority: _Optional[int] = ..., deadline_s: _Optional[float] = ..., goto: _Optional[_Union[GotoTask, _Mapping]] = ..., sample: _Optional[_Union[SampleTask, _Mapping]] = ..., excavate: _Optional[_Union[ExcavateTask, _Mapping]] = ..., haul: _Optional[_Union[HaulTask, _Mapping]] = ..., dock: _Optional[_Union[DockTask, _Mapping]] = ..., hop: _Optional[_Union[HopTask, _Mapping]] = ..., charge: _Optional[_Union[ChargeTask, _Mapping]] = ..., prospect: _Optional[_Union[ProspectTask, _Mapping]] = ..., directive: _Optional[str] = ..., params: _Optional[_Mapping[str, str]] = ...) -> None: ...

class Action(_message.Message):
    __slots__ = ("agent_id", "kind", "sim_time_s", "actuator", "mode", "task")
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    SIM_TIME_S_FIELD_NUMBER: _ClassVar[int]
    ACTUATOR_FIELD_NUMBER: _ClassVar[int]
    MODE_FIELD_NUMBER: _ClassVar[int]
    TASK_FIELD_NUMBER: _ClassVar[int]
    agent_id: str
    kind: str
    sim_time_s: float
    actuator: ActuatorCommand
    mode: ModeCommand
    task: TaskDirective
    def __init__(self, agent_id: _Optional[str] = ..., kind: _Optional[str] = ..., sim_time_s: _Optional[float] = ..., actuator: _Optional[_Union[ActuatorCommand, _Mapping]] = ..., mode: _Optional[_Union[ModeCommand, _Mapping]] = ..., task: _Optional[_Union[TaskDirective, _Mapping]] = ...) -> None: ...

class ActionBatch(_message.Message):
    __slots__ = ("actions",)
    ACTIONS_FIELD_NUMBER: _ClassVar[int]
    actions: _containers.RepeatedCompositeFieldContainer[Action]
    def __init__(self, actions: _Optional[_Iterable[_Union[Action, _Mapping]]] = ...) -> None: ...

class LinkBudget(_message.Message):
    __slots__ = ("eirp_dbw", "path_loss_db", "gt_db_per_k", "required_ebn0_db", "margin_db", "modcod")
    EIRP_DBW_FIELD_NUMBER: _ClassVar[int]
    PATH_LOSS_DB_FIELD_NUMBER: _ClassVar[int]
    GT_DB_PER_K_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_EBN0_DB_FIELD_NUMBER: _ClassVar[int]
    MARGIN_DB_FIELD_NUMBER: _ClassVar[int]
    MODCOD_FIELD_NUMBER: _ClassVar[int]
    eirp_dbw: float
    path_loss_db: float
    gt_db_per_k: float
    required_ebn0_db: float
    margin_db: float
    modcod: str
    def __init__(self, eirp_dbw: _Optional[float] = ..., path_loss_db: _Optional[float] = ..., gt_db_per_k: _Optional[float] = ..., required_ebn0_db: _Optional[float] = ..., margin_db: _Optional[float] = ..., modcod: _Optional[str] = ...) -> None: ...

class ContactNode(_message.Message):
    __slots__ = ("id", "role", "kind")
    ID_FIELD_NUMBER: _ClassVar[int]
    ROLE_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    id: str
    role: str
    kind: str
    def __init__(self, id: _Optional[str] = ..., role: _Optional[str] = ..., kind: _Optional[str] = ...) -> None: ...

class ContactInterval(_message.Message):
    __slots__ = ("node_a", "node_b", "start_tdb_s", "end_tdb_s", "max_rate_bps", "min_latency_s", "mean_latency_s", "margin_db", "confidence", "band", "modcod", "link_budget", "window")
    NODE_A_FIELD_NUMBER: _ClassVar[int]
    NODE_B_FIELD_NUMBER: _ClassVar[int]
    START_TDB_S_FIELD_NUMBER: _ClassVar[int]
    END_TDB_S_FIELD_NUMBER: _ClassVar[int]
    MAX_RATE_BPS_FIELD_NUMBER: _ClassVar[int]
    MIN_LATENCY_S_FIELD_NUMBER: _ClassVar[int]
    MEAN_LATENCY_S_FIELD_NUMBER: _ClassVar[int]
    MARGIN_DB_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    BAND_FIELD_NUMBER: _ClassVar[int]
    MODCOD_FIELD_NUMBER: _ClassVar[int]
    LINK_BUDGET_FIELD_NUMBER: _ClassVar[int]
    WINDOW_FIELD_NUMBER: _ClassVar[int]
    node_a: str
    node_b: str
    start_tdb_s: float
    end_tdb_s: float
    max_rate_bps: float
    min_latency_s: float
    mean_latency_s: float
    margin_db: float
    confidence: str
    band: str
    modcod: str
    link_budget: LinkBudget
    window: _units_pb2.EpochWindow
    def __init__(self, node_a: _Optional[str] = ..., node_b: _Optional[str] = ..., start_tdb_s: _Optional[float] = ..., end_tdb_s: _Optional[float] = ..., max_rate_bps: _Optional[float] = ..., min_latency_s: _Optional[float] = ..., mean_latency_s: _Optional[float] = ..., margin_db: _Optional[float] = ..., confidence: _Optional[str] = ..., band: _Optional[str] = ..., modcod: _Optional[str] = ..., link_budget: _Optional[_Union[LinkBudget, _Mapping]] = ..., window: _Optional[_Union[_units_pb2.EpochWindow, _Mapping]] = ...) -> None: ...

class Route(_message.Message):
    __slots__ = ("source", "dest", "hops", "store_and_forward", "earliest_delivery_tdb_s", "total_latency_s", "earliest_delivery")
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    DEST_FIELD_NUMBER: _ClassVar[int]
    HOPS_FIELD_NUMBER: _ClassVar[int]
    STORE_AND_FORWARD_FIELD_NUMBER: _ClassVar[int]
    EARLIEST_DELIVERY_TDB_S_FIELD_NUMBER: _ClassVar[int]
    TOTAL_LATENCY_S_FIELD_NUMBER: _ClassVar[int]
    EARLIEST_DELIVERY_FIELD_NUMBER: _ClassVar[int]
    source: str
    dest: str
    hops: _containers.RepeatedScalarFieldContainer[str]
    store_and_forward: bool
    earliest_delivery_tdb_s: float
    total_latency_s: float
    earliest_delivery: _units_pb2.Epoch
    def __init__(self, source: _Optional[str] = ..., dest: _Optional[str] = ..., hops: _Optional[_Iterable[str]] = ..., store_and_forward: _Optional[bool] = ..., earliest_delivery_tdb_s: _Optional[float] = ..., total_latency_s: _Optional[float] = ..., earliest_delivery: _Optional[_Union[_units_pb2.Epoch, _Mapping]] = ...) -> None: ...

class ContactPlan(_message.Message):
    __slots__ = ("nodes", "intervals", "routes", "epoch_start_tdb_s", "epoch_end_tdb_s", "window")
    NODES_FIELD_NUMBER: _ClassVar[int]
    INTERVALS_FIELD_NUMBER: _ClassVar[int]
    ROUTES_FIELD_NUMBER: _ClassVar[int]
    EPOCH_START_TDB_S_FIELD_NUMBER: _ClassVar[int]
    EPOCH_END_TDB_S_FIELD_NUMBER: _ClassVar[int]
    WINDOW_FIELD_NUMBER: _ClassVar[int]
    nodes: _containers.RepeatedCompositeFieldContainer[ContactNode]
    intervals: _containers.RepeatedCompositeFieldContainer[ContactInterval]
    routes: _containers.RepeatedCompositeFieldContainer[Route]
    epoch_start_tdb_s: float
    epoch_end_tdb_s: float
    window: _units_pb2.EpochWindow
    def __init__(self, nodes: _Optional[_Iterable[_Union[ContactNode, _Mapping]]] = ..., intervals: _Optional[_Iterable[_Union[ContactInterval, _Mapping]]] = ..., routes: _Optional[_Iterable[_Union[Route, _Mapping]]] = ..., epoch_start_tdb_s: _Optional[float] = ..., epoch_end_tdb_s: _Optional[float] = ..., window: _Optional[_Union[_units_pb2.EpochWindow, _Mapping]] = ...) -> None: ...
