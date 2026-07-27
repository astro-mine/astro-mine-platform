from astro_mine.core.units._proto import units_pb2 as _units_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SafetyDocument(_message.Message):
    __slots__ = ("safety_version", "safety")
    SAFETY_VERSION_FIELD_NUMBER: _ClassVar[int]
    SAFETY_FIELD_NUMBER: _ClassVar[int]
    safety_version: str
    safety: SafetySpec
    def __init__(self, safety_version: _Optional[str] = ..., safety: _Optional[_Union[SafetySpec, _Mapping]] = ...) -> None: ...

class SafetySpec(_message.Message):
    __slots__ = ("id", "name", "description", "scenario_ref", "signals", "constraints", "provenance", "safe_pose", "admissible_directives")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    SCENARIO_REF_FIELD_NUMBER: _ClassVar[int]
    SIGNALS_FIELD_NUMBER: _ClassVar[int]
    CONSTRAINTS_FIELD_NUMBER: _ClassVar[int]
    PROVENANCE_FIELD_NUMBER: _ClassVar[int]
    SAFE_POSE_FIELD_NUMBER: _ClassVar[int]
    ADMISSIBLE_DIRECTIVES_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    description: str
    scenario_ref: str
    signals: _containers.RepeatedCompositeFieldContainer[SignalRef]
    constraints: _containers.RepeatedCompositeFieldContainer[Constraint]
    provenance: Provenance
    safe_pose: SafePose
    admissible_directives: AdmissibleDirectives
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., description: _Optional[str] = ..., scenario_ref: _Optional[str] = ..., signals: _Optional[_Iterable[_Union[SignalRef, _Mapping]]] = ..., constraints: _Optional[_Iterable[_Union[Constraint, _Mapping]]] = ..., provenance: _Optional[_Union[Provenance, _Mapping]] = ..., safe_pose: _Optional[_Union[SafePose, _Mapping]] = ..., admissible_directives: _Optional[_Union[AdmissibleDirectives, _Mapping]] = ...) -> None: ...

class SafePose(_message.Message):
    __slots__ = ("frame", "position_m", "frame_ref")
    FRAME_FIELD_NUMBER: _ClassVar[int]
    POSITION_M_FIELD_NUMBER: _ClassVar[int]
    FRAME_REF_FIELD_NUMBER: _ClassVar[int]
    frame: str
    position_m: Vec3
    frame_ref: _units_pb2.ReferenceFrame
    def __init__(self, frame: _Optional[str] = ..., position_m: _Optional[_Union[Vec3, _Mapping]] = ..., frame_ref: _Optional[_Union[_units_pb2.ReferenceFrame, _Mapping]] = ...) -> None: ...

class AdmissibleDirectives(_message.Message):
    __slots__ = ("modes", "tasks")
    MODES_FIELD_NUMBER: _ClassVar[int]
    TASKS_FIELD_NUMBER: _ClassVar[int]
    modes: _containers.RepeatedScalarFieldContainer[str]
    tasks: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, modes: _Optional[_Iterable[str]] = ..., tasks: _Optional[_Iterable[str]] = ...) -> None: ...

class SignalRef(_message.Message):
    __slots__ = ("key", "unit", "source", "description")
    KEY_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    key: str
    unit: str
    source: str
    description: str
    def __init__(self, key: _Optional[str] = ..., unit: _Optional[str] = ..., source: _Optional[str] = ..., description: _Optional[str] = ...) -> None: ...

class Vec3(_message.Message):
    __slots__ = ("x", "y", "z")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    z: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ..., z: _Optional[float] = ...) -> None: ...

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

class Interval(_message.Message):
    __slots__ = ("lo", "hi")
    LO_FIELD_NUMBER: _ClassVar[int]
    HI_FIELD_NUMBER: _ClassVar[int]
    lo: float
    hi: float
    def __init__(self, lo: _Optional[float] = ..., hi: _Optional[float] = ...) -> None: ...

class KeepOutSphere(_message.Message):
    __slots__ = ("frame", "center_m", "radius_m", "frame_ref")
    FRAME_FIELD_NUMBER: _ClassVar[int]
    CENTER_M_FIELD_NUMBER: _ClassVar[int]
    RADIUS_M_FIELD_NUMBER: _ClassVar[int]
    FRAME_REF_FIELD_NUMBER: _ClassVar[int]
    frame: str
    center_m: Vec3
    radius_m: float
    frame_ref: _units_pb2.ReferenceFrame
    def __init__(self, frame: _Optional[str] = ..., center_m: _Optional[_Union[Vec3, _Mapping]] = ..., radius_m: _Optional[float] = ..., frame_ref: _Optional[_Union[_units_pb2.ReferenceFrame, _Mapping]] = ...) -> None: ...

class KeepOutHalfSpace(_message.Message):
    __slots__ = ("frame", "normal", "offset_m", "frame_ref")
    FRAME_FIELD_NUMBER: _ClassVar[int]
    NORMAL_FIELD_NUMBER: _ClassVar[int]
    OFFSET_M_FIELD_NUMBER: _ClassVar[int]
    FRAME_REF_FIELD_NUMBER: _ClassVar[int]
    frame: str
    normal: Vec3
    offset_m: float
    frame_ref: _units_pb2.ReferenceFrame
    def __init__(self, frame: _Optional[str] = ..., normal: _Optional[_Union[Vec3, _Mapping]] = ..., offset_m: _Optional[float] = ..., frame_ref: _Optional[_Union[_units_pb2.ReferenceFrame, _Mapping]] = ...) -> None: ...

class KeepOutVolume(_message.Message):
    __slots__ = ("shape", "box", "sphere", "half_space")
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    BOX_FIELD_NUMBER: _ClassVar[int]
    SPHERE_FIELD_NUMBER: _ClassVar[int]
    HALF_SPACE_FIELD_NUMBER: _ClassVar[int]
    shape: str
    box: Volume
    sphere: KeepOutSphere
    half_space: KeepOutHalfSpace
    def __init__(self, shape: _Optional[str] = ..., box: _Optional[_Union[Volume, _Mapping]] = ..., sphere: _Optional[_Union[KeepOutSphere, _Mapping]] = ..., half_space: _Optional[_Union[KeepOutHalfSpace, _Mapping]] = ...) -> None: ...

class STLFormula(_message.Message):
    __slots__ = ("op", "signal", "cmp", "threshold", "interval_s", "args")
    OP_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_FIELD_NUMBER: _ClassVar[int]
    CMP_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    INTERVAL_S_FIELD_NUMBER: _ClassVar[int]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    op: str
    signal: str
    cmp: str
    threshold: float
    interval_s: Interval
    args: _containers.RepeatedCompositeFieldContainer[STLFormula]
    def __init__(self, op: _Optional[str] = ..., signal: _Optional[str] = ..., cmp: _Optional[str] = ..., threshold: _Optional[float] = ..., interval_s: _Optional[_Union[Interval, _Mapping]] = ..., args: _Optional[_Iterable[_Union[STLFormula, _Mapping]]] = ...) -> None: ...

class KeepOutConstraint(_message.Message):
    __slots__ = ("volume", "margin_m", "collision_pair")
    VOLUME_FIELD_NUMBER: _ClassVar[int]
    MARGIN_M_FIELD_NUMBER: _ClassVar[int]
    COLLISION_PAIR_FIELD_NUMBER: _ClassVar[int]
    volume: KeepOutVolume
    margin_m: float
    collision_pair: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, volume: _Optional[_Union[KeepOutVolume, _Mapping]] = ..., margin_m: _Optional[float] = ..., collision_pair: _Optional[_Iterable[str]] = ...) -> None: ...

class PowerFloorConstraint(_message.Message):
    __slots__ = ("signal", "floor_w")
    SIGNAL_FIELD_NUMBER: _ClassVar[int]
    FLOOR_W_FIELD_NUMBER: _ClassVar[int]
    signal: str
    floor_w: float
    def __init__(self, signal: _Optional[str] = ..., floor_w: _Optional[float] = ...) -> None: ...

class EnergyFloorConstraint(_message.Message):
    __slots__ = ("signal", "floor_j")
    SIGNAL_FIELD_NUMBER: _ClassVar[int]
    FLOOR_J_FIELD_NUMBER: _ClassVar[int]
    signal: str
    floor_j: float
    def __init__(self, signal: _Optional[str] = ..., floor_j: _Optional[float] = ...) -> None: ...

class ThermalCeilingConstraint(_message.Message):
    __slots__ = ("signal", "limit_k")
    SIGNAL_FIELD_NUMBER: _ClassVar[int]
    LIMIT_K_FIELD_NUMBER: _ClassVar[int]
    signal: str
    limit_k: float
    def __init__(self, signal: _Optional[str] = ..., limit_k: _Optional[float] = ...) -> None: ...

class ThermalFloorConstraint(_message.Message):
    __slots__ = ("signal", "limit_k")
    SIGNAL_FIELD_NUMBER: _ClassVar[int]
    LIMIT_K_FIELD_NUMBER: _ClassVar[int]
    signal: str
    limit_k: float
    def __init__(self, signal: _Optional[str] = ..., limit_k: _Optional[float] = ...) -> None: ...

class TorqueCeilingConstraint(_message.Message):
    __slots__ = ("signal", "max_nm")
    SIGNAL_FIELD_NUMBER: _ClassVar[int]
    MAX_NM_FIELD_NUMBER: _ClassVar[int]
    signal: str
    max_nm: float
    def __init__(self, signal: _Optional[str] = ..., max_nm: _Optional[float] = ...) -> None: ...

class KinematicLimitConstraint(_message.Message):
    __slots__ = ("signal", "max_velocity_mps", "max_accel_mps2")
    SIGNAL_FIELD_NUMBER: _ClassVar[int]
    MAX_VELOCITY_MPS_FIELD_NUMBER: _ClassVar[int]
    MAX_ACCEL_MPS2_FIELD_NUMBER: _ClassVar[int]
    signal: str
    max_velocity_mps: float
    max_accel_mps2: float
    def __init__(self, signal: _Optional[str] = ..., max_velocity_mps: _Optional[float] = ..., max_accel_mps2: _Optional[float] = ...) -> None: ...

class TemporalConstraint(_message.Message):
    __slots__ = ("formula",)
    FORMULA_FIELD_NUMBER: _ClassVar[int]
    formula: STLFormula
    def __init__(self, formula: _Optional[_Union[STLFormula, _Mapping]] = ...) -> None: ...

class Constraint(_message.Message):
    __slots__ = ("kind", "id", "on_uncertain", "description", "keep_out", "power_floor", "energy_floor", "thermal_ceiling", "thermal_floor", "torque_ceiling", "kinematic_limit", "temporal")
    KIND_FIELD_NUMBER: _ClassVar[int]
    ID_FIELD_NUMBER: _ClassVar[int]
    ON_UNCERTAIN_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    KEEP_OUT_FIELD_NUMBER: _ClassVar[int]
    POWER_FLOOR_FIELD_NUMBER: _ClassVar[int]
    ENERGY_FLOOR_FIELD_NUMBER: _ClassVar[int]
    THERMAL_CEILING_FIELD_NUMBER: _ClassVar[int]
    THERMAL_FLOOR_FIELD_NUMBER: _ClassVar[int]
    TORQUE_CEILING_FIELD_NUMBER: _ClassVar[int]
    KINEMATIC_LIMIT_FIELD_NUMBER: _ClassVar[int]
    TEMPORAL_FIELD_NUMBER: _ClassVar[int]
    kind: str
    id: str
    on_uncertain: str
    description: str
    keep_out: KeepOutConstraint
    power_floor: PowerFloorConstraint
    energy_floor: EnergyFloorConstraint
    thermal_ceiling: ThermalCeilingConstraint
    thermal_floor: ThermalFloorConstraint
    torque_ceiling: TorqueCeilingConstraint
    kinematic_limit: KinematicLimitConstraint
    temporal: TemporalConstraint
    def __init__(self, kind: _Optional[str] = ..., id: _Optional[str] = ..., on_uncertain: _Optional[str] = ..., description: _Optional[str] = ..., keep_out: _Optional[_Union[KeepOutConstraint, _Mapping]] = ..., power_floor: _Optional[_Union[PowerFloorConstraint, _Mapping]] = ..., energy_floor: _Optional[_Union[EnergyFloorConstraint, _Mapping]] = ..., thermal_ceiling: _Optional[_Union[ThermalCeilingConstraint, _Mapping]] = ..., thermal_floor: _Optional[_Union[ThermalFloorConstraint, _Mapping]] = ..., torque_ceiling: _Optional[_Union[TorqueCeilingConstraint, _Mapping]] = ..., kinematic_limit: _Optional[_Union[KinematicLimitConstraint, _Mapping]] = ..., temporal: _Optional[_Union[TemporalConstraint, _Mapping]] = ...) -> None: ...

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
