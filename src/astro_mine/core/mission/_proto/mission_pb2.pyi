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

class ManeuverBudget(_message.Message):
    __slots__ = ("total_delta_v_mps", "time_of_flight_s", "margin_mps")
    TOTAL_DELTA_V_MPS_FIELD_NUMBER: _ClassVar[int]
    TIME_OF_FLIGHT_S_FIELD_NUMBER: _ClassVar[int]
    MARGIN_MPS_FIELD_NUMBER: _ClassVar[int]
    total_delta_v_mps: float
    time_of_flight_s: float
    margin_mps: float
    def __init__(self, total_delta_v_mps: _Optional[float] = ..., time_of_flight_s: _Optional[float] = ..., margin_mps: _Optional[float] = ...) -> None: ...

class Maneuver(_message.Message):
    __slots__ = ("epoch_tdb_s", "delta_v_mps", "direction", "maneuver_type", "epoch")
    EPOCH_TDB_S_FIELD_NUMBER: _ClassVar[int]
    DELTA_V_MPS_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    MANEUVER_TYPE_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    epoch_tdb_s: float
    delta_v_mps: float
    direction: Vec3
    maneuver_type: str
    epoch: _units_pb2.Epoch
    def __init__(self, epoch_tdb_s: _Optional[float] = ..., delta_v_mps: _Optional[float] = ..., direction: _Optional[_Union[Vec3, _Mapping]] = ..., maneuver_type: _Optional[str] = ..., epoch: _Optional[_Union[_units_pb2.Epoch, _Mapping]] = ...) -> None: ...

class ReferenceState(_message.Message):
    __slots__ = ("epoch_tdb_s", "position_m", "velocity_mps", "epoch")
    EPOCH_TDB_S_FIELD_NUMBER: _ClassVar[int]
    POSITION_M_FIELD_NUMBER: _ClassVar[int]
    VELOCITY_MPS_FIELD_NUMBER: _ClassVar[int]
    EPOCH_FIELD_NUMBER: _ClassVar[int]
    epoch_tdb_s: float
    position_m: Vec3
    velocity_mps: Vec3
    epoch: _units_pb2.Epoch
    def __init__(self, epoch_tdb_s: _Optional[float] = ..., position_m: _Optional[_Union[Vec3, _Mapping]] = ..., velocity_mps: _Optional[_Union[Vec3, _Mapping]] = ..., epoch: _Optional[_Union[_units_pb2.Epoch, _Mapping]] = ...) -> None: ...

class TrajectorySegment(_message.Message):
    __slots__ = ("id", "start_epoch_tdb_s", "end_epoch_tdb_s", "kind", "window")
    ID_FIELD_NUMBER: _ClassVar[int]
    START_EPOCH_TDB_S_FIELD_NUMBER: _ClassVar[int]
    END_EPOCH_TDB_S_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    WINDOW_FIELD_NUMBER: _ClassVar[int]
    id: str
    start_epoch_tdb_s: float
    end_epoch_tdb_s: float
    kind: str
    window: _units_pb2.EpochWindow
    def __init__(self, id: _Optional[str] = ..., start_epoch_tdb_s: _Optional[float] = ..., end_epoch_tdb_s: _Optional[float] = ..., kind: _Optional[str] = ..., window: _Optional[_Union[_units_pb2.EpochWindow, _Mapping]] = ...) -> None: ...

class TrajectoryRef(_message.Message):
    __slots__ = ("id", "frame", "segments", "maneuvers", "reference_states", "feasibility_margin", "provenance", "frame_ref")
    ID_FIELD_NUMBER: _ClassVar[int]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    SEGMENTS_FIELD_NUMBER: _ClassVar[int]
    MANEUVERS_FIELD_NUMBER: _ClassVar[int]
    REFERENCE_STATES_FIELD_NUMBER: _ClassVar[int]
    FEASIBILITY_MARGIN_FIELD_NUMBER: _ClassVar[int]
    PROVENANCE_FIELD_NUMBER: _ClassVar[int]
    FRAME_REF_FIELD_NUMBER: _ClassVar[int]
    id: str
    frame: str
    segments: _containers.RepeatedCompositeFieldContainer[TrajectorySegment]
    maneuvers: _containers.RepeatedCompositeFieldContainer[Maneuver]
    reference_states: _containers.RepeatedCompositeFieldContainer[ReferenceState]
    feasibility_margin: float
    provenance: Provenance
    frame_ref: _units_pb2.ReferenceFrame
    def __init__(self, id: _Optional[str] = ..., frame: _Optional[str] = ..., segments: _Optional[_Iterable[_Union[TrajectorySegment, _Mapping]]] = ..., maneuvers: _Optional[_Iterable[_Union[Maneuver, _Mapping]]] = ..., reference_states: _Optional[_Iterable[_Union[ReferenceState, _Mapping]]] = ..., feasibility_margin: _Optional[float] = ..., provenance: _Optional[_Union[Provenance, _Mapping]] = ..., frame_ref: _Optional[_Union[_units_pb2.ReferenceFrame, _Mapping]] = ...) -> None: ...

class PhaseBoundary(_message.Message):
    __slots__ = ("condition", "state_ref")
    CONDITION_FIELD_NUMBER: _ClassVar[int]
    STATE_REF_FIELD_NUMBER: _ClassVar[int]
    condition: str
    state_ref: str
    def __init__(self, condition: _Optional[str] = ..., state_ref: _Optional[str] = ...) -> None: ...

class Leg(_message.Message):
    __slots__ = ("id", "trajectory_ref", "maneuver_budget")
    ID_FIELD_NUMBER: _ClassVar[int]
    TRAJECTORY_REF_FIELD_NUMBER: _ClassVar[int]
    MANEUVER_BUDGET_FIELD_NUMBER: _ClassVar[int]
    id: str
    trajectory_ref: TrajectoryRef
    maneuver_budget: ManeuverBudget
    def __init__(self, id: _Optional[str] = ..., trajectory_ref: _Optional[_Union[TrajectoryRef, _Mapping]] = ..., maneuver_budget: _Optional[_Union[ManeuverBudget, _Mapping]] = ...) -> None: ...

class PhaseTransition(_message.Message):
    __slots__ = ("from_phase", "to_phase", "terminal_state_ref", "initial_state_ref")
    FROM_PHASE_FIELD_NUMBER: _ClassVar[int]
    TO_PHASE_FIELD_NUMBER: _ClassVar[int]
    TERMINAL_STATE_REF_FIELD_NUMBER: _ClassVar[int]
    INITIAL_STATE_REF_FIELD_NUMBER: _ClassVar[int]
    from_phase: str
    to_phase: str
    terminal_state_ref: str
    initial_state_ref: str
    def __init__(self, from_phase: _Optional[str] = ..., to_phase: _Optional[str] = ..., terminal_state_ref: _Optional[str] = ..., initial_state_ref: _Optional[str] = ...) -> None: ...

class Phase(_message.Message):
    __slots__ = ("id", "regime", "environment_ref", "assets_active", "entry", "exit", "campaign_ref", "legs")
    ID_FIELD_NUMBER: _ClassVar[int]
    REGIME_FIELD_NUMBER: _ClassVar[int]
    ENVIRONMENT_REF_FIELD_NUMBER: _ClassVar[int]
    ASSETS_ACTIVE_FIELD_NUMBER: _ClassVar[int]
    ENTRY_FIELD_NUMBER: _ClassVar[int]
    EXIT_FIELD_NUMBER: _ClassVar[int]
    CAMPAIGN_REF_FIELD_NUMBER: _ClassVar[int]
    LEGS_FIELD_NUMBER: _ClassVar[int]
    id: str
    regime: str
    environment_ref: str
    assets_active: _containers.RepeatedScalarFieldContainer[str]
    entry: PhaseBoundary
    exit: PhaseBoundary
    campaign_ref: str
    legs: _containers.RepeatedCompositeFieldContainer[Leg]
    def __init__(self, id: _Optional[str] = ..., regime: _Optional[str] = ..., environment_ref: _Optional[str] = ..., assets_active: _Optional[_Iterable[str]] = ..., entry: _Optional[_Union[PhaseBoundary, _Mapping]] = ..., exit: _Optional[_Union[PhaseBoundary, _Mapping]] = ..., campaign_ref: _Optional[str] = ..., legs: _Optional[_Iterable[_Union[Leg, _Mapping]]] = ...) -> None: ...

class MissionConstraints(_message.Message):
    __slots__ = ("budget", "schedule_s", "launch_capacity_kg", "export_gated")
    BUDGET_FIELD_NUMBER: _ClassVar[int]
    SCHEDULE_S_FIELD_NUMBER: _ClassVar[int]
    LAUNCH_CAPACITY_KG_FIELD_NUMBER: _ClassVar[int]
    EXPORT_GATED_FIELD_NUMBER: _ClassVar[int]
    budget: float
    schedule_s: float
    launch_capacity_kg: float
    export_gated: bool
    def __init__(self, budget: _Optional[float] = ..., schedule_s: _Optional[float] = ..., launch_capacity_kg: _Optional[float] = ..., export_gated: _Optional[bool] = ...) -> None: ...

class MissionSpec(_message.Message):
    __slots__ = ("id", "name", "description", "phases", "fleet", "objective_ref", "constraints", "provenance")
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    PHASES_FIELD_NUMBER: _ClassVar[int]
    FLEET_FIELD_NUMBER: _ClassVar[int]
    OBJECTIVE_REF_FIELD_NUMBER: _ClassVar[int]
    CONSTRAINTS_FIELD_NUMBER: _ClassVar[int]
    PROVENANCE_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    description: str
    phases: _containers.RepeatedCompositeFieldContainer[Phase]
    fleet: _containers.RepeatedScalarFieldContainer[str]
    objective_ref: str
    constraints: MissionConstraints
    provenance: Provenance
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., description: _Optional[str] = ..., phases: _Optional[_Iterable[_Union[Phase, _Mapping]]] = ..., fleet: _Optional[_Iterable[str]] = ..., objective_ref: _Optional[str] = ..., constraints: _Optional[_Union[MissionConstraints, _Mapping]] = ..., provenance: _Optional[_Union[Provenance, _Mapping]] = ...) -> None: ...

class MissionDocument(_message.Message):
    __slots__ = ("mission_version", "mission")
    MISSION_VERSION_FIELD_NUMBER: _ClassVar[int]
    MISSION_FIELD_NUMBER: _ClassVar[int]
    mission_version: str
    mission: MissionSpec
    def __init__(self, mission_version: _Optional[str] = ..., mission: _Optional[_Union[MissionSpec, _Mapping]] = ...) -> None: ...
