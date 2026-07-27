from astro_mine.core.units._proto import units_pb2 as _units_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CompiledSafetyModel(_message.Message):
    __slots__ = ("compiled_version", "spec_id", "spec_content_hash", "sample_period_s", "predicate_table", "scalar_bounds", "keep_out_terms", "monitors", "resource_bounds", "safe_pose", "action_limits", "admissible_directives")
    COMPILED_VERSION_FIELD_NUMBER: _ClassVar[int]
    SPEC_ID_FIELD_NUMBER: _ClassVar[int]
    SPEC_CONTENT_HASH_FIELD_NUMBER: _ClassVar[int]
    SAMPLE_PERIOD_S_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_TABLE_FIELD_NUMBER: _ClassVar[int]
    SCALAR_BOUNDS_FIELD_NUMBER: _ClassVar[int]
    KEEP_OUT_TERMS_FIELD_NUMBER: _ClassVar[int]
    MONITORS_FIELD_NUMBER: _ClassVar[int]
    RESOURCE_BOUNDS_FIELD_NUMBER: _ClassVar[int]
    SAFE_POSE_FIELD_NUMBER: _ClassVar[int]
    ACTION_LIMITS_FIELD_NUMBER: _ClassVar[int]
    ADMISSIBLE_DIRECTIVES_FIELD_NUMBER: _ClassVar[int]
    compiled_version: str
    spec_id: str
    spec_content_hash: str
    sample_period_s: float
    predicate_table: PredicateTable
    scalar_bounds: _containers.RepeatedCompositeFieldContainer[ScalarBound]
    keep_out_terms: _containers.RepeatedCompositeFieldContainer[KeepOutTerm]
    monitors: _containers.RepeatedCompositeFieldContainer[MonitorAutomaton]
    resource_bounds: ResourceBounds
    safe_pose: CompiledSafePose
    action_limits: ActionLimits
    admissible_directives: CompiledAdmissibleDirectives
    def __init__(self, compiled_version: _Optional[str] = ..., spec_id: _Optional[str] = ..., spec_content_hash: _Optional[str] = ..., sample_period_s: _Optional[float] = ..., predicate_table: _Optional[_Union[PredicateTable, _Mapping]] = ..., scalar_bounds: _Optional[_Iterable[_Union[ScalarBound, _Mapping]]] = ..., keep_out_terms: _Optional[_Iterable[_Union[KeepOutTerm, _Mapping]]] = ..., monitors: _Optional[_Iterable[_Union[MonitorAutomaton, _Mapping]]] = ..., resource_bounds: _Optional[_Union[ResourceBounds, _Mapping]] = ..., safe_pose: _Optional[_Union[CompiledSafePose, _Mapping]] = ..., action_limits: _Optional[_Union[ActionLimits, _Mapping]] = ..., admissible_directives: _Optional[_Union[CompiledAdmissibleDirectives, _Mapping]] = ...) -> None: ...

class CompiledAdmissibleDirectives(_message.Message):
    __slots__ = ("modes", "tasks")
    MODES_FIELD_NUMBER: _ClassVar[int]
    TASKS_FIELD_NUMBER: _ClassVar[int]
    modes: _containers.RepeatedScalarFieldContainer[str]
    tasks: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, modes: _Optional[_Iterable[str]] = ..., tasks: _Optional[_Iterable[str]] = ...) -> None: ...

class ActionLimits(_message.Message):
    __slots__ = ("max_velocity_mps", "max_accel_mps2")
    MAX_VELOCITY_MPS_FIELD_NUMBER: _ClassVar[int]
    MAX_ACCEL_MPS2_FIELD_NUMBER: _ClassVar[int]
    max_velocity_mps: float
    max_accel_mps2: float
    def __init__(self, max_velocity_mps: _Optional[float] = ..., max_accel_mps2: _Optional[float] = ...) -> None: ...

class CompiledSafePose(_message.Message):
    __slots__ = ("frame", "position", "frame_ref")
    FRAME_FIELD_NUMBER: _ClassVar[int]
    POSITION_FIELD_NUMBER: _ClassVar[int]
    FRAME_REF_FIELD_NUMBER: _ClassVar[int]
    frame: str
    position: _containers.RepeatedScalarFieldContainer[float]
    frame_ref: _units_pb2.ReferenceFrame
    def __init__(self, frame: _Optional[str] = ..., position: _Optional[_Iterable[float]] = ..., frame_ref: _Optional[_Union[_units_pb2.ReferenceFrame, _Mapping]] = ...) -> None: ...

class PredicateAtom(_message.Message):
    __slots__ = ("op", "signal_index", "threshold")
    OP_FIELD_NUMBER: _ClassVar[int]
    SIGNAL_INDEX_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    op: str
    signal_index: int
    threshold: float
    def __init__(self, op: _Optional[str] = ..., signal_index: _Optional[int] = ..., threshold: _Optional[float] = ...) -> None: ...

class PredicateTable(_message.Message):
    __slots__ = ("signals", "atoms")
    SIGNALS_FIELD_NUMBER: _ClassVar[int]
    ATOMS_FIELD_NUMBER: _ClassVar[int]
    signals: _containers.RepeatedScalarFieldContainer[str]
    atoms: _containers.RepeatedCompositeFieldContainer[PredicateAtom]
    def __init__(self, signals: _Optional[_Iterable[str]] = ..., atoms: _Optional[_Iterable[_Union[PredicateAtom, _Mapping]]] = ...) -> None: ...

class ScalarBound(_message.Message):
    __slots__ = ("constraint_id", "on_uncertain", "atom_index")
    CONSTRAINT_ID_FIELD_NUMBER: _ClassVar[int]
    ON_UNCERTAIN_FIELD_NUMBER: _ClassVar[int]
    ATOM_INDEX_FIELD_NUMBER: _ClassVar[int]
    constraint_id: str
    on_uncertain: str
    atom_index: int
    def __init__(self, constraint_id: _Optional[str] = ..., on_uncertain: _Optional[str] = ..., atom_index: _Optional[int] = ...) -> None: ...

class KeepOutTerm(_message.Message):
    __slots__ = ("constraint_id", "on_uncertain", "shape", "frame", "margin_m", "center", "half_extents", "radius", "normal", "offset", "collision_pair", "frame_ref")
    CONSTRAINT_ID_FIELD_NUMBER: _ClassVar[int]
    ON_UNCERTAIN_FIELD_NUMBER: _ClassVar[int]
    SHAPE_FIELD_NUMBER: _ClassVar[int]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    MARGIN_M_FIELD_NUMBER: _ClassVar[int]
    CENTER_FIELD_NUMBER: _ClassVar[int]
    HALF_EXTENTS_FIELD_NUMBER: _ClassVar[int]
    RADIUS_FIELD_NUMBER: _ClassVar[int]
    NORMAL_FIELD_NUMBER: _ClassVar[int]
    OFFSET_FIELD_NUMBER: _ClassVar[int]
    COLLISION_PAIR_FIELD_NUMBER: _ClassVar[int]
    FRAME_REF_FIELD_NUMBER: _ClassVar[int]
    constraint_id: str
    on_uncertain: str
    shape: str
    frame: str
    margin_m: float
    center: _containers.RepeatedScalarFieldContainer[float]
    half_extents: _containers.RepeatedScalarFieldContainer[float]
    radius: float
    normal: _containers.RepeatedScalarFieldContainer[float]
    offset: float
    collision_pair: _containers.RepeatedScalarFieldContainer[str]
    frame_ref: _units_pb2.ReferenceFrame
    def __init__(self, constraint_id: _Optional[str] = ..., on_uncertain: _Optional[str] = ..., shape: _Optional[str] = ..., frame: _Optional[str] = ..., margin_m: _Optional[float] = ..., center: _Optional[_Iterable[float]] = ..., half_extents: _Optional[_Iterable[float]] = ..., radius: _Optional[float] = ..., normal: _Optional[_Iterable[float]] = ..., offset: _Optional[float] = ..., collision_pair: _Optional[_Iterable[str]] = ..., frame_ref: _Optional[_Union[_units_pb2.ReferenceFrame, _Mapping]] = ...) -> None: ...

class CompiledNode(_message.Message):
    __slots__ = ("op", "predicate_index", "interval_lo_samples", "interval_hi_samples", "args")
    OP_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_INDEX_FIELD_NUMBER: _ClassVar[int]
    INTERVAL_LO_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    INTERVAL_HI_SAMPLES_FIELD_NUMBER: _ClassVar[int]
    ARGS_FIELD_NUMBER: _ClassVar[int]
    op: str
    predicate_index: int
    interval_lo_samples: int
    interval_hi_samples: int
    args: _containers.RepeatedCompositeFieldContainer[CompiledNode]
    def __init__(self, op: _Optional[str] = ..., predicate_index: _Optional[int] = ..., interval_lo_samples: _Optional[int] = ..., interval_hi_samples: _Optional[int] = ..., args: _Optional[_Iterable[_Union[CompiledNode, _Mapping]]] = ...) -> None: ...

class MonitorAutomaton(_message.Message):
    __slots__ = ("constraint_id", "on_uncertain", "root", "history_window_len", "node_count", "predicate_indices")
    CONSTRAINT_ID_FIELD_NUMBER: _ClassVar[int]
    ON_UNCERTAIN_FIELD_NUMBER: _ClassVar[int]
    ROOT_FIELD_NUMBER: _ClassVar[int]
    HISTORY_WINDOW_LEN_FIELD_NUMBER: _ClassVar[int]
    NODE_COUNT_FIELD_NUMBER: _ClassVar[int]
    PREDICATE_INDICES_FIELD_NUMBER: _ClassVar[int]
    constraint_id: str
    on_uncertain: str
    root: CompiledNode
    history_window_len: int
    node_count: int
    predicate_indices: _containers.RepeatedScalarFieldContainer[int]
    def __init__(self, constraint_id: _Optional[str] = ..., on_uncertain: _Optional[str] = ..., root: _Optional[_Union[CompiledNode, _Mapping]] = ..., history_window_len: _Optional[int] = ..., node_count: _Optional[int] = ..., predicate_indices: _Optional[_Iterable[int]] = ...) -> None: ...

class ResourceBounds(_message.Message):
    __slots__ = ("predicate_slot_count", "scalar_bound_count", "keep_out_term_count", "monitor_count", "max_history_len", "worst_case_term_count")
    PREDICATE_SLOT_COUNT_FIELD_NUMBER: _ClassVar[int]
    SCALAR_BOUND_COUNT_FIELD_NUMBER: _ClassVar[int]
    KEEP_OUT_TERM_COUNT_FIELD_NUMBER: _ClassVar[int]
    MONITOR_COUNT_FIELD_NUMBER: _ClassVar[int]
    MAX_HISTORY_LEN_FIELD_NUMBER: _ClassVar[int]
    WORST_CASE_TERM_COUNT_FIELD_NUMBER: _ClassVar[int]
    predicate_slot_count: int
    scalar_bound_count: int
    keep_out_term_count: int
    monitor_count: int
    max_history_len: int
    worst_case_term_count: int
    def __init__(self, predicate_slot_count: _Optional[int] = ..., scalar_bound_count: _Optional[int] = ..., keep_out_term_count: _Optional[int] = ..., monitor_count: _Optional[int] = ..., max_history_len: _Optional[int] = ..., worst_case_term_count: _Optional[int] = ...) -> None: ...
