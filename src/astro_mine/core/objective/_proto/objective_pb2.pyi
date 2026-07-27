from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ObjectiveDocument(_message.Message):
    __slots__ = ("objective_version", "objective")
    OBJECTIVE_VERSION_FIELD_NUMBER: _ClassVar[int]
    OBJECTIVE_FIELD_NUMBER: _ClassVar[int]
    objective_version: str
    objective: ObjectiveSpec
    def __init__(self, objective_version: _Optional[str] = ..., objective: _Optional[_Union[ObjectiveSpec, _Mapping]] = ...) -> None: ...

class ObjectiveSpec(_message.Message):
    __slots__ = ("id", "name", "description", "scenario_ref", "success_criteria", "labels", "provenance")
    class LabelsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    ID_FIELD_NUMBER: _ClassVar[int]
    NAME_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    SCENARIO_REF_FIELD_NUMBER: _ClassVar[int]
    SUCCESS_CRITERIA_FIELD_NUMBER: _ClassVar[int]
    LABELS_FIELD_NUMBER: _ClassVar[int]
    PROVENANCE_FIELD_NUMBER: _ClassVar[int]
    id: str
    name: str
    description: str
    scenario_ref: str
    success_criteria: _containers.RepeatedCompositeFieldContainer[SuccessCriterion]
    labels: _containers.ScalarMap[str, str]
    provenance: Provenance
    def __init__(self, id: _Optional[str] = ..., name: _Optional[str] = ..., description: _Optional[str] = ..., scenario_ref: _Optional[str] = ..., success_criteria: _Optional[_Iterable[_Union[SuccessCriterion, _Mapping]]] = ..., labels: _Optional[_Mapping[str, str]] = ..., provenance: _Optional[_Union[Provenance, _Mapping]] = ...) -> None: ...

class SuccessCriterion(_message.Message):
    __slots__ = ("id", "description", "binding", "required", "weight", "deadline_s")
    ID_FIELD_NUMBER: _ClassVar[int]
    DESCRIPTION_FIELD_NUMBER: _ClassVar[int]
    BINDING_FIELD_NUMBER: _ClassVar[int]
    REQUIRED_FIELD_NUMBER: _ClassVar[int]
    WEIGHT_FIELD_NUMBER: _ClassVar[int]
    DEADLINE_S_FIELD_NUMBER: _ClassVar[int]
    id: str
    description: str
    binding: MetricBinding
    required: bool
    weight: float
    deadline_s: float
    def __init__(self, id: _Optional[str] = ..., description: _Optional[str] = ..., binding: _Optional[_Union[MetricBinding, _Mapping]] = ..., required: _Optional[bool] = ..., weight: _Optional[float] = ..., deadline_s: _Optional[float] = ...) -> None: ...

class EvaluationWindow(_message.Message):
    __slots__ = ("kind", "duration_s")
    KIND_FIELD_NUMBER: _ClassVar[int]
    DURATION_S_FIELD_NUMBER: _ClassVar[int]
    kind: str
    duration_s: float
    def __init__(self, kind: _Optional[str] = ..., duration_s: _Optional[float] = ...) -> None: ...

class MetricBinding(_message.Message):
    __slots__ = ("metric", "unit", "direction", "target", "tolerance", "aggregation", "threshold", "evaluation_window")
    METRIC_FIELD_NUMBER: _ClassVar[int]
    UNIT_FIELD_NUMBER: _ClassVar[int]
    DIRECTION_FIELD_NUMBER: _ClassVar[int]
    TARGET_FIELD_NUMBER: _ClassVar[int]
    TOLERANCE_FIELD_NUMBER: _ClassVar[int]
    AGGREGATION_FIELD_NUMBER: _ClassVar[int]
    THRESHOLD_FIELD_NUMBER: _ClassVar[int]
    EVALUATION_WINDOW_FIELD_NUMBER: _ClassVar[int]
    metric: str
    unit: str
    direction: str
    target: float
    tolerance: float
    aggregation: str
    threshold: float
    evaluation_window: EvaluationWindow
    def __init__(self, metric: _Optional[str] = ..., unit: _Optional[str] = ..., direction: _Optional[str] = ..., target: _Optional[float] = ..., tolerance: _Optional[float] = ..., aggregation: _Optional[str] = ..., threshold: _Optional[float] = ..., evaluation_window: _Optional[_Union[EvaluationWindow, _Mapping]] = ...) -> None: ...

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
