from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class DescribeRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DescribeResponse(_message.Message):
    __slots__ = ("possible_agents", "core_interfaces", "frame_name", "scenario", "dt_s")
    class CoreInterfacesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    POSSIBLE_AGENTS_FIELD_NUMBER: _ClassVar[int]
    CORE_INTERFACES_FIELD_NUMBER: _ClassVar[int]
    FRAME_NAME_FIELD_NUMBER: _ClassVar[int]
    SCENARIO_FIELD_NUMBER: _ClassVar[int]
    DT_S_FIELD_NUMBER: _ClassVar[int]
    possible_agents: _containers.RepeatedScalarFieldContainer[str]
    core_interfaces: _containers.ScalarMap[str, str]
    frame_name: str
    scenario: str
    dt_s: float
    def __init__(self, possible_agents: _Optional[_Iterable[str]] = ..., core_interfaces: _Optional[_Mapping[str, str]] = ..., frame_name: _Optional[str] = ..., scenario: _Optional[str] = ..., dt_s: _Optional[float] = ...) -> None: ...

class ResetRequest(_message.Message):
    __slots__ = ("seed",)
    SEED_FIELD_NUMBER: _ClassVar[int]
    seed: int
    def __init__(self, seed: _Optional[int] = ...) -> None: ...

class ResetResponse(_message.Message):
    __slots__ = ("observations", "agents")
    OBSERVATIONS_FIELD_NUMBER: _ClassVar[int]
    AGENTS_FIELD_NUMBER: _ClassVar[int]
    observations: bytes
    agents: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, observations: _Optional[bytes] = ..., agents: _Optional[_Iterable[str]] = ...) -> None: ...

class StepRequest(_message.Message):
    __slots__ = ("action_batch_json", "steps")
    ACTION_BATCH_JSON_FIELD_NUMBER: _ClassVar[int]
    STEPS_FIELD_NUMBER: _ClassVar[int]
    action_batch_json: str
    steps: int
    def __init__(self, action_batch_json: _Optional[str] = ..., steps: _Optional[int] = ...) -> None: ...

class StepResponse(_message.Message):
    __slots__ = ("observations", "tick", "sim_time_s", "dt_s", "terminations", "truncations", "agents")
    class TerminationsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: bool
        def __init__(self, key: _Optional[str] = ..., value: _Optional[bool] = ...) -> None: ...
    class TruncationsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: bool
        def __init__(self, key: _Optional[str] = ..., value: _Optional[bool] = ...) -> None: ...
    OBSERVATIONS_FIELD_NUMBER: _ClassVar[int]
    TICK_FIELD_NUMBER: _ClassVar[int]
    SIM_TIME_S_FIELD_NUMBER: _ClassVar[int]
    DT_S_FIELD_NUMBER: _ClassVar[int]
    TERMINATIONS_FIELD_NUMBER: _ClassVar[int]
    TRUNCATIONS_FIELD_NUMBER: _ClassVar[int]
    AGENTS_FIELD_NUMBER: _ClassVar[int]
    observations: bytes
    tick: int
    sim_time_s: float
    dt_s: float
    terminations: _containers.ScalarMap[str, bool]
    truncations: _containers.ScalarMap[str, bool]
    agents: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, observations: _Optional[bytes] = ..., tick: _Optional[int] = ..., sim_time_s: _Optional[float] = ..., dt_s: _Optional[float] = ..., terminations: _Optional[_Mapping[str, bool]] = ..., truncations: _Optional[_Mapping[str, bool]] = ..., agents: _Optional[_Iterable[str]] = ...) -> None: ...
