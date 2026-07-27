from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class DecisionVariable(_message.Message):
    __slots__ = ("id", "kind", "lower", "upper", "semantic", "task_ref", "asset_ref")
    ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    LOWER_FIELD_NUMBER: _ClassVar[int]
    UPPER_FIELD_NUMBER: _ClassVar[int]
    SEMANTIC_FIELD_NUMBER: _ClassVar[int]
    TASK_REF_FIELD_NUMBER: _ClassVar[int]
    ASSET_REF_FIELD_NUMBER: _ClassVar[int]
    id: str
    kind: str
    lower: float
    upper: float
    semantic: str
    task_ref: str
    asset_ref: str
    def __init__(self, id: _Optional[str] = ..., kind: _Optional[str] = ..., lower: _Optional[float] = ..., upper: _Optional[float] = ..., semantic: _Optional[str] = ..., task_ref: _Optional[str] = ..., asset_ref: _Optional[str] = ...) -> None: ...

class ConstraintTerm(_message.Message):
    __slots__ = ("var_ref", "coefficient")
    VAR_REF_FIELD_NUMBER: _ClassVar[int]
    COEFFICIENT_FIELD_NUMBER: _ClassVar[int]
    var_ref: str
    coefficient: float
    def __init__(self, var_ref: _Optional[str] = ..., coefficient: _Optional[float] = ...) -> None: ...

class Constraint(_message.Message):
    __slots__ = ("id", "kind", "terms", "sense", "rhs")
    ID_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    TERMS_FIELD_NUMBER: _ClassVar[int]
    SENSE_FIELD_NUMBER: _ClassVar[int]
    RHS_FIELD_NUMBER: _ClassVar[int]
    id: str
    kind: str
    terms: _containers.RepeatedCompositeFieldContainer[ConstraintTerm]
    sense: str
    rhs: float
    def __init__(self, id: _Optional[str] = ..., kind: _Optional[str] = ..., terms: _Optional[_Iterable[_Union[ConstraintTerm, _Mapping]]] = ..., sense: _Optional[str] = ..., rhs: _Optional[float] = ...) -> None: ...

class ObjectiveTerm(_message.Message):
    __slots__ = ("id", "var_ref", "coefficient")
    ID_FIELD_NUMBER: _ClassVar[int]
    VAR_REF_FIELD_NUMBER: _ClassVar[int]
    COEFFICIENT_FIELD_NUMBER: _ClassVar[int]
    id: str
    var_ref: str
    coefficient: float
    def __init__(self, id: _Optional[str] = ..., var_ref: _Optional[str] = ..., coefficient: _Optional[float] = ...) -> None: ...

class AllocationIR(_message.Message):
    __slots__ = ("ir_version", "variables", "constraints", "objective_terms", "objective_sense", "metadata")
    class MetadataEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: str
        def __init__(self, key: _Optional[str] = ..., value: _Optional[str] = ...) -> None: ...
    IR_VERSION_FIELD_NUMBER: _ClassVar[int]
    VARIABLES_FIELD_NUMBER: _ClassVar[int]
    CONSTRAINTS_FIELD_NUMBER: _ClassVar[int]
    OBJECTIVE_TERMS_FIELD_NUMBER: _ClassVar[int]
    OBJECTIVE_SENSE_FIELD_NUMBER: _ClassVar[int]
    METADATA_FIELD_NUMBER: _ClassVar[int]
    ir_version: str
    variables: _containers.RepeatedCompositeFieldContainer[DecisionVariable]
    constraints: _containers.RepeatedCompositeFieldContainer[Constraint]
    objective_terms: _containers.RepeatedCompositeFieldContainer[ObjectiveTerm]
    objective_sense: str
    metadata: _containers.ScalarMap[str, str]
    def __init__(self, ir_version: _Optional[str] = ..., variables: _Optional[_Iterable[_Union[DecisionVariable, _Mapping]]] = ..., constraints: _Optional[_Iterable[_Union[Constraint, _Mapping]]] = ..., objective_terms: _Optional[_Iterable[_Union[ObjectiveTerm, _Mapping]]] = ..., objective_sense: _Optional[str] = ..., metadata: _Optional[_Mapping[str, str]] = ...) -> None: ...
