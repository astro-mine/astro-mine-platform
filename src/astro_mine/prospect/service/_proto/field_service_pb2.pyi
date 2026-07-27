from astro_mine.core.units._proto import units_pb2 as _units_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Observation(_message.Message):
    __slots__ = ("x_m", "y_m", "z_m", "value", "noise_sigma", "time_s", "sensor", "likelihood")
    X_M_FIELD_NUMBER: _ClassVar[int]
    Y_M_FIELD_NUMBER: _ClassVar[int]
    Z_M_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    NOISE_SIGMA_FIELD_NUMBER: _ClassVar[int]
    TIME_S_FIELD_NUMBER: _ClassVar[int]
    SENSOR_FIELD_NUMBER: _ClassVar[int]
    LIKELIHOOD_FIELD_NUMBER: _ClassVar[int]
    x_m: float
    y_m: float
    z_m: float
    value: float
    noise_sigma: float
    time_s: float
    sensor: str
    likelihood: str
    def __init__(self, x_m: _Optional[float] = ..., y_m: _Optional[float] = ..., z_m: _Optional[float] = ..., value: _Optional[float] = ..., noise_sigma: _Optional[float] = ..., time_s: _Optional[float] = ..., sensor: _Optional[str] = ..., likelihood: _Optional[str] = ...) -> None: ...

class FieldRequest(_message.Message):
    __slots__ = ("field_id",)
    FIELD_ID_FIELD_NUMBER: _ClassVar[int]
    field_id: str
    def __init__(self, field_id: _Optional[str] = ...) -> None: ...

class FieldSnapshot(_message.Message):
    __slots__ = ("field_id", "prior_bundle", "observations", "content_hash", "revision", "frame", "crs")
    FIELD_ID_FIELD_NUMBER: _ClassVar[int]
    PRIOR_BUNDLE_FIELD_NUMBER: _ClassVar[int]
    OBSERVATIONS_FIELD_NUMBER: _ClassVar[int]
    CONTENT_HASH_FIELD_NUMBER: _ClassVar[int]
    REVISION_FIELD_NUMBER: _ClassVar[int]
    FRAME_FIELD_NUMBER: _ClassVar[int]
    CRS_FIELD_NUMBER: _ClassVar[int]
    field_id: str
    prior_bundle: bytes
    observations: _containers.RepeatedCompositeFieldContainer[Observation]
    content_hash: str
    revision: int
    frame: _units_pb2.ReferenceFrame
    crs: _units_pb2.PlanetaryCRS
    def __init__(self, field_id: _Optional[str] = ..., prior_bundle: _Optional[bytes] = ..., observations: _Optional[_Iterable[_Union[Observation, _Mapping]]] = ..., content_hash: _Optional[str] = ..., revision: _Optional[int] = ..., frame: _Optional[_Union[_units_pb2.ReferenceFrame, _Mapping]] = ..., crs: _Optional[_Union[_units_pb2.PlanetaryCRS, _Mapping]] = ...) -> None: ...

class ObservationBatch(_message.Message):
    __slots__ = ("field_id", "observations")
    FIELD_ID_FIELD_NUMBER: _ClassVar[int]
    OBSERVATIONS_FIELD_NUMBER: _ClassVar[int]
    field_id: str
    observations: _containers.RepeatedCompositeFieldContainer[Observation]
    def __init__(self, field_id: _Optional[str] = ..., observations: _Optional[_Iterable[_Union[Observation, _Mapping]]] = ...) -> None: ...

class UpdateAck(_message.Message):
    __slots__ = ("field_id", "content_hash", "revision", "applied")
    FIELD_ID_FIELD_NUMBER: _ClassVar[int]
    CONTENT_HASH_FIELD_NUMBER: _ClassVar[int]
    REVISION_FIELD_NUMBER: _ClassVar[int]
    APPLIED_FIELD_NUMBER: _ClassVar[int]
    field_id: str
    content_hash: str
    revision: int
    applied: int
    def __init__(self, field_id: _Optional[str] = ..., content_hash: _Optional[str] = ..., revision: _Optional[int] = ..., applied: _Optional[int] = ...) -> None: ...

class SubscribeRequest(_message.Message):
    __slots__ = ("field_id", "from_revision")
    FIELD_ID_FIELD_NUMBER: _ClassVar[int]
    FROM_REVISION_FIELD_NUMBER: _ClassVar[int]
    field_id: str
    from_revision: int
    def __init__(self, field_id: _Optional[str] = ..., from_revision: _Optional[int] = ...) -> None: ...

class BeliefUpdate(_message.Message):
    __slots__ = ("field_id", "observations", "content_hash", "revision")
    FIELD_ID_FIELD_NUMBER: _ClassVar[int]
    OBSERVATIONS_FIELD_NUMBER: _ClassVar[int]
    CONTENT_HASH_FIELD_NUMBER: _ClassVar[int]
    REVISION_FIELD_NUMBER: _ClassVar[int]
    field_id: str
    observations: _containers.RepeatedCompositeFieldContainer[Observation]
    content_hash: str
    revision: int
    def __init__(self, field_id: _Optional[str] = ..., observations: _Optional[_Iterable[_Union[Observation, _Mapping]]] = ..., content_hash: _Optional[str] = ..., revision: _Optional[int] = ...) -> None: ...
