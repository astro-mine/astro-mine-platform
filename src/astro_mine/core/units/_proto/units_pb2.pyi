from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ReferenceFrame(_message.Message):
    __slots__ = ("name", "frame_class", "center")
    NAME_FIELD_NUMBER: _ClassVar[int]
    FRAME_CLASS_FIELD_NUMBER: _ClassVar[int]
    CENTER_FIELD_NUMBER: _ClassVar[int]
    name: str
    frame_class: str
    center: str
    def __init__(self, name: _Optional[str] = ..., frame_class: _Optional[str] = ..., center: _Optional[str] = ...) -> None: ...

class PlanetaryCRS(_message.Message):
    __slots__ = ("body", "body_fixed_frame", "reference_radius_m", "projection", "datum")
    BODY_FIELD_NUMBER: _ClassVar[int]
    BODY_FIXED_FRAME_FIELD_NUMBER: _ClassVar[int]
    REFERENCE_RADIUS_M_FIELD_NUMBER: _ClassVar[int]
    PROJECTION_FIELD_NUMBER: _ClassVar[int]
    DATUM_FIELD_NUMBER: _ClassVar[int]
    body: str
    body_fixed_frame: str
    reference_radius_m: float
    projection: str
    datum: str
    def __init__(self, body: _Optional[str] = ..., body_fixed_frame: _Optional[str] = ..., reference_radius_m: _Optional[float] = ..., projection: _Optional[str] = ..., datum: _Optional[str] = ...) -> None: ...

class Epoch(_message.Message):
    __slots__ = ("tdb_seconds", "scale")
    TDB_SECONDS_FIELD_NUMBER: _ClassVar[int]
    SCALE_FIELD_NUMBER: _ClassVar[int]
    tdb_seconds: float
    scale: str
    def __init__(self, tdb_seconds: _Optional[float] = ..., scale: _Optional[str] = ...) -> None: ...

class EpochWindow(_message.Message):
    __slots__ = ("start", "end")
    START_FIELD_NUMBER: _ClassVar[int]
    END_FIELD_NUMBER: _ClassVar[int]
    start: Epoch
    end: Epoch
    def __init__(self, start: _Optional[_Union[Epoch, _Mapping]] = ..., end: _Optional[_Union[Epoch, _Mapping]] = ...) -> None: ...
