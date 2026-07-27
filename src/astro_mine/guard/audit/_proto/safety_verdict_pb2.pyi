from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable
from typing import ClassVar as _ClassVar, Optional as _Optional

DESCRIPTOR: _descriptor.FileDescriptor

class SafetyVerdict(_message.Message):
    __slots__ = ("verdict_version", "agent_id", "tick", "sim_time_s", "spec_id", "spec_content_hash", "compiled_content_hash", "guard_code_version", "layer", "intervention", "reason", "backup_kind", "constraint_ids", "certified_action", "min_barrier_margin", "action_divergence", "inputs_content_hash", "shield_latency_us")
    VERDICT_VERSION_FIELD_NUMBER: _ClassVar[int]
    AGENT_ID_FIELD_NUMBER: _ClassVar[int]
    TICK_FIELD_NUMBER: _ClassVar[int]
    SIM_TIME_S_FIELD_NUMBER: _ClassVar[int]
    SPEC_ID_FIELD_NUMBER: _ClassVar[int]
    SPEC_CONTENT_HASH_FIELD_NUMBER: _ClassVar[int]
    COMPILED_CONTENT_HASH_FIELD_NUMBER: _ClassVar[int]
    GUARD_CODE_VERSION_FIELD_NUMBER: _ClassVar[int]
    LAYER_FIELD_NUMBER: _ClassVar[int]
    INTERVENTION_FIELD_NUMBER: _ClassVar[int]
    REASON_FIELD_NUMBER: _ClassVar[int]
    BACKUP_KIND_FIELD_NUMBER: _ClassVar[int]
    CONSTRAINT_IDS_FIELD_NUMBER: _ClassVar[int]
    CERTIFIED_ACTION_FIELD_NUMBER: _ClassVar[int]
    MIN_BARRIER_MARGIN_FIELD_NUMBER: _ClassVar[int]
    ACTION_DIVERGENCE_FIELD_NUMBER: _ClassVar[int]
    INPUTS_CONTENT_HASH_FIELD_NUMBER: _ClassVar[int]
    SHIELD_LATENCY_US_FIELD_NUMBER: _ClassVar[int]
    verdict_version: str
    agent_id: str
    tick: int
    sim_time_s: float
    spec_id: str
    spec_content_hash: str
    compiled_content_hash: str
    guard_code_version: str
    layer: str
    intervention: str
    reason: str
    backup_kind: str
    constraint_ids: _containers.RepeatedScalarFieldContainer[str]
    certified_action: _containers.RepeatedScalarFieldContainer[float]
    min_barrier_margin: float
    action_divergence: float
    inputs_content_hash: str
    shield_latency_us: float
    def __init__(self, verdict_version: _Optional[str] = ..., agent_id: _Optional[str] = ..., tick: _Optional[int] = ..., sim_time_s: _Optional[float] = ..., spec_id: _Optional[str] = ..., spec_content_hash: _Optional[str] = ..., compiled_content_hash: _Optional[str] = ..., guard_code_version: _Optional[str] = ..., layer: _Optional[str] = ..., intervention: _Optional[str] = ..., reason: _Optional[str] = ..., backup_kind: _Optional[str] = ..., constraint_ids: _Optional[_Iterable[str]] = ..., certified_action: _Optional[_Iterable[float]] = ..., min_barrier_margin: _Optional[float] = ..., action_divergence: _Optional[float] = ..., inputs_content_hash: _Optional[str] = ..., shield_latency_us: _Optional[float] = ...) -> None: ...
