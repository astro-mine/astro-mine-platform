from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class CoveragePoint(_message.Message):
    __slots__ = ("nominal", "empirical")
    NOMINAL_FIELD_NUMBER: _ClassVar[int]
    EMPIRICAL_FIELD_NUMBER: _ClassVar[int]
    nominal: float
    empirical: float
    def __init__(self, nominal: _Optional[float] = ..., empirical: _Optional[float] = ...) -> None: ...

class TailBehavior(_message.Message):
    __slots__ = ("p95_abs_error", "p99_abs_error", "max_abs_error")
    P95_ABS_ERROR_FIELD_NUMBER: _ClassVar[int]
    P99_ABS_ERROR_FIELD_NUMBER: _ClassVar[int]
    MAX_ABS_ERROR_FIELD_NUMBER: _ClassVar[int]
    p95_abs_error: float
    p99_abs_error: float
    max_abs_error: float
    def __init__(self, p95_abs_error: _Optional[float] = ..., p99_abs_error: _Optional[float] = ..., max_abs_error: _Optional[float] = ...) -> None: ...

class ContinuousMetrics(_message.Message):
    __slots__ = ("unit", "rmse", "coverage", "tail")
    UNIT_FIELD_NUMBER: _ClassVar[int]
    RMSE_FIELD_NUMBER: _ClassVar[int]
    COVERAGE_FIELD_NUMBER: _ClassVar[int]
    TAIL_FIELD_NUMBER: _ClassVar[int]
    unit: str
    rmse: float
    coverage: _containers.RepeatedCompositeFieldContainer[CoveragePoint]
    tail: TailBehavior
    def __init__(self, unit: _Optional[str] = ..., rmse: _Optional[float] = ..., coverage: _Optional[_Iterable[_Union[CoveragePoint, _Mapping]]] = ..., tail: _Optional[_Union[TailBehavior, _Mapping]] = ...) -> None: ...

class CategoricalMetrics(_message.Message):
    __slots__ = ("classes", "accuracy", "reliability")
    CLASSES_FIELD_NUMBER: _ClassVar[int]
    ACCURACY_FIELD_NUMBER: _ClassVar[int]
    RELIABILITY_FIELD_NUMBER: _ClassVar[int]
    classes: _containers.RepeatedScalarFieldContainer[str]
    accuracy: float
    reliability: _containers.RepeatedCompositeFieldContainer[CoveragePoint]
    def __init__(self, classes: _Optional[_Iterable[str]] = ..., accuracy: _Optional[float] = ..., reliability: _Optional[_Iterable[_Union[CoveragePoint, _Mapping]]] = ...) -> None: ...

class ChannelError(_message.Message):
    __slots__ = ("channel", "kind", "continuous", "categorical")
    CHANNEL_FIELD_NUMBER: _ClassVar[int]
    KIND_FIELD_NUMBER: _ClassVar[int]
    CONTINUOUS_FIELD_NUMBER: _ClassVar[int]
    CATEGORICAL_FIELD_NUMBER: _ClassVar[int]
    channel: str
    kind: str
    continuous: ContinuousMetrics
    categorical: CategoricalMetrics
    def __init__(self, channel: _Optional[str] = ..., kind: _Optional[str] = ..., continuous: _Optional[_Union[ContinuousMetrics, _Mapping]] = ..., categorical: _Optional[_Union[CategoricalMetrics, _Mapping]] = ...) -> None: ...

class Bound(_message.Message):
    __slots__ = ("low", "high")
    LOW_FIELD_NUMBER: _ClassVar[int]
    HIGH_FIELD_NUMBER: _ClassVar[int]
    low: float
    high: float
    def __init__(self, low: _Optional[float] = ..., high: _Optional[float] = ...) -> None: ...

class TrustRegion(_message.Message):
    __slots__ = ("bounds",)
    class BoundsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Bound
        def __init__(self, key: _Optional[str] = ..., value: _Optional[_Union[Bound, _Mapping]] = ...) -> None: ...
    BOUNDS_FIELD_NUMBER: _ClassVar[int]
    bounds: _containers.MessageMap[str, Bound]
    def __init__(self, bounds: _Optional[_Mapping[str, Bound]] = ...) -> None: ...

class OracleRef(_message.Message):
    __slots__ = ("producer", "producer_version", "config_hash")
    PRODUCER_FIELD_NUMBER: _ClassVar[int]
    PRODUCER_VERSION_FIELD_NUMBER: _ClassVar[int]
    CONFIG_HASH_FIELD_NUMBER: _ClassVar[int]
    producer: str
    producer_version: str
    config_hash: str
    def __init__(self, producer: _Optional[str] = ..., producer_version: _Optional[str] = ..., config_hash: _Optional[str] = ...) -> None: ...

class SubstitutionPolicy(_message.Message):
    __slots__ = ("recommended_error_budget", "escalate_on_ood", "budget_horizon_steps")
    class RecommendedErrorBudgetEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    RECOMMENDED_ERROR_BUDGET_FIELD_NUMBER: _ClassVar[int]
    ESCALATE_ON_OOD_FIELD_NUMBER: _ClassVar[int]
    BUDGET_HORIZON_STEPS_FIELD_NUMBER: _ClassVar[int]
    recommended_error_budget: _containers.ScalarMap[str, float]
    escalate_on_ood: bool
    budget_horizon_steps: int
    def __init__(self, recommended_error_budget: _Optional[_Mapping[str, float]] = ..., escalate_on_ood: _Optional[bool] = ..., budget_horizon_steps: _Optional[int] = ...) -> None: ...

class RolloutError(_message.Message):
    __slots__ = ("horizon_steps", "rmse_by_horizon")
    HORIZON_STEPS_FIELD_NUMBER: _ClassVar[int]
    RMSE_BY_HORIZON_FIELD_NUMBER: _ClassVar[int]
    horizon_steps: int
    rmse_by_horizon: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, horizon_steps: _Optional[int] = ..., rmse_by_horizon: _Optional[_Iterable[float]] = ...) -> None: ...

class ErrorReport(_message.Message):
    __slots__ = ("surrogate_name", "surrogate_version", "domain", "channels", "trust_region", "validation_dataset_hash", "oracle", "substitution_policy", "rollout")
    SURROGATE_NAME_FIELD_NUMBER: _ClassVar[int]
    SURROGATE_VERSION_FIELD_NUMBER: _ClassVar[int]
    DOMAIN_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    TRUST_REGION_FIELD_NUMBER: _ClassVar[int]
    VALIDATION_DATASET_HASH_FIELD_NUMBER: _ClassVar[int]
    ORACLE_FIELD_NUMBER: _ClassVar[int]
    SUBSTITUTION_POLICY_FIELD_NUMBER: _ClassVar[int]
    ROLLOUT_FIELD_NUMBER: _ClassVar[int]
    surrogate_name: str
    surrogate_version: str
    domain: str
    channels: _containers.RepeatedCompositeFieldContainer[ChannelError]
    trust_region: TrustRegion
    validation_dataset_hash: str
    oracle: OracleRef
    substitution_policy: SubstitutionPolicy
    rollout: RolloutError
    def __init__(self, surrogate_name: _Optional[str] = ..., surrogate_version: _Optional[str] = ..., domain: _Optional[str] = ..., channels: _Optional[_Iterable[_Union[ChannelError, _Mapping]]] = ..., trust_region: _Optional[_Union[TrustRegion, _Mapping]] = ..., validation_dataset_hash: _Optional[str] = ..., oracle: _Optional[_Union[OracleRef, _Mapping]] = ..., substitution_policy: _Optional[_Union[SubstitutionPolicy, _Mapping]] = ..., rollout: _Optional[_Union[RolloutError, _Mapping]] = ...) -> None: ...
