"""The stack spec — Mind's declarative autonomy-stack description (RM-P1-MIND-01).

The authored artifact (YAML/JSON, JSON-Schema-validated, Pydantic-typed) the composer
turns into a runnable hierarchy. See :mod:`astro_mine.mind.spec.model` for the shape and
:mod:`astro_mine.mind.spec.loader` for the load/validate pipeline.
"""

from __future__ import annotations

from astro_mine.mind.spec.enums import (
    CoordinationKind,
    ExecutionKind,
    ReplanTriggerKind,
    TierRole,
)
from astro_mine.mind.spec.loader import (
    StackSpecError,
    StackSpecValidationError,
    load_schema,
    load_stack_spec,
    validate_stack_spec,
)
from astro_mine.mind.spec.model import (
    STACK_SPEC_VERSION,
    CoordinationSpec,
    ExecutionSpec,
    FallbackBinding,
    ReplanTrigger,
    ShieldBinding,
    SpecProvenance,
    StackSpec,
    StackSpecDocument,
    TierBinding,
)

__all__ = [
    "STACK_SPEC_VERSION",
    "CoordinationKind",
    "CoordinationSpec",
    "ExecutionKind",
    "ExecutionSpec",
    "FallbackBinding",
    "ReplanTrigger",
    "ReplanTriggerKind",
    "ShieldBinding",
    "SpecProvenance",
    "StackSpec",
    "StackSpecDocument",
    "StackSpecError",
    "StackSpecValidationError",
    "TierBinding",
    "TierRole",
    "load_schema",
    "load_stack_spec",
    "validate_stack_spec",
]
