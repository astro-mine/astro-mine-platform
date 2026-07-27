"""Policy/Planner API — closed vocabularies (Core-owned, RM-P1-CORE-01).

The closed enums the :class:`~astro_mine.core.policy.model.PolicyPackage` sidecar draws
from. Like the SADF/registry vocabularies they are deliberately small and grow only by
RFC (conventions.md §3); adding a member is append-only — members are never removed or
repurposed.
"""

from __future__ import annotations

from enum import StrEnum

__all__ = ["TensorDType"]


class TensorDType(StrEnum):
    """Element type of an ONNX-graph input/output tensor (ONNX tensor dtypes).

    Declared per tensor in a :class:`~astro_mine.core.policy.model.TensorSpec` so a host
    ONNX-Runtime session (Mind) and a scorer (Bench) agree on the graph's I/O element
    types without inspecting the graph. Restricted to the dtypes a swarm policy's
    observation/action tensors actually use; extend by RFC if a backend needs more."""

    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT32 = "int32"
    INT64 = "int64"
    UINT8 = "uint8"
    BOOL = "bool"
