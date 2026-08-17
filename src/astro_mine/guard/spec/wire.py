# SPDX-License-Identifier: Apache-2.0
"""SafetySpec / CompiledSafetyModel <-> Protobuf wire form (the canonical interchange).

Conversion goes through protobuf's ``json_format`` rather than hand-written field mapping:
each model's JSON projection already matches the proto field names (both mirror the canonical
JSON Schema), so ``ParseDict`` / ``MessageToDict`` map cleanly. This is the Core objective/SADF
pattern (``astro_mine.core.objective.wire``).

``to_wire`` serializes **deterministically** so the encoding is byte-stable: the same document
always produces identical bytes, and a model -> bytes -> model round-trip reproduces the
document exactly. The one wrinkle is the ``collision_pair`` field — a ``tuple[str, str] | None``
that maps to a proto ``repeated string``: an unset pair round-trips as an empty list, which is
scrubbed back to ``None`` so the typed model stays exact.
"""

from __future__ import annotations

from typing import Any

from google.protobuf import json_format

from astro_mine.guard.spec._proto import compiled_safety_model_pb2, safety_spec_pb2
from astro_mine.guard.spec.ir import CompiledSafetyModel
from astro_mine.guard.spec.model import SafetyDocument

__all__ = [
    "compiled_from_proto",
    "compiled_from_wire",
    "compiled_to_proto",
    "compiled_to_wire",
    "from_proto",
    "from_wire",
    "to_proto",
    "to_wire",
]


def _drop_empty_collision_pairs(data: Any) -> Any:
    """Recursively drop ``collision_pair`` keys whose value is an empty list.

    A proto ``repeated string`` with no presence tracking is emitted as ``[]`` even when
    unset; the typed model carries the pair as ``tuple[str, str] | None``, so an empty list
    must map back to absence (``None``), not an empty tuple."""
    if isinstance(data, dict):
        out = {
            k: _drop_empty_collision_pairs(v)
            for k, v in data.items()
            if not (k == "collision_pair" and v == [])
        }
        return out
    if isinstance(data, list):
        return [_drop_empty_collision_pairs(v) for v in data]
    return data


# --- SafetyDocument --------------------------------------------------------------


def to_proto(doc: SafetyDocument) -> safety_spec_pb2.SafetyDocument:
    """Convert a SafetySpec document to its protobuf message."""
    msg = safety_spec_pb2.SafetyDocument()
    data: dict[str, Any] = doc.model_dump(by_alias=True, mode="json", exclude_none=True)
    json_format.ParseDict(data, msg)
    return msg


def to_wire(doc: SafetyDocument) -> bytes:
    """Serialize a SafetySpec document to its canonical, byte-stable wire form."""
    return to_proto(doc).SerializeToString(deterministic=True)


def from_proto(msg: safety_spec_pb2.SafetyDocument) -> SafetyDocument:
    """Convert a protobuf message back to a typed SafetySpec document."""
    data: dict[str, Any] = json_format.MessageToDict(
        msg,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=True,
    )
    return SafetyDocument.model_validate(_drop_empty_collision_pairs(data))


def from_wire(data: bytes) -> SafetyDocument:
    """Parse a protobuf wire-form payload into a typed SafetySpec document."""
    msg = safety_spec_pb2.SafetyDocument()
    msg.ParseFromString(data)
    return from_proto(msg)


# --- CompiledSafetyModel ---------------------------------------------------------


def compiled_to_proto(model: CompiledSafetyModel) -> compiled_safety_model_pb2.CompiledSafetyModel:
    """Convert a compiled safety model to its protobuf message."""
    msg = compiled_safety_model_pb2.CompiledSafetyModel()
    data: dict[str, Any] = model.model_dump(by_alias=True, mode="json", exclude_none=True)
    json_format.ParseDict(data, msg)
    return msg


def compiled_to_wire(model: CompiledSafetyModel) -> bytes:
    """Serialize a compiled safety model to its canonical, byte-stable wire form."""
    return compiled_to_proto(model).SerializeToString(deterministic=True)


def compiled_from_proto(
    msg: compiled_safety_model_pb2.CompiledSafetyModel,
) -> CompiledSafetyModel:
    """Convert a protobuf message back to a typed compiled safety model."""
    data: dict[str, Any] = json_format.MessageToDict(
        msg,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=True,
    )
    return CompiledSafetyModel.model_validate(_drop_empty_collision_pairs(data))


def compiled_from_wire(data: bytes) -> CompiledSafetyModel:
    """Parse a protobuf wire-form payload into a typed compiled safety model."""
    msg = compiled_safety_model_pb2.CompiledSafetyModel()
    msg.ParseFromString(data)
    return compiled_from_proto(msg)
