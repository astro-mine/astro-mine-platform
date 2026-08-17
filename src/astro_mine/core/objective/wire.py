# SPDX-License-Identifier: Apache-2.0
"""ObjectiveSpec <-> Protobuf wire form (the canonical interchange encoding).

Conversion goes through protobuf's ``json_format`` rather than hand-written field
mapping: the model's JSON projection already matches the proto field names (both
mirror the canonical JSON Schema), so ``ParseDict`` / ``MessageToDict`` map cleanly.

``to_wire`` serializes deterministically so the encoding is **byte-stable**: the same
document always produces identical bytes, and a model -> bytes -> model round-trip
reproduces the document exactly. This is the SADF pattern (astro_mine.core.sadf.wire).
"""

from __future__ import annotations

from typing import Any

from google.protobuf import json_format

from astro_mine.core.objective._proto import objective_pb2
from astro_mine.core.objective.model import ObjectiveDocument

__all__ = ["from_proto", "from_wire", "to_proto", "to_wire"]


def _to_proto_dict(doc: ObjectiveDocument) -> dict[str, Any]:
    # exclude_none drops unset optionals so they map to proto field absence; empty
    # lists/defaults are preserved.
    data: dict[str, Any] = doc.model_dump(by_alias=True, mode="json", exclude_none=True)
    return data


def to_proto(doc: ObjectiveDocument) -> objective_pb2.ObjectiveDocument:
    """Convert an objective document to its protobuf message."""
    msg = objective_pb2.ObjectiveDocument()
    json_format.ParseDict(_to_proto_dict(doc), msg)
    return msg


def to_wire(doc: ObjectiveDocument) -> bytes:
    """Serialize an objective document to its canonical, byte-stable wire form."""
    return to_proto(doc).SerializeToString(deterministic=True)


def from_proto(msg: objective_pb2.ObjectiveDocument) -> ObjectiveDocument:
    """Convert a protobuf message back to a typed objective document."""
    data: dict[str, Any] = json_format.MessageToDict(
        msg,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=True,
    )
    return ObjectiveDocument.model_validate(data)


def from_wire(data: bytes) -> ObjectiveDocument:
    """Parse a protobuf wire-form payload into a typed objective document."""
    msg = objective_pb2.ObjectiveDocument()
    msg.ParseFromString(data)
    return from_proto(msg)
