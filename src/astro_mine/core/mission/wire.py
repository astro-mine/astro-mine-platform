# SPDX-License-Identifier: Apache-2.0
"""MissionSpec <-> Protobuf wire form (the canonical interchange encoding).

Conversion goes through protobuf's ``json_format`` rather than hand-written field
mapping: the model's JSON projection already matches the proto field names (both mirror
the canonical JSON Schema), so ``ParseDict`` / ``MessageToDict`` map cleanly. Closed
vocabularies (regime, maneuver_type) ride as strings, so the StrEnum values map with no
special-casing.

``to_wire`` serializes deterministically so the encoding is **byte-stable** (the SADF/
ObjectiveSpec pattern): the same document always produces identical bytes, and a
model -> bytes -> model round-trip reproduces the document exactly.
"""

from __future__ import annotations

from typing import Any

from google.protobuf import json_format

from astro_mine.core.mission._proto import mission_pb2
from astro_mine.core.mission.model import MissionDocument

__all__ = ["from_proto", "from_wire", "to_proto", "to_wire"]


def _to_proto_dict(doc: MissionDocument) -> dict[str, Any]:
    # exclude_none drops unset optionals so they map to proto field absence; empty
    # lists/defaults are preserved.
    data: dict[str, Any] = doc.model_dump(by_alias=True, mode="json", exclude_none=True)
    return data


def to_proto(doc: MissionDocument) -> mission_pb2.MissionDocument:
    """Convert a mission document to its protobuf message."""
    msg = mission_pb2.MissionDocument()
    json_format.ParseDict(_to_proto_dict(doc), msg)
    return msg


def to_wire(doc: MissionDocument) -> bytes:
    """Serialize a mission document to its canonical, byte-stable wire form."""
    return to_proto(doc).SerializeToString(deterministic=True)


def from_proto(msg: mission_pb2.MissionDocument) -> MissionDocument:
    """Convert a protobuf message back to a typed mission document."""
    data: dict[str, Any] = json_format.MessageToDict(
        msg,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=True,
    )
    return MissionDocument.model_validate(data)


def from_wire(data: bytes) -> MissionDocument:
    """Parse a protobuf wire-form payload into a typed mission document."""
    msg = mission_pb2.MissionDocument()
    msg.ParseFromString(data)
    return from_proto(msg)
