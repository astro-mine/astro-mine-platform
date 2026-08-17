# SPDX-License-Identifier: Apache-2.0
"""SADF <-> Protobuf wire form (the canonical interchange encoding).

Conversion goes through protobuf's ``json_format`` rather than hand-written field
mapping: the model's JSON projection already matches the proto field names (both
mirror the canonical JSON Schema), so ``ParseDict``/``MessageToDict`` map cleanly.
The single exception is the asset's ``return`` block (a Python keyword), stored as
``return_spec`` in proto and remapped here.

``to_wire`` serializes deterministically so the encoding is **byte-stable**: the
same document always produces identical bytes, and a model -> bytes -> model
round-trip reproduces the document exactly.
"""

from __future__ import annotations

from typing import Any

from google.protobuf import json_format

from astro_mine.core.sadf._proto import sadf_pb2
from astro_mine.core.sadf.model import SadfDocument

__all__ = ["from_proto", "from_wire", "to_proto", "to_wire"]


def _to_proto_dict(doc: SadfDocument) -> dict[str, Any]:
    # exclude_none drops unset optionals so they map to proto field absence (rather
    # than relying on JSON null handling); empty lists/defaults are preserved.
    data: dict[str, Any] = doc.model_dump(by_alias=True, mode="json", exclude_none=True)
    asset = data.get("asset")
    if isinstance(asset, dict) and "return" in asset:
        asset["return_spec"] = asset.pop("return")
    return data


def to_proto(doc: SadfDocument) -> sadf_pb2.SadfDocument:
    """Convert a SADF document to its protobuf message."""
    msg = sadf_pb2.SadfDocument()
    json_format.ParseDict(_to_proto_dict(doc), msg)
    return msg


def to_wire(doc: SadfDocument) -> bytes:
    """Serialize a SADF document to its canonical, byte-stable protobuf wire form."""
    return to_proto(doc).SerializeToString(deterministic=True)


def from_proto(msg: sadf_pb2.SadfDocument) -> SadfDocument:
    """Convert a protobuf message back to a typed SADF document."""
    data: dict[str, Any] = json_format.MessageToDict(
        msg,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=True,
    )
    asset = data.get("asset")
    if isinstance(asset, dict) and "return_spec" in asset:
        asset["return"] = asset.pop("return_spec")
    return SadfDocument.model_validate(data)


def from_wire(data: bytes) -> SadfDocument:
    """Parse a protobuf wire-form payload into a typed SADF document."""
    msg = sadf_pb2.SadfDocument()
    msg.ParseFromString(data)
    return from_proto(msg)
