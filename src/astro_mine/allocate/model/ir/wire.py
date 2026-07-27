"""``AllocationIR`` <-> Protobuf wire form (canonical cross-language interchange).

The IR serializes to Protobuf so Sim/Bench and cross-language solver backends can consume
it (allocate.md §5). Conversion goes through protobuf's ``json_format``; the model's JSON
projection matches the proto field names (both mirror the canonical JSON Schema).
Serialization is deterministic, so the encoding is **byte-stable** and a
model -> bytes -> model round-trip is exact — the Core ``messages.wire`` / surrogate
``wire`` pattern, the determinism prerequisite for the golden-plan gate (RM-P1-ALLOC-07).
"""

from __future__ import annotations

from typing import Any

from google.protobuf import json_format

from astro_mine.allocate.model.ir._proto import allocation_ir_pb2
from astro_mine.allocate.model.ir.model import AllocationIR

__all__ = ["ir_from_wire", "ir_to_proto", "ir_to_wire"]


def ir_to_proto(ir: AllocationIR) -> allocation_ir_pb2.AllocationIR:
    """Convert an :class:`AllocationIR` to its protobuf message."""
    data: dict[str, Any] = ir.model_dump(mode="json", exclude_none=True)
    msg = allocation_ir_pb2.AllocationIR()
    json_format.ParseDict(data, msg)
    return msg


def ir_to_wire(ir: AllocationIR) -> bytes:
    """Serialize an :class:`AllocationIR` to its canonical, byte-stable wire form."""
    return ir_to_proto(ir).SerializeToString(deterministic=True)


def ir_from_wire(data: bytes) -> AllocationIR:
    """Parse a protobuf wire-form payload into a typed :class:`AllocationIR`."""
    msg = allocation_ir_pb2.AllocationIR()
    msg.ParseFromString(data)
    payload: dict[str, Any] = json_format.MessageToDict(
        msg,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=True,
    )
    return AllocationIR.model_validate(payload)
