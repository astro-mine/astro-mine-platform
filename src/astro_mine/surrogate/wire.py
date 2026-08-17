# SPDX-License-Identifier: Apache-2.0
"""``ErrorReport`` <-> Protobuf wire form (canonical cross-language interchange).

The ErrorReport serializes to Protobuf so Sim's multi-fidelity scheduler can consume it
across languages (surrogate.md §5). Conversion goes through protobuf's ``json_format``;
the model's JSON projection matches the proto field names (both mirror the canonical JSON
Schema). Serialization is deterministic, so the encoding is **byte-stable** and a
model -> bytes -> model round-trip is exact — the Core ``messages.wire`` pattern.
"""

from __future__ import annotations

from typing import Any

from google.protobuf import json_format

from astro_mine.surrogate._proto import surrogate_pb2
from astro_mine.surrogate.report import ErrorReport

__all__ = ["error_report_from_wire", "error_report_to_proto", "error_report_to_wire"]


def error_report_to_proto(report: ErrorReport) -> surrogate_pb2.ErrorReport:
    """Convert an :class:`ErrorReport` to its protobuf message."""
    data: dict[str, Any] = report.model_dump(by_alias=True, mode="json", exclude_none=True)
    msg = surrogate_pb2.ErrorReport()
    json_format.ParseDict(data, msg)
    return msg


def error_report_to_wire(report: ErrorReport) -> bytes:
    """Serialize an :class:`ErrorReport` to its canonical, byte-stable wire form."""
    return error_report_to_proto(report).SerializeToString(deterministic=True)


def error_report_from_wire(data: bytes) -> ErrorReport:
    """Parse a protobuf wire-form payload into a typed :class:`ErrorReport`."""
    msg = surrogate_pb2.ErrorReport()
    msg.ParseFromString(data)
    payload: dict[str, Any] = json_format.MessageToDict(
        msg,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=True,
    )
    return ErrorReport.model_validate(payload)
