"""SafetyVerdict <-> Protobuf wire form (the Core-catalogued cross-language interchange).

Mirrors :mod:`astro_mine.guard.spec.wire`: conversion goes through protobuf's ``json_format``
(the model's JSON projection already matches the proto field names — both mirror the canonical
JSON Schema). ``verdict_to_wire`` serializes **deterministically** so the encoding is byte-stable
and a model -> bytes -> model round-trip reproduces the verdict exactly.

The one wrinkle a verdict adds over the compiled model: ``min_barrier_margin`` is legitimately
``+inf`` on fallback ticks (no keep-out term applied), but proto3 canonical JSON represents a
non-finite ``double`` as the *string* ``"Infinity"`` / ``"-Infinity"`` / ``"NaN"`` — and
``json_format.ParseDict`` rejects a raw Python ``float('inf')``. So non-finite doubles are
translated to those tokens on the way in and back to floats on the way out (the same recursive
dict-fixup shape as the compiled model's ``collision_pair`` scrub).

The high-rate MCAP verdict stream is jsonschema-encoded (:mod:`astro_mine.guard.audit.stream`);
this Protobuf form is the non-hot-path, cross-language wire the async plane / Bench / View
consume (guard.md §5 "everything non-hot-path is Protobuf").
"""

from __future__ import annotations

import math
from typing import Any

from google.protobuf import json_format

from astro_mine.guard.audit._proto import safety_verdict_pb2
from astro_mine.guard.audit.model import SafetyVerdict

__all__ = [
    "verdict_from_proto",
    "verdict_from_wire",
    "verdict_to_proto",
    "verdict_to_wire",
]

_NONFINITE_TOKENS = {"Infinity": math.inf, "-Infinity": -math.inf, "NaN": math.nan}


def _encode_nonfinite(data: Any) -> Any:
    """Recursively map non-finite floats to their proto3-JSON string tokens (``json_format``
    accepts ``"Infinity"`` / ``"-Infinity"`` / ``"NaN"`` but not a raw ``float('inf')``)."""
    if isinstance(data, float) and not math.isfinite(data):
        if math.isnan(data):
            return "NaN"
        return "Infinity" if data > 0 else "-Infinity"
    if isinstance(data, dict):
        return {k: _encode_nonfinite(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_encode_nonfinite(v) for v in data]
    return data


def _decode_nonfinite(data: Any) -> Any:
    """Inverse of :func:`_encode_nonfinite`: map proto3-JSON non-finite tokens back to floats."""
    if isinstance(data, str) and data in _NONFINITE_TOKENS:
        return _NONFINITE_TOKENS[data]
    if isinstance(data, dict):
        return {k: _decode_nonfinite(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_decode_nonfinite(v) for v in data]
    return data


def verdict_to_proto(verdict: SafetyVerdict) -> safety_verdict_pb2.SafetyVerdict:
    """Convert a SafetyVerdict to its protobuf message."""
    msg = safety_verdict_pb2.SafetyVerdict()
    data: dict[str, Any] = verdict.model_dump(by_alias=True, mode="json", exclude_none=True)
    json_format.ParseDict(_encode_nonfinite(data), msg)
    return msg


def verdict_to_wire(verdict: SafetyVerdict) -> bytes:
    """Serialize a SafetyVerdict to its canonical, byte-stable wire form."""
    return verdict_to_proto(verdict).SerializeToString(deterministic=True)


def verdict_from_proto(msg: safety_verdict_pb2.SafetyVerdict) -> SafetyVerdict:
    """Convert a protobuf message back to a typed SafetyVerdict."""
    data: dict[str, Any] = json_format.MessageToDict(
        msg,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=True,
    )
    return SafetyVerdict.model_validate(_decode_nonfinite(data))


def verdict_from_wire(data: bytes) -> SafetyVerdict:
    """Parse a protobuf wire-form payload into a typed SafetyVerdict."""
    msg = safety_verdict_pb2.SafetyVerdict()
    msg.ParseFromString(data)
    return verdict_from_proto(msg)
