# SPDX-License-Identifier: Apache-2.0
"""Units frame/CRS/time types <-> Protobuf wire form (RFC-0007, RM-P1-CORE-07).

The SPICE-shaped value types (:class:`ReferenceFrame`, :class:`PlanetaryCRS`,
:class:`Epoch`, :class:`EpochWindow`) put on the control-plane wire via ``units.proto``.

Conversion goes through protobuf's ``json_format`` rather than hand-written field
mapping: each model's JSON projection already matches the proto field names (both mirror
``units.schema.json``), and the :class:`~astro_mine.core.units.FrameClass` /
:class:`~astro_mine.core.units.TimeScale` closed vocabularies ride as their ``StrEnum``
string values with no special-casing (matching the Cap'n Proto ``Text`` encoding). This
is the free-function ``*_to_proto`` / ``*_from_proto`` module pattern ``messages`` /
``mission`` / ``objective`` / ``sadf`` already use — not ``.to_proto()`` methods.

``*_to_wire`` serializes deterministically so the encoding is **byte-stable**: the same
value always produces identical bytes, and a value -> bytes -> value round-trip
reproduces the value exactly (including ``TimeScale.ET``).
"""

from __future__ import annotations

from typing import Any

from google.protobuf import json_format
from google.protobuf.message import Message
from pydantic import BaseModel

from astro_mine.core.units._proto import units_pb2
from astro_mine.core.units.model import Epoch, EpochWindow, PlanetaryCRS, ReferenceFrame

__all__ = [
    "epoch_from_proto",
    "epoch_from_wire",
    "epoch_to_proto",
    "epoch_to_wire",
    "epoch_window_from_proto",
    "epoch_window_from_wire",
    "epoch_window_to_proto",
    "epoch_window_to_wire",
    "planetary_crs_from_proto",
    "planetary_crs_from_wire",
    "planetary_crs_to_proto",
    "planetary_crs_to_wire",
    "reference_frame_from_proto",
    "reference_frame_from_wire",
    "reference_frame_to_proto",
    "reference_frame_to_wire",
]


def _to_proto[P: Message](model: BaseModel, msg: P) -> P:
    # exclude_none drops unset optionals so they map to proto field absence; required
    # scalars and closed-vocab strings are always present.
    json_format.ParseDict(model.model_dump(by_alias=True, mode="json", exclude_none=True), msg)
    return msg


def _from_proto[M: BaseModel](msg: Message, model: type[M]) -> M:
    data: dict[str, Any] = json_format.MessageToDict(
        msg,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=True,
    )
    return model.model_validate(data)


# --- ReferenceFrame ---------------------------------------------------------------


def reference_frame_to_proto(frame: ReferenceFrame) -> units_pb2.ReferenceFrame:
    return _to_proto(frame, units_pb2.ReferenceFrame())


def reference_frame_from_proto(msg: units_pb2.ReferenceFrame) -> ReferenceFrame:
    return _from_proto(msg, ReferenceFrame)


def reference_frame_to_wire(frame: ReferenceFrame) -> bytes:
    return reference_frame_to_proto(frame).SerializeToString(deterministic=True)


def reference_frame_from_wire(data: bytes) -> ReferenceFrame:
    msg = units_pb2.ReferenceFrame()
    msg.ParseFromString(data)
    return reference_frame_from_proto(msg)


# --- PlanetaryCRS ------------------------------------------------------------------


def planetary_crs_to_proto(crs: PlanetaryCRS) -> units_pb2.PlanetaryCRS:
    return _to_proto(crs, units_pb2.PlanetaryCRS())


def planetary_crs_from_proto(msg: units_pb2.PlanetaryCRS) -> PlanetaryCRS:
    return _from_proto(msg, PlanetaryCRS)


def planetary_crs_to_wire(crs: PlanetaryCRS) -> bytes:
    return planetary_crs_to_proto(crs).SerializeToString(deterministic=True)


def planetary_crs_from_wire(data: bytes) -> PlanetaryCRS:
    msg = units_pb2.PlanetaryCRS()
    msg.ParseFromString(data)
    return planetary_crs_from_proto(msg)


# --- Epoch -------------------------------------------------------------------------


def epoch_to_proto(epoch: Epoch) -> units_pb2.Epoch:
    return _to_proto(epoch, units_pb2.Epoch())


def epoch_from_proto(msg: units_pb2.Epoch) -> Epoch:
    return _from_proto(msg, Epoch)


def epoch_to_wire(epoch: Epoch) -> bytes:
    return epoch_to_proto(epoch).SerializeToString(deterministic=True)


def epoch_from_wire(data: bytes) -> Epoch:
    msg = units_pb2.Epoch()
    msg.ParseFromString(data)
    return epoch_from_proto(msg)


# --- EpochWindow -------------------------------------------------------------------


def epoch_window_to_proto(window: EpochWindow) -> units_pb2.EpochWindow:
    return _to_proto(window, units_pb2.EpochWindow())


def epoch_window_from_proto(msg: units_pb2.EpochWindow) -> EpochWindow:
    return _from_proto(msg, EpochWindow)


def epoch_window_to_wire(window: EpochWindow) -> bytes:
    return epoch_window_to_proto(window).SerializeToString(deterministic=True)


def epoch_window_from_wire(data: bytes) -> EpochWindow:
    msg = units_pb2.EpochWindow()
    msg.ParseFromString(data)
    return epoch_window_from_proto(msg)
