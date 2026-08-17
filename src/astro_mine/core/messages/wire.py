# SPDX-License-Identifier: Apache-2.0
"""Control-plane messages <-> Protobuf wire form (canonical interchange encoding).

The action and contact-plan families serialize to Protobuf (the per-tick observation
family uses Cap'n Proto — see :mod:`astro_mine.core.messages.hotpath`). Conversion goes
through protobuf's ``json_format``; the model's JSON projection matches the proto field
names (both mirror the canonical JSON Schema). Serialization is deterministic, so the
encoding is **byte-stable** and a model -> bytes -> model round-trip is exact (the SADF
pattern).
"""

from __future__ import annotations

from typing import Any

from google.protobuf import json_format
from google.protobuf.message import Message
from pydantic import BaseModel

from astro_mine.core.messages._proto import messages_pb2
from astro_mine.core.messages.model import ActionBatch, ContactPlan

__all__ = [
    "action_batch_from_wire",
    "action_batch_to_proto",
    "action_batch_to_wire",
    "contact_plan_from_wire",
    "contact_plan_to_proto",
    "contact_plan_to_wire",
]


def _to_proto(model: ActionBatch | ContactPlan, msg: Message) -> Message:
    data: dict[str, Any] = model.model_dump(by_alias=True, mode="json", exclude_none=True)
    json_format.ParseDict(data, msg)
    return msg


def _from_proto[M: BaseModel](msg: Message, model: type[M]) -> M:
    data: dict[str, Any] = json_format.MessageToDict(
        msg,
        preserving_proto_field_name=True,
        always_print_fields_with_no_presence=True,
    )
    return model.model_validate(data)


def action_batch_to_proto(batch: ActionBatch) -> messages_pb2.ActionBatch:
    """Convert an action batch to its protobuf message."""
    return _to_proto(batch, messages_pb2.ActionBatch())  # type: ignore[return-value]


def action_batch_to_wire(batch: ActionBatch) -> bytes:
    """Serialize an action batch to its canonical, byte-stable wire form."""
    return action_batch_to_proto(batch).SerializeToString(deterministic=True)


def action_batch_from_wire(data: bytes) -> ActionBatch:
    """Parse a protobuf wire-form payload into a typed :class:`ActionBatch`."""
    msg = messages_pb2.ActionBatch()
    msg.ParseFromString(data)
    return _from_proto(msg, ActionBatch)


def contact_plan_to_proto(plan: ContactPlan) -> messages_pb2.ContactPlan:
    """Convert a contact plan to its protobuf message."""
    return _to_proto(plan, messages_pb2.ContactPlan())  # type: ignore[return-value]


def contact_plan_to_wire(plan: ContactPlan) -> bytes:
    """Serialize a contact plan to its canonical, byte-stable wire form."""
    return contact_plan_to_proto(plan).SerializeToString(deterministic=True)


def contact_plan_from_wire(data: bytes) -> ContactPlan:
    """Parse a protobuf wire-form payload into a typed :class:`ContactPlan`."""
    msg = messages_pb2.ContactPlan()
    msg.ParseFromString(data)
    return _from_proto(msg, ContactPlan)
