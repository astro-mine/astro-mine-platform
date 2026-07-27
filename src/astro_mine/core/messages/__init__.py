"""Message-schema catalog — the typed cross-component vocabulary (RM-P0-CORE-04).

The typed messages every plane exchanges, co-designed with the Environment API
(RM-P0-CORE-02). Encoding follows latency class (conventions.md §3):

- **hot path / per-tick** — the observation family (:class:`Observation`,
  :class:`StateSample`, :class:`SensorReading`, :class:`CommsObservationMask`) in **Cap'n Proto**,
  with zero-copy decode via :mod:`~astro_mine.core.messages.hotpath`
  (:func:`to_bytes` / :func:`from_bytes` / :func:`reader`);
- **control plane** — the action family (:class:`Action`, :class:`ActionBatch`, the
  typed :class:`TaskDirective`) and the contact-plan family (:class:`ContactPlan`,
  :class:`ContactInterval`, :class:`Route`) in **Protobuf**
  (:mod:`~astro_mine.core.messages.wire`), validated against the canonical JSON Schema.

The shared **ObjectiveSpec** contract lives in :mod:`astro_mine.core.objective`.

Backlog: RM-P0-CORE-04 — https://github.com/astro-mine/astro-mine-core/issues/4
"""

from __future__ import annotations

from astro_mine.core.messages import enums, hotpath, model
from astro_mine.core.messages.loader import (
    MessagesError,
    MessagesValidationError,
    load_action_batch,
    load_contact_plan,
    load_schema,
    validate_action_batch,
    validate_contact_plan,
)
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    ActuatorCommand,
    CommsObservationMask,
    ContactInterval,
    ContactNode,
    ContactPlan,
    ModeCommand,
    Observation,
    PeerLink,
    Route,
    SensorReading,
    StateSample,
    TaskDirective,
)
from astro_mine.core.messages.wire import (
    action_batch_from_wire,
    action_batch_to_wire,
    contact_plan_from_wire,
    contact_plan_to_wire,
)

__all__ = [
    "Action",
    "ActionBatch",
    "ActuatorCommand",
    "CommsObservationMask",
    "ContactInterval",
    "ContactNode",
    "ContactPlan",
    "MessagesError",
    "MessagesValidationError",
    "ModeCommand",
    "Observation",
    "PeerLink",
    "Route",
    "SensorReading",
    "StateSample",
    "TaskDirective",
    "action_batch_from_wire",
    "action_batch_to_wire",
    "contact_plan_from_wire",
    "contact_plan_to_wire",
    "enums",
    "hotpath",
    "load_action_batch",
    "load_contact_plan",
    "load_schema",
    "model",
    "validate_action_batch",
    "validate_contact_plan",
]
