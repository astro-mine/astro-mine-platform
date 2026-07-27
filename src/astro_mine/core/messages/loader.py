"""Message-catalog validation (control plane) and union semantic checks.

Mirrors the SADF/ObjectiveSpec loaders: structural validation against the canonical
JSON Schema (``messages.schema.json``) followed by semantic checks that exceed JSON
Schema's expressiveness — here, the **tagged-union consistency** of
:class:`~astro_mine.core.messages.model.Action` and
:class:`~astro_mine.core.messages.model.TaskDirective` (exactly the payload that
matches the discriminant), and contact-interval ordering.

The per-tick observation family is Cap'n Proto (see
:mod:`astro_mine.core.messages.hotpath`); its validation is the round-trip itself.
"""

from __future__ import annotations

import json
from functools import cache, lru_cache
from importlib import resources
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ValidationError

from astro_mine.core.messages.enums import ActionKind, TaskKind
from astro_mine.core.messages.model import ActionBatch, ContactPlan, TaskDirective
from astro_mine.core.schemas import schema_registry

__all__ = [
    "MessagesError",
    "MessagesValidationError",
    "load_action_batch",
    "load_contact_plan",
    "load_schema",
    "validate_action_batch",
    "validate_contact_plan",
]

_SCHEMA_RESOURCE = "schema/messages.schema.json"

_ACTION_KIND_FIELD = {
    ActionKind.ACTUATOR: "actuator",
    ActionKind.MODE: "mode",
    ActionKind.TASK: "task",
}
_TASK_KIND_FIELD = {
    TaskKind.GOTO: "goto",
    TaskKind.SAMPLE: "sample",
    TaskKind.EXCAVATE: "excavate",
    TaskKind.HAUL: "haul",
    TaskKind.DOCK: "dock",
    TaskKind.HOP: "hop",
    TaskKind.CHARGE: "charge",
    TaskKind.PROSPECT: "prospect",
}
_TYPED_TASK_FIELDS = tuple(_TASK_KIND_FIELD.values())
_PARAMLESS_TASKS = {TaskKind.DEPLOY, TaskKind.STANDBY}


class MessagesError(Exception):
    """Base class for message-catalog errors."""


class MessagesValidationError(MessagesError):
    """Raised when a message fails structural or semantic validation."""


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    """Return the canonical control-plane message JSON Schema (shipped in-package)."""
    text = (
        resources.files("astro_mine.core.messages")
        .joinpath(_SCHEMA_RESOURCE)
        .read_text(encoding="utf-8")
    )
    schema: dict[str, Any] = json.loads(text)
    return schema


@cache
def _root_validator(root: str) -> Draft202012Validator:
    # messages.schema.json $refs units.schema.json across files (RFC-0007); the units
    # registry resolves that cross-file reference for the validator.
    schema = load_schema()
    return Draft202012Validator(
        {"$ref": schema["$id"] + f"#/$defs/{root}"}, registry=schema_registry(schema)
    )


def _parse(source: str | bytes) -> Any:
    if isinstance(source, bytes):
        source = source.decode("utf-8")
    return yaml.safe_load(source)


def _check_structural(data: Any, root: str) -> None:
    if not isinstance(data, dict):
        raise MessagesValidationError(f"{root} document must be a YAML/JSON mapping")
    errors = sorted(_root_validator(root).iter_errors(data), key=lambda e: list(e.path))
    if errors:
        rendered = "; ".join(
            f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors[:10]
        )
        raise MessagesValidationError(f"{root} failed schema validation: {rendered}")


def _check_task_semantics(task: TaskDirective, where: str) -> None:
    set_typed = [f for f in _TYPED_TASK_FIELDS if getattr(task, f) is not None]
    if task.task_kind == TaskKind.CUSTOM:
        if task.directive is None:
            raise MessagesValidationError(f"{where}: task_kind=custom requires a 'directive'")
        if set_typed:
            raise MessagesValidationError(
                f"{where}: task_kind=custom must not set a typed task payload {set_typed}"
            )
    elif task.task_kind in _PARAMLESS_TASKS:
        if set_typed:
            raise MessagesValidationError(
                f"{where}: task_kind={task.task_kind} carries no payload, got {set_typed}"
            )
    else:
        expected = _TASK_KIND_FIELD[task.task_kind]
        if set_typed != [expected]:
            raise MessagesValidationError(
                f"{where}: task_kind={task.task_kind} requires exactly the '{expected}' payload, "
                f"got {set_typed or 'none'}"
            )


def _check_action_semantics(batch: ActionBatch) -> None:
    for i, action in enumerate(batch.actions):
        where = f"actions[{i}]"
        expected = _ACTION_KIND_FIELD[action.kind]
        set_fields = [f for f in ("actuator", "mode", "task") if getattr(action, f) is not None]
        if set_fields != [expected]:
            raise MessagesValidationError(
                f"{where}: kind={action.kind} requires exactly the '{expected}' payload, "
                f"got {set_fields or 'none'}"
            )
        if action.task is not None:
            _check_task_semantics(action.task, f"{where}.task")


def _check_contact_plan_semantics(plan: ContactPlan) -> None:
    for i, iv in enumerate(plan.intervals):
        if iv.end_tdb_s < iv.start_tdb_s:
            raise MessagesValidationError(
                f"intervals[{i}]: end_tdb_s ({iv.end_tdb_s}) precedes "
                f"start_tdb_s ({iv.start_tdb_s})"
            )


def _build[M: BaseModel](data: Any, root: str, model: type[M]) -> M:
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise MessagesValidationError(f"{root} failed model validation: {exc}") from exc


def load_action_batch(source: str | bytes) -> ActionBatch:
    """Parse, validate, and return a typed :class:`ActionBatch`."""
    data = _parse(source)
    _check_structural(data, "ActionBatch")
    batch = _build(data, "ActionBatch", ActionBatch)
    _check_action_semantics(batch)
    return batch


def load_contact_plan(source: str | bytes) -> ContactPlan:
    """Parse, validate, and return a typed :class:`ContactPlan`."""
    data = _parse(source)
    _check_structural(data, "ContactPlan")
    plan = _build(data, "ContactPlan", ContactPlan)
    _check_contact_plan_semantics(plan)
    return plan


def validate_action_batch(document: Any) -> None:
    """Validate an :class:`ActionBatch` (model, mapping, or YAML/JSON text/bytes)."""
    if isinstance(document, ActionBatch):
        _check_structural(document.model_dump(by_alias=True, mode="json"), "ActionBatch")
        _check_action_semantics(document)
        return
    if isinstance(document, str | bytes):
        load_action_batch(document)
        return
    if isinstance(document, dict):
        _check_structural(document, "ActionBatch")
        _check_action_semantics(_build(document, "ActionBatch", ActionBatch))
        return
    raise MessagesValidationError(f"cannot validate object of type {type(document).__name__}")


def validate_contact_plan(document: Any) -> None:
    """Validate a :class:`ContactPlan` (model, mapping, or YAML/JSON text/bytes)."""
    if isinstance(document, ContactPlan):
        _check_structural(document.model_dump(by_alias=True, mode="json"), "ContactPlan")
        _check_contact_plan_semantics(document)
        return
    if isinstance(document, str | bytes):
        load_contact_plan(document)
        return
    if isinstance(document, dict):
        _check_structural(document, "ContactPlan")
        _check_contact_plan_semantics(_build(document, "ContactPlan", ContactPlan))
        return
    raise MessagesValidationError(f"cannot validate object of type {type(document).__name__}")
