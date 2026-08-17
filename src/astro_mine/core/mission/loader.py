# SPDX-License-Identifier: Apache-2.0
"""MissionSpec loading and validation (RM-P1-CORE-04 / RFC-0001).

Pipeline (``load_mission``), mirroring SADF/ObjectiveSpec:

1. parse YAML/JSON (YAML is a JSON superset, so one parser handles both);
2. **structural** validation against the canonical JSON Schema (rejects unknown/typo'd
   fields, bad regime/maneuver-type values, missing required, wrong types);
3. build the typed :class:`~astro_mine.core.mission.model.MissionDocument`.

There are no semantic checks beyond structure in v0.1: the mission hooks are **reserved
schema, no mechanism** (mission-model.md §3), so Core validates the shape and nothing
about phase sequencing or trajectory feasibility (those are Phase-3 implementations).
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from astro_mine.core.mission.model import MissionDocument
from astro_mine.core.schemas import schema_registry

__all__ = [
    "MissionError",
    "MissionValidationError",
    "load_mission",
    "load_schema",
    "validate_mission",
]

_SCHEMA_RESOURCE = "schema/mission.schema.json"


class MissionError(Exception):
    """Base class for mission errors."""


class MissionValidationError(MissionError):
    """Raised when a mission document fails structural validation."""


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    """Return the canonical MissionSpec JSON Schema (shipped inside the package)."""
    text = (
        resources.files("astro_mine.core.mission")
        .joinpath(_SCHEMA_RESOURCE)
        .read_text(encoding="utf-8")
    )
    schema: dict[str, Any] = json.loads(text)
    return schema


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    # mission.schema.json $refs units.schema.json across files (RFC-0007); the units
    # registry resolves that cross-file reference for the validator.
    schema = load_schema()
    return Draft202012Validator(schema, registry=schema_registry(schema))


def _parse(source: str | bytes) -> Any:
    if isinstance(source, bytes):
        source = source.decode("utf-8")
    return yaml.safe_load(source)


def _check_structural(data: Any) -> None:
    if not isinstance(data, dict):
        raise MissionValidationError("mission document must be a YAML/JSON mapping")
    errors = sorted(_validator().iter_errors(data), key=lambda e: list(e.path))
    if errors:
        rendered = "; ".join(
            f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors[:10]
        )
        raise MissionValidationError(f"mission failed schema validation: {rendered}")


def load_mission(source: str | bytes) -> MissionDocument:
    """Parse, validate, and return a typed mission document.

    Raises :class:`MissionValidationError` on any structural failure.
    """
    data = _parse(source)
    _check_structural(data)
    try:
        return MissionDocument.model_validate(data)
    except ValidationError as exc:
        raise MissionValidationError(f"mission failed model validation: {exc}") from exc


def validate_mission(document: Any) -> None:
    """Validate a mission document without returning it.

    Accepts raw YAML/JSON text/bytes, a parsed mapping, or a
    :class:`~astro_mine.core.mission.model.MissionDocument`. Raises
    :class:`MissionValidationError` on failure.
    """
    if isinstance(document, MissionDocument):
        _check_structural(document.model_dump(by_alias=True, mode="json"))
        return
    if isinstance(document, str | bytes):
        load_mission(document)
        return
    if isinstance(document, dict):
        _check_structural(document)
        try:
            MissionDocument.model_validate(document)
        except ValidationError as exc:
            raise MissionValidationError(f"mission failed model validation: {exc}") from exc
        return
    raise MissionValidationError(f"cannot validate object of type {type(document).__name__}")
