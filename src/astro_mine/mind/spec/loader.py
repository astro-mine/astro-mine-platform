# SPDX-License-Identifier: Apache-2.0
"""Stack-spec loading and validation (RM-P1-MIND-01).

Pipeline (``load_stack_spec``), mirroring Core's manifest / objective loaders:

1. parse YAML/JSON (YAML is a JSON superset, so one parser handles both);
2. **structural** validation against the canonical JSON Schema (rejects unknown/typo'd
   fields, bad enum values, missing required, wrong types);
3. build the typed :class:`~astro_mine.mind.spec.model.StackSpecDocument` (whose model
   validators reject duplicate tier roles);
4. **semantic** checks that exceed JSON Schema's expressiveness — trigger consistency
   (a ``periodic`` replan trigger MUST carry ``every_ticks``).

Graph-validity — that a tier's ``plugin`` resolves in the registry, is a ``policy`` kind,
composes in canonical order, and that the shield is present — is **not** checked here; it
depends on a registry and is the composer's job
(:mod:`astro_mine.mind.compose.composer`), the same split Core draws between its manifest
loader and its plugin registry.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from astro_mine.mind.spec.enums import ReplanTriggerKind
from astro_mine.mind.spec.model import StackSpecDocument

__all__ = [
    "StackSpecError",
    "StackSpecValidationError",
    "load_schema",
    "load_stack_spec",
    "validate_stack_spec",
]

_SCHEMA_RESOURCE = "schema/stack_spec.schema.json"


class StackSpecError(Exception):
    """Base class for stack-spec errors."""


class StackSpecValidationError(StackSpecError):
    """Raised when a stack-spec document fails structural or semantic validation."""


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    """Return the canonical stack-spec JSON Schema (shipped inside the package)."""
    text = (
        resources.files("astro_mine.mind.spec")
        .joinpath(_SCHEMA_RESOURCE)
        .read_text(encoding="utf-8")
    )
    schema: dict[str, Any] = json.loads(text)
    return schema


@lru_cache(maxsize=1)
def _validator() -> Draft202012Validator:
    return Draft202012Validator(load_schema())


def _parse(source: str | bytes) -> Any:
    if isinstance(source, bytes):
        source = source.decode("utf-8")
    return yaml.safe_load(source)


def _check_structural(data: Any) -> None:
    if not isinstance(data, dict):
        raise StackSpecValidationError("stack spec must be a YAML/JSON mapping")
    errors = sorted(_validator().iter_errors(data), key=lambda e: list(e.path))
    if errors:
        rendered = "; ".join(
            f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors[:10]
        )
        raise StackSpecValidationError(f"stack spec failed schema validation: {rendered}")


def _check_semantics(doc: StackSpecDocument) -> None:
    spec = doc.stack_spec
    for tier in spec.tiers:
        for trigger in tier.replan_triggers:
            if trigger.kind is ReplanTriggerKind.PERIODIC and trigger.every_ticks is None:
                raise StackSpecValidationError(
                    f"stack spec {spec.id!r}: tier {tier.role.value!r} has a 'periodic' "
                    f"replan trigger without 'every_ticks'"
                )
            if trigger.kind is ReplanTriggerKind.PLAN_EXPIRED and tier.validity_horizon_s is None:
                raise StackSpecValidationError(
                    f"stack spec {spec.id!r}: tier {tier.role.value!r} has a 'plan_expired' "
                    f"replan trigger but no 'validity_horizon_s' to expire against"
                )


def load_stack_spec(source: str | bytes) -> StackSpecDocument:
    """Parse, validate, and return a typed stack-spec document.

    Raises :class:`StackSpecValidationError` on any structural or semantic failure.
    """
    data = _parse(source)
    _check_structural(data)
    try:
        doc = StackSpecDocument.model_validate(data)
    except ValidationError as exc:
        raise StackSpecValidationError(f"stack spec failed model validation: {exc}") from exc
    _check_semantics(doc)
    return doc


def validate_stack_spec(document: Any) -> None:
    """Validate a stack-spec document without returning it.

    Accepts raw YAML/JSON text/bytes, a parsed mapping, or a
    :class:`~astro_mine.mind.spec.model.StackSpecDocument`. Raises
    :class:`StackSpecValidationError` on failure.
    """
    if isinstance(document, StackSpecDocument):
        _check_structural(document.model_dump(mode="json"))
        _check_semantics(document)
        return
    if isinstance(document, str | bytes):
        load_stack_spec(document)
        return
    if isinstance(document, dict):
        _check_structural(document)
        try:
            doc = StackSpecDocument.model_validate(document)
        except ValidationError as exc:
            raise StackSpecValidationError(f"stack spec failed model validation: {exc}") from exc
        _check_semantics(doc)
        return
    raise StackSpecValidationError(f"cannot validate object of type {type(document).__name__}")
