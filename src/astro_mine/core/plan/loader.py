"""Plan document loading, validation, and semantic checks (RFC-0006).

Pipeline (``load_plan``), mirroring the ObjectiveSpec loader:

1. parse YAML/JSON (YAML is a JSON superset, so one parser handles both);
2. **structural** validation against the canonical JSON Schema (rejects unknown/typo'd fields,
   bad types, missing required);
3. build the typed :class:`~astro_mine.core.plan.model.PlanDocument`;
4. **semantic** checks that exceed JSON Schema's expressiveness — non-empty contingency
   trigger/action labels, and unique branch triggers within a plan.

The semantic checks run for both ``load_plan`` and ``validate_plan`` so JSON Schema and Pydantic
stay a single structural contract.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from astro_mine.core.plan.model import PlanDocument

__all__ = [
    "PlanError",
    "PlanValidationError",
    "load_plan",
    "load_schema",
    "validate_plan",
]

_SCHEMA_RESOURCE = "schema/plan.schema.json"


class PlanError(Exception):
    """Base class for plan-document errors."""


class PlanValidationError(PlanError):
    """Raised when a plan document fails structural or semantic validation."""


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    """Return the canonical Plan JSON Schema (shipped inside the package)."""
    text = (
        resources.files("astro_mine.core.plan")
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
        raise PlanValidationError("plan document must be a YAML/JSON mapping")
    errors = sorted(_validator().iter_errors(data), key=lambda e: list(e.path))
    if errors:
        rendered = "; ".join(
            f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors[:10]
        )
        raise PlanValidationError(f"plan failed schema validation: {rendered}")


def _check_semantics(doc: PlanDocument) -> None:
    triggers = [branch.trigger for branch in doc.plan.branches]
    for branch in doc.plan.branches:
        if not branch.trigger.strip():
            raise PlanValidationError("contingency branch trigger must be a non-empty label")
        if not branch.action.strip():
            raise PlanValidationError(
                f"contingency branch {branch.trigger!r}: action must be a non-empty label"
            )
    dupes = sorted({t for t in triggers if triggers.count(t) > 1})
    if dupes:
        raise PlanValidationError(f"duplicate contingency trigger(s): {', '.join(dupes)}")


def load_plan(source: str | bytes) -> PlanDocument:
    """Parse, validate, and return a typed plan document.

    Raises :class:`PlanValidationError` on any structural or semantic failure.
    """
    data = _parse(source)
    _check_structural(data)
    try:
        doc = PlanDocument.model_validate(data)
    except ValidationError as exc:
        raise PlanValidationError(f"plan failed model validation: {exc}") from exc
    _check_semantics(doc)
    return doc


def validate_plan(document: Any) -> None:
    """Validate a plan document without returning it.

    Accepts raw YAML/JSON text/bytes, a parsed mapping, or a
    :class:`~astro_mine.core.plan.model.PlanDocument`. Raises :class:`PlanValidationError`.
    """
    if isinstance(document, PlanDocument):
        _check_structural(document.model_dump(mode="json"))
        _check_semantics(document)
        return
    if isinstance(document, str | bytes):
        load_plan(document)
        return
    if isinstance(document, dict):
        _check_structural(document)
        try:
            doc = PlanDocument.model_validate(document)
        except ValidationError as exc:
            raise PlanValidationError(f"plan failed model validation: {exc}") from exc
        _check_semantics(doc)
        return
    raise PlanValidationError(f"cannot validate object of type {type(document).__name__}")
