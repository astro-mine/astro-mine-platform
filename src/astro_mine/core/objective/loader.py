"""ObjectiveSpec loading, validation, and semantic checks.

Pipeline (``load_objective``), mirroring SADF (astro_mine.core.sadf.loader):

1. parse YAML/JSON (YAML is a JSON superset, so one parser handles both);
2. **structural** validation against the canonical JSON Schema (rejects unknown/
   typo'd fields, bad enum values, missing required, negative tolerance, etc.);
3. build the typed :class:`~astro_mine.core.objective.model.ObjectiveDocument`;
4. **semantic** checks that exceed JSON Schema's expressiveness — here, unique
   success-criterion ids.

The semantic checks run for *both* ``load_objective`` and ``validate_objective`` so
JSON Schema and Pydantic stay a single structural contract.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from astro_mine.core.objective.enums import WindowKind
from astro_mine.core.objective.model import ObjectiveDocument

__all__ = [
    "ObjectiveError",
    "ObjectiveValidationError",
    "load_objective",
    "load_schema",
    "validate_objective",
]

_SCHEMA_RESOURCE = "schema/objective.schema.json"


class ObjectiveError(Exception):
    """Base class for ObjectiveSpec errors."""


class ObjectiveValidationError(ObjectiveError):
    """Raised when an objective document fails structural or semantic validation."""


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    """Return the canonical ObjectiveSpec JSON Schema (shipped inside the package)."""
    text = (
        resources.files("astro_mine.core.objective")
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
        raise ObjectiveValidationError("ObjectiveSpec document must be a YAML/JSON mapping")
    errors = sorted(_validator().iter_errors(data), key=lambda e: list(e.path))
    if errors:
        rendered = "; ".join(
            f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors[:10]
        )
        raise ObjectiveValidationError(f"ObjectiveSpec failed schema validation: {rendered}")


def _check_semantics(doc: ObjectiveDocument) -> None:
    ids = [c.id for c in doc.objective.success_criteria]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ObjectiveValidationError(
            f"duplicate success_criteria id(s): {', '.join(dupes)} — criterion ids must be unique"
        )
    for c in doc.objective.success_criteria:
        # The binding must name a real Bench metric and an explicit SI unit — a blank
        # string is not a metric key. Reject it at the boundary (fail early and loudly,
        # core.md §2 principle 7) rather than let Bench fail to resolve it downstream.
        if not c.binding.metric.strip():
            raise ObjectiveValidationError(
                f"criterion {c.id!r}: metric must be a non-empty Bench metric key"
            )
        if not c.binding.unit.strip():
            raise ObjectiveValidationError(
                f"criterion {c.id!r}: unit must be an explicit, non-empty SI unit"
            )
        window = c.binding.evaluation_window
        if window is None:
            continue
        if window.kind == WindowKind.ROLLING and window.duration_s is None:
            raise ObjectiveValidationError(
                f"criterion {c.id!r}: a rolling evaluation_window requires duration_s"
            )
        if window.kind != WindowKind.ROLLING and window.duration_s is not None:
            raise ObjectiveValidationError(
                f"criterion {c.id!r}: a {window.kind} evaluation_window must not set duration_s"
            )


def load_objective(source: str | bytes) -> ObjectiveDocument:
    """Parse, validate, and return a typed objective document.

    Raises :class:`ObjectiveValidationError` on any structural or semantic failure.
    """
    data = _parse(source)
    _check_structural(data)
    try:
        doc = ObjectiveDocument.model_validate(data)
    except ValidationError as exc:
        raise ObjectiveValidationError(f"ObjectiveSpec failed model validation: {exc}") from exc
    _check_semantics(doc)
    return doc


def validate_objective(document: Any) -> None:
    """Validate an objective document without returning it.

    Accepts raw YAML/JSON text/bytes, a parsed mapping, or an
    :class:`~astro_mine.core.objective.model.ObjectiveDocument`. Raises
    :class:`ObjectiveValidationError` on failure.
    """
    if isinstance(document, ObjectiveDocument):
        _check_structural(document.model_dump(by_alias=True, mode="json"))
        _check_semantics(document)
        return
    if isinstance(document, str | bytes):
        load_objective(document)
        return
    if isinstance(document, dict):
        _check_structural(document)
        try:
            doc = ObjectiveDocument.model_validate(document)
        except ValidationError as exc:
            raise ObjectiveValidationError(f"ObjectiveSpec failed model validation: {exc}") from exc
        _check_semantics(doc)
        return
    raise ObjectiveValidationError(f"cannot validate object of type {type(document).__name__}")
