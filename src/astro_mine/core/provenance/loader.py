"""Run-provenance loading and validation.

Pipeline (``load_run_provenance``), mirroring SADF/ObjectiveSpec/manifest:

1. parse YAML/JSON (YAML is a JSON superset, so one parser handles both);
2. **structural** validation against the canonical JSON Schema (rejects unknown/typo'd
   fields, bad enum values, missing required, wrong types);
3. build the typed :class:`~astro_mine.core.provenance.model.RunProvenanceDocument`.

Unlike the manifest/objective loaders there are **no semantic checks beyond structure**
in v0.1: JSON Schema fully expresses the run-provenance contract (there is no cross-field
rule Core should enforce — e.g. ``within_budget`` is the producer's verdict, not
something Core recomputes, per the mechanism-not-policy rule in core.md §2). A semantic
hook is added here additively if that ever changes.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from astro_mine.core.provenance.model import RunProvenanceDocument

__all__ = [
    "RunProvenanceError",
    "RunProvenanceValidationError",
    "load_run_provenance",
    "load_schema",
    "validate_run_provenance",
]

_SCHEMA_RESOURCE = "schema/run_provenance.schema.json"


class RunProvenanceError(Exception):
    """Base class for run-provenance errors."""


class RunProvenanceValidationError(RunProvenanceError):
    """Raised when a run-provenance document fails structural validation."""


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    """Return the canonical run-provenance JSON Schema (shipped inside the package)."""
    text = (
        resources.files("astro_mine.core.provenance")
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
        raise RunProvenanceValidationError("run provenance must be a YAML/JSON mapping")
    errors = sorted(_validator().iter_errors(data), key=lambda e: list(e.path))
    if errors:
        rendered = "; ".join(
            f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors[:10]
        )
        raise RunProvenanceValidationError(f"run provenance failed schema validation: {rendered}")


def load_run_provenance(source: str | bytes) -> RunProvenanceDocument:
    """Parse, validate, and return a typed run-provenance document.

    Raises :class:`RunProvenanceValidationError` on any structural failure.
    """
    data = _parse(source)
    _check_structural(data)
    try:
        return RunProvenanceDocument.model_validate(data)
    except ValidationError as exc:
        raise RunProvenanceValidationError(
            f"run provenance failed model validation: {exc}"
        ) from exc


def validate_run_provenance(document: Any) -> None:
    """Validate a run-provenance document without returning it.

    Accepts raw YAML/JSON text/bytes, a parsed mapping, or a
    :class:`~astro_mine.core.provenance.model.RunProvenanceDocument`. Raises
    :class:`RunProvenanceValidationError` on failure.
    """
    if isinstance(document, RunProvenanceDocument):
        _check_structural(document.model_dump(mode="json"))
        return
    if isinstance(document, str | bytes):
        load_run_provenance(document)
        return
    if isinstance(document, dict):
        _check_structural(document)
        try:
            RunProvenanceDocument.model_validate(document)
        except ValidationError as exc:
            raise RunProvenanceValidationError(
                f"run provenance failed model validation: {exc}"
            ) from exc
        return
    raise RunProvenanceValidationError(f"cannot validate object of type {type(document).__name__}")
