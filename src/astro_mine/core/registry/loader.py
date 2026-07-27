"""Plugin manifest loading, validation, and the dual-use (export-control) gate.

Pipeline (``load_manifest``), mirroring SADF/ObjectiveSpec:

1. parse YAML/JSON (YAML is a JSON superset, so one parser handles both);
2. **structural** validation against the canonical JSON Schema (rejects unknown/typo'd
   fields, bad enum values, missing required, etc.);
3. build the typed :class:`~astro_mine.core.registry.model.ManifestDocument`;
4. **semantic** checks that exceed JSON Schema's expressiveness — the reserved/gated
   capability-tag gate (mirroring :func:`astro_mine.core.sadf.loader` so a capability
   means the same at the asset and the plugin boundary) and well-formedness of the
   declared Core interface versions.

The semantic checks run for *both* ``load_manifest`` and ``validate_manifest`` so JSON
Schema and Pydantic stay a single structural contract. Load-time **version negotiation**
against *this* Core and the **signature** gate are separate (they depend on the running
Core and the registry's policy) and live in
:mod:`astro_mine.core.registry.registry`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from astro_mine.core.compat import parse_version
from astro_mine.core.registry.model import ManifestDocument
from astro_mine.core.sadf.enums import GATED_CAPABILITY_TAGS

__all__ = [
    "ManifestValidationError",
    "RegistryError",
    "load_manifest",
    "load_schema",
    "validate_manifest",
]

_SCHEMA_RESOURCE = "schema/manifest.schema.json"


class RegistryError(Exception):
    """Base class for plugin-registry errors."""


class ManifestValidationError(RegistryError):
    """Raised when a plugin manifest fails structural or semantic validation."""


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    """Return the canonical plugin-manifest JSON Schema (shipped inside the package)."""
    text = (
        resources.files("astro_mine.core.registry")
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
        raise ManifestValidationError("plugin manifest must be a YAML/JSON mapping")
    errors = sorted(_validator().iter_errors(data), key=lambda e: list(e.path))
    if errors:
        rendered = "; ".join(
            f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors[:10]
        )
        raise ManifestValidationError(f"plugin manifest failed schema validation: {rendered}")


def _check_semantics(doc: ManifestDocument) -> None:
    manifest = doc.manifest
    gated = [c for c in manifest.capability_tags if c in GATED_CAPABILITY_TAGS]
    if gated:
        names = ", ".join(sorted(str(c) for c in gated))
        raise ManifestValidationError(
            f"plugin {manifest.name!r} declares reserved/gated capability tag(s) not "
            f"permitted in the open commons: {names} (RFC-0001 §6; conventions.md §12)"
        )
    for interface, version in manifest.core_interfaces.items():
        try:
            parse_version(version)
        except ValueError as exc:
            raise ManifestValidationError(
                f"plugin {manifest.name!r}: core_interfaces[{interface!r}] is not a "
                f"MAJOR.MINOR.PATCH version: {exc}"
            ) from exc


def load_manifest(source: str | bytes) -> ManifestDocument:
    """Parse, validate, and return a typed plugin-manifest document.

    Raises :class:`ManifestValidationError` on any structural or semantic failure.
    """
    data = _parse(source)
    _check_structural(data)
    try:
        doc = ManifestDocument.model_validate(data)
    except ValidationError as exc:
        raise ManifestValidationError(f"plugin manifest failed model validation: {exc}") from exc
    _check_semantics(doc)
    return doc


def validate_manifest(document: Any) -> None:
    """Validate a plugin-manifest document without returning it.

    Accepts raw YAML/JSON text/bytes, a parsed mapping, or a
    :class:`~astro_mine.core.registry.model.ManifestDocument`. Raises
    :class:`ManifestValidationError` on failure.
    """
    if isinstance(document, ManifestDocument):
        _check_structural(document.model_dump(by_alias=True, mode="json"))
        _check_semantics(document)
        return
    if isinstance(document, str | bytes):
        load_manifest(document)
        return
    if isinstance(document, dict):
        _check_structural(document)
        try:
            doc = ManifestDocument.model_validate(document)
        except ValidationError as exc:
            raise ManifestValidationError(
                f"plugin manifest failed model validation: {exc}"
            ) from exc
        _check_semantics(doc)
        return
    raise ManifestValidationError(f"cannot validate object of type {type(document).__name__}")
