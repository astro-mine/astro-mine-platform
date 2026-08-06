"""Plugin manifest loading, validation, and the dual-use (export-control) gate.

Two entry points, because a manifest has two forms and conflating them is how a publisher and a
reader end up disagreeing about what is stored (astro-mine-platform#14). ``load_manifest`` reads
the authored **document** — the YAML/JSON file carrying ``manifest_version``.
``load_plugin_manifest`` reads the **bare manifest**, which is what a publisher stores as an OCI
config blob. Both run the same pipeline and the same checks; they differ only in whether the
envelope is expected.

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
from astro_mine.core.registry.model import MANIFEST_VERSION, ManifestDocument, PluginManifest
from astro_mine.core.sadf.enums import GATED_CAPABILITY_TAGS

__all__ = [
    "ManifestValidationError",
    "RegistryError",
    "load_manifest",
    "load_plugin_manifest",
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
    """Parse, validate, and return a typed plugin-manifest **document**.

    For the authored file — the YAML/JSON a publisher writes, carrying ``manifest_version``.
    For the *stored* form, which is a bare manifest, use :func:`load_plugin_manifest`.

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


def load_plugin_manifest(source: str | bytes) -> PluginManifest:
    """Parse, validate, and return a **bare** plugin manifest — the stored config-blob form.

    **The document envelope and the config blob are different things, and this is the one for
    the wire.** ``ManifestDocument`` versions a *file*: ``manifest_version`` pins the schema minor
    of something a human authored, and :func:`load_manifest` is what reads it. What a publisher
    *stores* is the manifest itself — ``hub.md`` §2 principle 2, "Hub indexes artifacts by the Core
    plugin manifest" — because the OCI ``artifactType``
    (``application/vnd.astro-mine.<kind>.v1``) already carries the version discriminator on that
    side, and two of them would be one too many.

    Until now Core offered no way to read that form *with its checks*, so every reader of a config
    blob reached for ``PluginManifest.model_validate_json`` and got Pydantic alone — no schema
    validation, and **no gated-capability-tag gate**. That is tolerable for a first-party artifact
    a deployment published itself, and it is not tolerable on
    [Bench](https://github.com/astro-mine/docs/blob/main/architecture/bench.md)'s community-
    submission intake, which is precisely where a third party's manifest arrives. This is the
    missing primitive, not a convenience wrapper.

    Validation goes through the document schema by wrapping — the same move
    :meth:`~astro_mine.core.registry.registry.PluginRegistry.register` already makes for a bare
    manifest. That keeps JSON Schema and Pydantic a single structural contract, which is this
    module's stated invariant; a second schema for the unwrapped shape would be two contracts to
    keep in step, and they would not stay in step.

    Raises :class:`ManifestValidationError` on any structural or semantic failure — and names the
    envelope case explicitly, because being handed the wrong one of these two shapes is the whole
    of astro-mine-platform#14 and Pydantic's own report for it is misleading: a document fails as
    five missing required fields (``name``, ``version``, ``kind``, …), which reads like a corrupt
    manifest rather than a well-formed one at the wrong level.
    """
    data = _parse(source)
    if not isinstance(data, dict):
        raise ManifestValidationError("plugin manifest must be a YAML/JSON mapping")
    if "manifest_version" in data and "manifest" in data:
        raise ManifestValidationError(
            "expected a bare plugin manifest and got a manifest *document* (it carries "
            "'manifest_version' and 'manifest'). A stored config blob is the manifest itself "
            "(hub.md §2 principle 2); use load_manifest() for an authored document, or pass "
            "the document's .manifest here."
        )
    try:
        manifest = PluginManifest.model_validate(data)
    except ValidationError as exc:
        raise ManifestValidationError(f"plugin manifest failed model validation: {exc}") from exc
    doc = ManifestDocument(manifest_version=MANIFEST_VERSION, manifest=manifest)
    _check_structural(doc.model_dump(by_alias=True, mode="json"))
    _check_semantics(doc)
    return doc.manifest


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
