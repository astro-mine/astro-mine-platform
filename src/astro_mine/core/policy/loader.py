# SPDX-License-Identifier: Apache-2.0
"""PolicyPackage loading and validation (RM-P1-CORE-01).

Pipeline (``load_policy_package``), mirroring the manifest / run-provenance loaders:

1. parse YAML/JSON (YAML is a JSON superset, so one parser handles both);
2. **structural** validation against the canonical JSON Schema (rejects unknown/typo'd
   fields, bad enum values, missing required, wrong types);
3. build the typed :class:`~astro_mine.core.policy.model.PolicyPackageDocument`.

There are no semantic checks beyond structure in v0.1: JSON Schema fully expresses the
sidecar contract, and interface-version negotiation is the caller's step (via
:meth:`~astro_mine.core.policy.model.PolicyPackage.assert_core_compatible`), applied when
a host binds the package to a runnable policy — not a property of the document itself.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Any

import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from astro_mine.core.policy.model import PolicyPackageDocument

__all__ = [
    "PolicyPackageError",
    "PolicyPackageValidationError",
    "load_policy_package",
    "load_schema",
    "validate_policy_package",
]

_SCHEMA_RESOURCE = "schema/policy_package.schema.json"


class PolicyPackageError(Exception):
    """Base class for PolicyPackage errors."""


class PolicyPackageValidationError(PolicyPackageError):
    """Raised when a PolicyPackage document fails structural validation."""


@lru_cache(maxsize=1)
def load_schema() -> dict[str, Any]:
    """Return the canonical PolicyPackage JSON Schema (shipped inside the package)."""
    text = (
        resources.files("astro_mine.core.policy")
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
        raise PolicyPackageValidationError("PolicyPackage must be a YAML/JSON mapping")
    errors = sorted(_validator().iter_errors(data), key=lambda e: list(e.path))
    if errors:
        rendered = "; ".join(
            f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors[:10]
        )
        raise PolicyPackageValidationError(f"PolicyPackage failed schema validation: {rendered}")


def load_policy_package(source: str | bytes) -> PolicyPackageDocument:
    """Parse, validate, and return a typed PolicyPackage document.

    Raises :class:`PolicyPackageValidationError` on any structural failure.
    """
    data = _parse(source)
    _check_structural(data)
    try:
        return PolicyPackageDocument.model_validate(data)
    except ValidationError as exc:
        raise PolicyPackageValidationError(f"PolicyPackage failed model validation: {exc}") from exc


def validate_policy_package(document: Any) -> None:
    """Validate a PolicyPackage document without returning it.

    Accepts raw YAML/JSON text/bytes, a parsed mapping, or a
    :class:`~astro_mine.core.policy.model.PolicyPackageDocument`. Raises
    :class:`PolicyPackageValidationError` on failure.
    """
    if isinstance(document, PolicyPackageDocument):
        _check_structural(document.model_dump(mode="json"))
        return
    if isinstance(document, str | bytes):
        load_policy_package(document)
        return
    if isinstance(document, dict):
        _check_structural(document)
        try:
            PolicyPackageDocument.model_validate(document)
        except ValidationError as exc:
            raise PolicyPackageValidationError(
                f"PolicyPackage failed model validation: {exc}"
            ) from exc
        return
    raise PolicyPackageValidationError(f"cannot validate object of type {type(document).__name__}")
