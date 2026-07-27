"""Pin the content-addressed ``PlanetaryCRS`` to Core's canonical units schema (RM-P1-STUDIO-08).

Studio content-addresses artifacts that embed Core's ``PlanetaryCRS`` — ``GeoRegion.crs`` on
an ``IntentDraft``, carried unchanged into a ``Campaign`` (studio.md §3, §9). The content hash
is the reproducibility identity (CX-REPRO), and the CRS it hashes is otherwise an unschematized
dict: if Core's ``units.schema.json`` ever disagrees with what ``model_dump`` emits, every
Studio content hash silently means something different (RFC-0007 §1a, §3; conventions.md §5).

This module closes that gap without re-deriving the contract (conventions.md §1 tenet 1):

* :func:`validate_crs_schema` validates the CRS dict against Core's canonical
  ``units.schema.json`` ``PlanetaryCRS`` ``$def`` at the point it *enters* a content-addressed
  artifact — naming Core's schema by its absolute ``$id`` and resolving the cross-file ``$ref``
  offline through the public :func:`astro_mine.core.schema_registry` (RFC-0009 §1, §2), so Studio
  validates against Core's schema, not a private copy.
* :func:`core_units_schema_digest` is the schema's own content digest, computed with Core's
  content-hash helper over the canonical JSON of the loaded schema. It is recorded *alongside*
  a campaign's content hash (never inside the hashed payload) so a rehydrated campaign can prove
  which vocabulary version it was authored against.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from jsonschema import Draft202012Validator

from astro_mine.core import schema_registry
from astro_mine.core.hashing import canonical_json, content_hash
from astro_mine.core.schemas import core_schema


@lru_cache(maxsize=1)
def _units_schema() -> dict[str, Any]:
    """Core's canonical ``units.schema.json``, via Core's public accessor (cached)."""
    return core_schema("astro_mine.core.units", "units.schema.json")


# A minimal consumer schema that names Core's units vocabulary by its absolute ``$id`` — public,
# append-only API — which :func:`astro_mine.core.schema_registry` resolves offline (RFC-0009 §1,
# §2). The probe's own ``$id`` sits under **Studio's** namespace: a package declares ``$id``s only
# under its own, and two packages must never publish the same one (RFC-0009 §1). The ``$id`` is
# derived from the installed Core rather than hand-copied, so it tracks the pinned Core.
_CRS_PROBE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://schemas.astro-mine.org/studio/v0.1/crs-probe.schema.json",
    "$ref": f"{_units_schema()['$id']}#/$defs/PlanetaryCRS",
}


class CrsSchemaError(ValueError):
    """A ``PlanetaryCRS`` entering a content-addressed artifact fails Core's units schema."""


@lru_cache(maxsize=1)
def _crs_validator() -> Draft202012Validator:
    return Draft202012Validator(_CRS_PROBE_SCHEMA, registry=schema_registry(_CRS_PROBE_SCHEMA))


def validate_crs_schema(crs: Any) -> dict[str, Any]:
    """Validate a ``PlanetaryCRS`` (a model or its JSON dict) against Core's units schema.

    Returns the validated JSON dict — the exact shape that gets content-addressed. Raises
    :class:`CrsSchemaError`, listing every schema violation, if the CRS does not conform: the
    fail-loud guard that keeps the reproducibility chain pinned to Core's vocabulary
    (RM-P1-STUDIO-08). This is a structural check against ``units.schema.json``; the semantic
    rules (rule 6: no Earth marker on a non-Earth body) are enforced by Core's ``require_crs``
    guard at the intent boundary.
    """
    payload = crs.model_dump(mode="json") if hasattr(crs, "model_dump") else dict(crs)
    errors = sorted(_crs_validator().iter_errors(payload), key=lambda err: err.json_path)
    if errors:
        detail = "; ".join(f"{err.json_path}: {err.message}" for err in errors)
        raise CrsSchemaError(f"PlanetaryCRS does not conform to Core units.schema.json ({detail})")
    return payload


@lru_cache(maxsize=1)
def core_units_schema_digest() -> str:
    """Content digest (``sha256:<hex>``) of Core's canonical ``units.schema.json``.

    The units-vocabulary version a content-addressed CRS was authored against. Computed with
    Core's own content-hash helper over the canonical JSON of the *loaded* schema, so it is
    stable regardless of on-disk formatting and matches a digest computed anywhere else on the
    platform (conventions.md §5; RFC-0007 §3).
    """
    return content_hash(canonical_json(_units_schema()))
