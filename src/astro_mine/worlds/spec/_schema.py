# SPDX-License-Identifier: Apache-2.0
"""Emit-time validation of ``world.json``'s units objects against Core's schema (RM-P1-WORLDS-17).

``world.json`` carries a :class:`~astro_mine.core.units.PlanetaryCRS` (its ``crs`` field) and a
:class:`~astro_mine.core.units.ReferenceFrame` (its ``tiles_anchor.frame`` field) as plain
``model_dump(mode="json")`` mappings. Worlds reuses Core's Python models, but a non-Python consumer
(View, RM-P1-VIEW-06) only has the *serialized* form — so before it goes on disk the serialized form
is validated against Core's canonical ``units.schema.json`` (RM-P1-CORE-06, RFC-0007 Design §1a),
and the manifest publishes the schema ``$ref`` it conforms to. This closes the "unschema'd
``model_dump()``" gap RFC-0007 Motivation §5 names.

The validator names Core's units schema by its absolute ``$id`` — public, append-only API — and
resolves the cross-file ``$defs`` through Core's public :func:`astro_mine.core.schema_registry`
(RFC-0009 §1, §2) rather than re-deriving the schema, so Worlds and Core share one authority
(conventions.md §1 tenet 1: no private side-channels).
"""

from __future__ import annotations

from functools import cache
from typing import Any

from jsonschema import Draft202012Validator

from astro_mine.core import schema_registry
from astro_mine.core.schemas import core_schema

__all__ = [
    "PLANETARY_CRS_DEF",
    "REFERENCE_FRAME_DEF",
    "UNITS_SCHEMA_ID",
    "WorldSchemaError",
    "units_def_ref",
    "validate_units_object",
]

#: The ``$def`` names in ``units.schema.json`` the ``world.json`` manifest pins to.
PLANETARY_CRS_DEF = "PlanetaryCRS"
REFERENCE_FRAME_DEF = "ReferenceFrame"

#: The canonical units-schema ``$id`` (``.../core/units/v0.1/units.schema.json``), read from the
#: installed Core package through its public accessor so it tracks the pinned Core rev rather than
#: a hand-copied string. This value is written into every published ``world.json`` (``_bundle.py``),
#: so Core's ``$id`` is load-bearing in content-addressed output — which is why RFC-0009 §1 makes it
#: append-only API.
UNITS_SCHEMA_ID: str = str(core_schema("astro_mine.core.units", "units.schema.json")["$id"])


class WorldSchemaError(Exception):
    """Raised when a units object written into ``world.json`` fails schema validation."""


def units_def_ref(defname: str) -> str:
    """The canonical cross-file ``$ref`` URI for a units ``$def`` (e.g. ``PlanetaryCRS``)."""
    return f"{UNITS_SCHEMA_ID}#/$defs/{defname}"


@cache
def _validator(defname: str) -> Draft202012Validator:
    # A validator whose schema is just a ``$ref`` into the units vocabulary; the registry resolves
    # that cross-file reference (and the nested ``FrameClass`` ref) offline. ``schema_registry()``
    # carries every Core schema keyed by its own ``$id``, so nothing extra need be registered here
    # — the probe schema has no ``$id`` of its own and refs Core absolutely (RFC-0009 §2). Worlds
    # previously passed Core's units schema as its *own* consumer schema so that a units-only
    # registry would collapse both slots onto Core's ``$id``; the public API makes that unnecessary.
    return Draft202012Validator({"$ref": units_def_ref(defname)}, registry=schema_registry())


def validate_units_object(instance: Any, defname: str) -> None:
    """Validate a serialized units object against a ``units.schema.json`` ``$def``.

    ``instance`` is a ``model_dump(mode="json")`` mapping (a CRS or a reference frame); ``defname``
    is the ``$def`` it must conform to. Raises :class:`WorldSchemaError` — listing every schema
    error — when it does not, so a malformed CRS/frame is refused at emit rather than by a
    downstream consumer.
    """
    errors = sorted(_validator(defname).iter_errors(instance), key=lambda e: list(e.path))
    if errors:
        rendered = "; ".join(
            f"{'/'.join(str(p) for p in e.path) or '<root>'}: {e.message}" for e in errors
        )
        raise WorldSchemaError(
            f"{defname} object does not validate against {units_def_ref(defname)}: {rendered}"
        )
