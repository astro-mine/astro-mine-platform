# SPDX-License-Identifier: Apache-2.0
"""The published ``WorldSpec`` JSON Schema — shipped, not just derivable (worlds.md §11).

``worlds.md §11`` says the ``WorldSpec`` is "**JSON Schema** + Pydantic v2, versioned and owned by
Worlds". The Pydantic half has always been true; the JSON Schema half was a method nobody called —
:meth:`~astro_mine.worlds.spec.WorldSpec.json_schema` existed and nothing emitted or consumed it.
So a non-Python consumer — View ingesting a world bundle, a CI gate checking an authored document,
anyone outside this wheel — had no schema to check against, only a Python class they could not
import.

This module ships the generated document as package data and reads it back, so the schema is a
**file with a name** rather than a method call: fetchable, cacheable, and ``$ref``-able by its
absolute ``$id`` — the same discipline :mod:`astro_mine.worlds.spec._schema` already applies when
it names Core's units schema (RFC-0009 §1).

**Generated, and checked rather than maintained.** The model is the source of truth here — the
opposite of Core, whose hand-written schemas are authoritative and whose models mirror them — so
the shipped copy is regenerated from the model rather than edited:

.. code-block:: console

    astro-mine worlds schema > src/astro_mine/worlds/spec/schema/worldspec.schema.json

A test asserts the shipped file equals what the model generates today. That is the whole guard: a
model change that moves the schema fails there, with the regeneration command in the message, so
the published contract cannot drift silently behind the code it describes.

**No runtime dependency.** Emitting is ``json.dumps`` over a dict. ``jsonschema`` is a *dev*
dependency of this package — it validates the shipped example against the shipped schema in the
test suite — and must not leak into this path, or the base wheel gains a dependency it does not
declare.
"""

from __future__ import annotations

import json
from importlib import resources
from typing import Any

__all__ = ["SCHEMA_RESOURCE", "published_json_schema", "published_json_schema_text"]

#: The schema's path within this package's data.
SCHEMA_RESOURCE = "schema/worldspec.schema.json"

#: Anchored on ``spec`` because the schema directory holds data, not code, so it is not a package.
_ANCHOR = "astro_mine.worlds.spec"


def published_json_schema_text() -> str:
    """The shipped schema exactly as it is on disk — the bytes a consumer fetches.

    Read through :mod:`importlib.resources` rather than by path, so it resolves identically from a
    source checkout, an installed wheel, and a zipped one.
    """
    return resources.files(_ANCHOR).joinpath(SCHEMA_RESOURCE).read_text(encoding="utf-8")


def published_json_schema() -> dict[str, Any]:
    """The shipped schema, parsed.

    Prefer this over :meth:`WorldSpec.json_schema` when what you want is *the published contract*
    rather than *what this build would generate*. The test suite asserts they are equal, and the
    distinction is the point: one is the artifact, the other is its source.
    """
    parsed: dict[str, Any] = json.loads(published_json_schema_text())
    return parsed
