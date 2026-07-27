"""The canonical Core JSON Schemas, and offline resolution of their cross-file ``$ref``s.

**The contract other packages build against** (RFC-0009). A Core schema is named by its
absolute ``$id`` — that URI is public, append-only API — and :func:`schema_registry` resolves
it. A consuming package ``$ref``s Core by ``$id`` and validates like this::

    from jsonschema import Draft202012Validator
    from astro_mine.core import schema_registry

    validator = Draft202012Validator(my_schema, registry=schema_registry(my_schema))

That is the whole contract. There is nothing else to know, and nothing private to reach for.

Why this module exists: Core owned schemas that other packages needed to ``$ref``
(above all the units vocabulary, RFC-0007) but never said **how**. It solved only its own
need, with a private helper keyed on a URI that was not the schema's ``$id``. Six packages
then invented five different ways to name one schema — path arithmetic reconstructing Core's
directory layout, ``$id`` squatting inside Core's namespace, a hardcoded copy of the private
URI, runtime derivation, and a vendored byte-copy. Only derivation survived Core moving the
URI. This module is the public answer, so nobody has to guess a sixth time.

The ``$id`` URIs are **nominal**: nothing serves them, and resolution must work offline
(``conventions.md``). Resolution is therefore always by registry, never over the network.
"""

from __future__ import annotations

import json
from functools import cache, lru_cache
from importlib import resources
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # ``referencing`` is imported lazily — see :func:`schema_registry`.
    from referencing import Registry

__all__ = [
    "CORE_JSON_SCHEMAS",
    "core_schema",
    "core_schema_documents",
    "schema_registry",
]

#: Every canonical Core JSON Schema, as ``(package, filename)``. Schemas ship *inside* their
#: component package so they load in editable and wheel installs alike (``importlib.resources``).
#:
#: **The single declaration.** ``scripts/build_schema_bundle.py`` derives its bundle list from
#: this tuple rather than keeping a second copy, and ``tests/test_schema_bundle.py`` reconciles
#: it against the tree — a schema that reaches the repo without reaching this tuple fails CI.
#: Four hand-maintained inventories of this same set drifted apart before that rule existed
#: (astro-mine-core#50, #52); this is the one that remains.
CORE_JSON_SCHEMAS: tuple[tuple[str, str], ...] = (
    ("astro_mine.core.sadf", "sadf.schema.json"),
    ("astro_mine.core.objective", "objective.schema.json"),
    ("astro_mine.core.messages", "messages.schema.json"),
    ("astro_mine.core.registry", "manifest.schema.json"),
    ("astro_mine.core.provenance", "run_provenance.schema.json"),
    ("astro_mine.core.policy", "policy_package.schema.json"),
    ("astro_mine.core.mission", "mission.schema.json"),
    ("astro_mine.core.units", "units.schema.json"),
    ("astro_mine.core.plan", "plan.schema.json"),
)


@cache
def core_schema(package: str, filename: str) -> dict[str, Any]:
    """One canonical Core schema document, by ``(package, filename)`` (cached)."""
    text = resources.files(package).joinpath("schema", filename).read_text(encoding="utf-8")
    schema: dict[str, Any] = json.loads(text)
    return schema


@lru_cache(maxsize=1)
def core_schema_documents() -> tuple[dict[str, Any], ...]:
    """Every canonical Core schema document, in :data:`CORE_JSON_SCHEMAS` order (cached)."""
    return tuple(core_schema(pkg, name) for pkg, name in CORE_JSON_SCHEMAS)


def schema_registry(*extra: dict[str, Any]) -> Registry[Any]:
    """A ``referencing`` registry that resolves any Core schema ``$ref``, offline.

    Registers **every** Core schema under its own ``$id``, plus each schema in ``extra`` under
    its own — a consumer passes its schema(s) so that its *own* ``$ref``s resolve too::

        Draft202012Validator(my_schema, registry=schema_registry(my_schema))

    ``extra`` is variadic because a consumer may span several documents: ``allocate`` reaches
    Core's ``messages`` catalog *and*, through it, ``units`` — a chain it previously had to
    hand-assemble by extending a units-only registry.

    A Core schema resolves by its ``$id`` and by nothing else. The retired path-shaped units URI
    that three packages once reverse-engineered is **not** registered (RFC-0009 §4 step 5): the
    migration window is closed, and a URI Core never declared must not resolve — leaving it
    resolvable is what let the private convention masquerade as contract.
    """
    from referencing import Registry, Resource

    pairs: list[tuple[str, Resource[Any]]] = [
        (str(schema["$id"]), Resource.from_contents(schema)) for schema in core_schema_documents()
    ]
    # A consumer must never shadow a Core `$id`: Core owns those names (RFC-0009 §1), so its
    # document wins and a colliding `extra` is dropped rather than layered over.
    core_ids = {uri for uri, _ in pairs}
    pairs += [
        (str(schema["$id"]), Resource.from_contents(schema))
        for schema in extra
        if str(schema["$id"]) not in core_ids
    ]

    return Registry().with_resources(pairs)
