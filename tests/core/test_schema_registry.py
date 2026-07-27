"""The public cross-package schema contract (RFC-0009).

Core owns schemas other packages ``$ref`` across files, and for a long time said nothing about
how. Six packages invented five techniques to name one schema; three of them reverse-engineered
a *private*, path-shaped URI, and astro-mine-core#54 retired it out from under them.

:func:`astro_mine.core.schema_registry` is the public answer: every Core schema resolvable by
its own ``$id``. These tests hold the two halves of the contract — that the public path works,
and that the deprecated one keeps working until every consumer has migrated off it.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from referencing.exceptions import NoSuchResource, Unresolvable

from astro_mine.core import schema_registry
from astro_mine.core.schemas import (
    CORE_JSON_SCHEMAS,
    core_schema,
    core_schema_documents,
)

UNITS_ID = "https://schemas.astro-mine.org/core/units/v0.1/units.schema.json"
MISSION_ID = "https://schemas.astro-mine.org/core/mission/v0.1/mission.schema.json"
RETIRED_UNITS_URI = "https://schemas.astro-mine.org/core/units/schema/units.schema.json"


def _maneuver(scale: str) -> dict[str, Any]:
    """A mission Maneuver whose ``epoch`` is a units-typed ``Epoch`` — the cross-file ``$ref``."""
    return {
        "epoch_tdb_s": 0.0,
        "delta_v_mps": 1.0,
        "direction": {"x": 1.0, "y": 0.0, "z": 0.0},
        "maneuver_type": "impulsive",
        "epoch": {"tdb_seconds": 0.0, "scale": scale},
    }


def test_every_core_schema_resolves_by_its_id() -> None:
    """The contract: a Core schema is named by its ``$id``, and the registry resolves it."""
    registry = schema_registry()
    for schema in core_schema_documents():
        resolved = registry.get_or_retrieve(str(schema["$id"])).value.contents
        assert resolved["$id"] == schema["$id"]


def test_a_consumer_resolves_a_cross_file_ref() -> None:
    """The whole point, end to end: `$ref` Core by `$id`, validate, descend into a Core `$def`.

    A consumer schema in its *own* namespace, referencing Core's units vocabulary absolutely —
    the shape `guard` / `prospect` / `studio` migrate to. It must descend: `$ref` resolution is
    lazy, so a test that merely builds the validator passes even when the ref is broken.
    """
    consumer = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://schemas.astro-mine.org/example/v0.1/consumer.schema.json",
        "$ref": f"{UNITS_ID}#/$defs/Epoch",
    }
    validator = Draft202012Validator(consumer, registry=schema_registry(consumer))
    assert not list(validator.iter_errors({"tdb_seconds": 0.0, "scale": "tdb"}))
    assert list(validator.iter_errors({"tdb_seconds": 0.0, "scale": "utc"})), (
        "a bad TimeScale was accepted — the cross-file $ref was never actually enforced"
    )


def test_registry_serves_a_multi_document_chain() -> None:
    """`allocate`'s case: request -> messages -> units, three documents deep.

    It used to hand-extend Core's units-only registry with `.with_resource(messages…)`. The
    public registry carries every Core schema, so the chain just resolves."""
    consumer = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://schemas.astro-mine.org/example/v0.1/request.schema.json",
        "$ref": "https://schemas.astro-mine.org/core/messages/v0.1/messages.schema.json#/$defs/Volume",
    }
    validator = Draft202012Validator(consumer, registry=schema_registry(consumer))
    # Volume carries a units-typed `frame_ref`, so this descends messages -> units.
    errors = list(validator.iter_errors({"frame_ref": {"frame": "not-a-frame-object"}}))
    assert errors, "the messages -> units hop was not enforced"


def test_core_schema_ids_are_not_shadowed_by_a_consumer() -> None:
    """Core owns its `$id`s; a consumer passing one cannot override Core's document.

    `worlds` really does pass Core's own units schema as its "consumer" (a workaround for the
    old units-only API), so this must be a silent no-op rather than an error."""
    units = core_schema("astro_mine.core.units", "units.schema.json")
    impostor = {"$id": UNITS_ID, "type": "string"}  # would break everything if it won

    registry = schema_registry(impostor, units)
    resolved = registry.get_or_retrieve(UNITS_ID).value.contents
    assert resolved.get("$defs"), "a consumer shadowed a Core $id"
    assert resolved["$id"] == units["$id"]


# --- the migration window is closed: the retired URI must NOT resolve ------------------------


def test_the_retired_uri_no_longer_resolves() -> None:
    """RFC-0009 §4 step 5: the legacy alias is gone, and the retired URI resolves to nothing.

    This is the inverse of the test that used to live here. During the migration window Core
    deliberately kept the path-shaped URI resolvable so `guard` / `prospect` / `studio` could
    move independently instead of in a flag day. All of them have (guard#26, prospect#37,
    studio#26), so the shim is deleted.

    It is asserted rather than merely deleted because *resolvable* is how a private convention
    masquerades as contract: the URI is one nobody declares and Core never promised, and every
    day it kept working was a day a new package could reverse-engineer it and re-enter the loop
    RFC-0009 closed. A `$ref` to it must now fail loudly.
    """
    consumer = {  # the shape prospect's schema had before it migrated
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://schemas.astro-mine.org/prospect/conditioning/manifest.crs.schema.json",
        "$ref": f"{RETIRED_UNITS_URI}#/$defs/PlanetaryCRS",
    }
    validator = Draft202012Validator(consumer, registry=schema_registry(consumer))
    with pytest.raises(Unresolvable):
        list(validator.iter_errors({"body": "not-a-crs"}))


def test_the_registry_carries_no_uri_that_is_not_an_id() -> None:
    """Every URI the registry resolves is some Core schema's own ``$id`` — and nothing else.

    The stronger, general form of the test above: rather than blacklisting the one URI we
    happen to have retired, assert the invariant that made it wrong. A Core schema is named by
    its ``$id`` (RFC-0009 §1) and a registry keyed by anything else is a private name in
    disguise — which is exactly what `_UNITS_REF_URI` was.
    """
    ids = {str(schema["$id"]) for schema in core_schema_documents()}
    registry = schema_registry()
    resolvable = {uri for uri in ids if registry.get_or_retrieve(uri).value.contents}
    assert resolvable == ids, "a Core $id does not resolve"
    # A *direct* registry lookup of an unregistered URI raises `NoSuchResource`; the same miss
    # surfaces as `Unresolvable` when it happens behind a `$ref` (the test above). The two are
    # unrelated types in `referencing`, so both paths are pinned rather than assumed equivalent.
    with pytest.raises(NoSuchResource):  # nothing outside the $id set resolves
        registry.get_or_retrieve(RETIRED_UNITS_URI)


def test_declaration_matches_the_shipped_schemas() -> None:
    """`CORE_JSON_SCHEMAS` is the single declaration — every entry must load and self-identify."""
    assert len(CORE_JSON_SCHEMAS) == len(core_schema_documents())
    for (pkg, name), doc in zip(CORE_JSON_SCHEMAS, core_schema_documents(), strict=True):
        assert doc is core_schema(pkg, name)
        assert str(doc["$id"]).startswith("https://schemas.astro-mine.org/"), (
            f"{name} declares an $id outside the astro-mine namespace"
        )
    assert json.dumps(sorted(s["$id"] for s in core_schema_documents()))  # all $ids unique-able
    ids = [s["$id"] for s in core_schema_documents()]
    assert len(ids) == len(set(ids)), "two Core schemas declare the same $id"
