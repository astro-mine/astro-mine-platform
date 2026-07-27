"""JSON Schema drift guard + machine-consumability (RM-P1-ALLOC-01).

The checked-in schema files under ``src/astro_mine/allocate/schema/`` must equal the Pydantic
models' exported JSON Schema (regenerate-and-compare drift guard), and each schema must
actually validate an instance and reject a malformed one — "machine-consumable, not prose"
(conventions.md §3). This is the JSON-Schema half of the anchor round-trip acceptance
criterion (the Protobuf half is ``test_ir_wire_roundtrip``).
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from astro_mine.allocate import compile_request
from astro_mine.allocate._schema import json_schemas
from astro_mine.core import schema_registry
from tests.allocate.factories import anchor_request, solved

_SCHEMA_DIR = (
    Path(__file__).resolve().parent.parent.parent / "src" / "astro_mine" / "allocate" / "schema"
)


def _request_validator() -> jsonschema.Draft202012Validator:
    """A validator for the AllocationRequest schema with Core's cross-file ``$ref``\\s resolved.

    ``Task.location`` ``$ref``\\s Core's canonical ``Volume`` in ``messages.schema.json``, which
    in turn ``$ref``\\s ``units.schema.json`` for its ``ReferenceFrame`` (RFC-0007) — the deepest
    resolution chain in the org, three documents down.

    :func:`~astro_mine.core.schema_registry` carries **every** Core schema keyed by its ``$id``
    (RFC-0009), so the whole chain resolves from one call. This used to hand-extend Core's
    units-only registry with the messages catalog, because no public API could serve a consumer
    that needed more than units — exactly the gap RFC-0009 closed.
    """
    schema = json_schemas()["allocation_request.schema.json"]
    return jsonschema.Draft202012Validator(schema, registry=schema_registry(schema))


@pytest.mark.parametrize("filename", list(json_schemas()))
def test_checked_in_schema_matches_the_model(filename: str) -> None:
    on_disk = json.loads((_SCHEMA_DIR / filename).read_text())
    assert on_disk == json_schemas()[filename], (
        f"{filename} is stale — run `uv run python scripts/export_schemas.py`"
    )


def test_exported_schemas_carry_id_and_dialect() -> None:
    for schema in json_schemas().values():
        assert schema["$id"].startswith("https://schemas.astro-mine.org/allocate/v0.1/")
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"


def test_all_three_contracts_are_exported() -> None:
    assert set(json_schemas()) == {
        "allocation_ir.schema.json",
        "allocation_request.schema.json",
        "allocation.schema.json",
    }


def test_request_schema_validates_the_anchor_instance() -> None:
    # The request schema $refs Core's canonical Volume (→ ReferenceFrame → units) across files
    # (RFC-0007); _request_validator wires the referencing registry that resolves it offline.
    _request_validator().validate(anchor_request().model_dump(mode="json"))


def test_ir_schema_validates_the_compiled_anchor() -> None:
    schema = json_schemas()["allocation_ir.schema.json"]
    ir = compile_request(anchor_request())
    jsonschema.validate(instance=ir.model_dump(mode="json"), schema=schema)


def test_allocation_schema_validates_a_solved_plan() -> None:
    schema = json_schemas()["allocation.schema.json"]
    _, allocation = solved(anchor_request())
    jsonschema.validate(instance=allocation.model_dump(mode="json"), schema=schema)


def test_request_schema_rejects_a_malformed_instance() -> None:
    bad = anchor_request().model_dump(mode="json")
    del bad["tasks"]  # a required field
    with pytest.raises(jsonschema.ValidationError):
        _request_validator().validate(bad)
