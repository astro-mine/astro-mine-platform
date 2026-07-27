"""JSON Schema drift guard + machine-consumability (RM-P1-SURR-01).

The checked-in schema files under ``src/astro_mine/surrogate/schema/`` must equal the
Pydantic models' exported JSON Schema (regenerate-and-compare drift guard), and the
ErrorReport schema must actually validate an ErrorReport instance — "machine-consumable,
not prose" (surrogate.md §6).
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from astro_mine.surrogate._schema import json_schemas
from tests.surrogate.factories import granular_report, illumination_report

_SCHEMA_DIR = (
    Path(__file__).resolve().parent.parent.parent / "src" / "astro_mine" / "surrogate" / "schema"
)


@pytest.mark.parametrize("filename", list(json_schemas()))
def test_checked_in_schema_matches_the_model(filename: str) -> None:
    on_disk = json.loads((_SCHEMA_DIR / filename).read_text())
    assert on_disk == json_schemas()[filename], (
        f"{filename} is stale — run `uv run python scripts/export_schemas.py`"
    )


def test_exported_schemas_carry_id_and_dialect() -> None:
    for schema in json_schemas().values():
        assert schema["$id"].startswith("https://schemas.astro-mine.org/surrogate/v0.1/")
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"


@pytest.mark.parametrize(
    "report", [granular_report(), illumination_report()], ids=["granular", "illumination"]
)
def test_error_report_schema_validates_an_instance(report) -> None:
    schema = json_schemas()["error_report.schema.json"]
    jsonschema.validate(instance=report.model_dump(mode="json"), schema=schema)


def test_error_report_schema_rejects_a_malformed_instance() -> None:
    schema = json_schemas()["error_report.schema.json"]
    bad = granular_report().model_dump(mode="json")
    del bad["trust_region"]  # a required field
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=schema)
