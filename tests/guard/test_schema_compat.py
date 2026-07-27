"""Additive-only evolution guard for the SafetySpec (RM-P1-GUARD-01 acceptance).

The SafetySpec is a safety contract: schema changes must be **additive, RFC-gated**. This test
is the Python-level complement to ``buf breaking`` (which gates the wire form): against a
checked-in baseline snapshot it asserts (a) enum members are append-only — none removed or
renamed — and (b) no **new required field** is added to an existing object def. Adding a new
optional field or a whole new enum member / def is allowed; removing or re-typing is not.

Refresh the baseline (for an intentional, RFC-approved change) with ``scripts/gen_golden.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from astro_mine.guard.spec.loader import load_schema as load_spec_schema

BASELINE = Path(__file__).resolve().parent / "schema_compat_baseline.json"
COMPILED_SCHEMA = (
    Path(__file__).resolve().parents[2]
    / "src/astro_mine/guard/spec/schema/compiled_safety_model.schema.json"
)


def _live_schemas() -> dict[str, dict[str, Any]]:
    return {
        "safety_spec": load_spec_schema(),
        "compiled_safety_model": json.loads(COMPILED_SCHEMA.read_text(encoding="utf-8")),
    }


def _baseline() -> dict[str, Any]:
    return json.loads(BASELINE.read_text(encoding="utf-8"))


@pytest.mark.parametrize("schema_name", ["safety_spec", "compiled_safety_model"])
def test_enums_are_append_only(schema_name: str) -> None:
    live = _live_schemas()[schema_name]
    base_enums = _baseline()[schema_name]["enums"]
    live_defs = live.get("$defs", {})
    for name, base_values in base_enums.items():
        assert name in live_defs, f"enum {name!r} was removed (breaking)"
        live_values = set(live_defs[name]["enum"])
        removed = set(base_values) - live_values
        assert not removed, f"enum {name!r} removed/renamed members {sorted(removed)} (breaking)"


@pytest.mark.parametrize("schema_name", ["safety_spec", "compiled_safety_model"])
def test_no_new_required_fields(schema_name: str) -> None:
    live = _live_schemas()[schema_name]
    base_required = _baseline()[schema_name]["required"]
    live_defs = live.get("$defs", {})
    for name, base_fields in base_required.items():
        if name == "<root>":
            live_fields = set(live.get("required", []))
        else:
            assert name in live_defs, f"object def {name!r} was removed (breaking)"
            live_fields = set(live_defs[name].get("required", []))
        added = live_fields - set(base_fields)
        assert not added, f"{name!r} added new required field(s) {sorted(added)} (breaking)"
        removed = set(base_fields) - live_fields
        assert not removed, f"{name!r} dropped required field(s) {sorted(removed)} (breaking)"


def test_baseline_covers_current_enums() -> None:
    # Guard against a silently-forgotten baseline: every current enum is snapshotted.
    live = _live_schemas()
    base = _baseline()
    for schema_name, schema in live.items():
        live_enums = {n for n, d in schema.get("$defs", {}).items() if "enum" in d}
        assert live_enums <= set(base[schema_name]["enums"]) | live_enums  # sanity: no crash
        # a newly-added enum must be added to the baseline in the same change
        missing = live_enums - set(base[schema_name]["enums"])
        assert not missing, (
            f"{schema_name}: enum(s) {sorted(missing)} missing from baseline snapshot"
        )
