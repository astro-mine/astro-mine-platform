#!/usr/bin/env python
"""Regenerate the compiler golden and the schema-compat baseline (RM-P1-GUARD-01).

The compiled IR of the anchor SafetySpec is pinned byte-for-byte in
``tests/golden/anchor.compiled.json`` (the golden/determinism gate), and the SafetySpec /
CompiledSafetyModel schema shape is snapshotted in ``tests/schema_compat_baseline.json`` (the
additive-only guard). Both are **reviewed safety artifacts** — regenerate them only for an
intentional change and review the diff.

Usage: ``uv run python scripts/gen_golden.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from astro_mine.core.hashing import canonical_json
from astro_mine.guard.reference import anchor_safety_spec_text
from astro_mine.guard.spec import compile_spec, load_safety_spec
from astro_mine.guard.spec.loader import load_schema as load_spec_schema

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "golden" / "anchor.compiled.json"
BASELINE = ROOT / "tests" / "schema_compat_baseline.json"
COMPILED_SCHEMA = ROOT / "src/astro_mine/guard/spec/schema/compiled_safety_model.schema.json"


def _snapshot(schema: dict) -> dict:
    defs = schema.get("$defs", {})
    enums = {n: sorted(d["enum"]) for n, d in defs.items() if "enum" in d}
    required = {
        n: sorted(d.get("required", [])) for n, d in defs.items() if d.get("type") == "object"
    }
    required["<root>"] = sorted(schema.get("required", []))
    return {"enums": enums, "required": required}


def main() -> None:
    doc = load_safety_spec(anchor_safety_spec_text())
    compiled = compile_spec(doc)
    GOLDEN.parent.mkdir(parents=True, exist_ok=True)
    GOLDEN.write_bytes(canonical_json(compiled.model_dump(mode="json")))
    print(f"wrote {GOLDEN.relative_to(ROOT)} ({GOLDEN.stat().st_size} bytes)")

    baseline = {
        "safety_spec": _snapshot(load_spec_schema()),
        "compiled_safety_model": _snapshot(json.loads(COMPILED_SCHEMA.read_text(encoding="utf-8"))),
    }
    BASELINE.write_text(json.dumps(baseline, indent=2, sort_keys=True) + "\n")
    print(f"wrote {BASELINE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
