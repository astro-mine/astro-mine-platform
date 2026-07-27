#!/usr/bin/env python
"""Drift guard: the hand-written Pydantic models must not diverge from the canonical
JSON Schemas (RM-P1-GUARD-01, the Core RM-P0-CORE-07 idiom).

The JSON Schema is the source of truth; the Pydantic models in
``astro_mine.guard.spec.model`` / ``.ir`` (and the shared ``.enums``) are hand-written so
they keep their curated docstrings and exact semantics. This script regenerates each model
from its schema with ``datamodel-code-generator`` and fails if the *structure* drifts:

- every generated model class must exist in the hand-written module with the **same
  field-name set** (catches a schema field added/removed/renamed but not mirrored, and
  vice-versa);
- every generated enum must have the **same value set** as the hand-written enum.

Cosmetic generator-vs-hand differences (optionality/defaults, docstrings, base class, enum
member *names*) are ignored — only real structural drift fails. Hand-written classes with no
schema counterpart (e.g. Core ``Vec3``/``Volume`` re-used by the SafetySpec model) are allowed.

Usage: ``uv run python scripts/check_model_drift.py`` (run in CI). Exit 1 on drift.
"""

from __future__ import annotations

import enum
import importlib
import importlib.util
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from types import ModuleType

from pydantic import BaseModel

from astro_mine.core.schemas import core_schema

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCHEMA_DIR = SRC / "astro_mine/guard/spec/schema"

#: Every Core schema a Guard schema ``$ref``s across files, as ``(package, filename)`` — the
#: ``frame_ref`` fields reach ``units`` (RFC-0007), and ``AdmissibleDirectives.tasks`` reaches
#: ``messages`` for Core's closed ``TaskKind`` vocabulary (RFC-0004 Amendment 2). Each is named by
#: its absolute ``$id`` (RFC-0009 §1), read from the *installed* Core through its public accessor so
#: it tracks the pinned Core rather than a hand-copied string that can go stale (which is exactly
#: how astro-mine-core#54 broke Guard).
_CORE_SCHEMAS: list[tuple[str, str]] = [
    ("astro_mine.core.units", "units.schema.json"),
    ("astro_mine.core.messages", "messages.schema.json"),
]

#: ``$id`` → the sibling filename the staged copies rewrite it to (see :func:`_staged_schema`).
#: ``messages.schema.json`` itself ``$ref``s ``units`` by ``$id``, so the staged Core copies are
#: rewritten with this same table — the rewrite has to be transitive or the generator cannot
#: resolve the second hop.
CORE_SCHEMA_IDS: dict[str, str] = {
    str(core_schema(pkg, name)["$id"]): name for pkg, name in _CORE_SCHEMAS
}


@dataclass(frozen=True)
class Component:
    name: str
    schema: Path
    model_module: str
    enums_module: str
    root_class: str


COMPONENTS = [
    Component(
        "safety_spec",
        SCHEMA_DIR / "safety_spec.schema.json",
        "astro_mine.guard.spec.model",
        "astro_mine.guard.spec.enums",
        "SafetyDocument",
    ),
    Component(
        "compiled_safety_model",
        SCHEMA_DIR / "compiled_safety_model.schema.json",
        "astro_mine.guard.spec.ir",
        "astro_mine.guard.spec.enums",
        "CompiledSafetyModel",
    ),
    # The SafetyVerdict audit record (RM-P1-GUARD-06) is a Guard-owned safety-contract message,
    # so its hand-written Pydantic model is drift-guarded against its canonical JSON Schema exactly
    # like the spec/IR. It carries no enums (closed vocabularies ride as plain strings), so the
    # enums module is the model module itself (an empty enum set).
    Component(
        "safety_verdict",
        SRC / "astro_mine/guard/audit/schema/safety_verdict.schema.json",
        "astro_mine.guard.audit.model",
        "astro_mine.guard.audit.model",
        "SafetyVerdict",
    ),
]


def _rewrite_core_ids(text: str) -> str:
    """Rewrite every Core ``$id`` in ``text`` to the sibling filename the staging dir holds."""
    for schema_id, filename in CORE_SCHEMA_IDS.items():
        text = text.replace(schema_id, filename)
    return text


def _staged_schema(component: Component, staging: Path) -> Path:
    """The schema path to feed datamodel-codegen, with every cross-file Core ``$ref`` resolvable.

    Guard's schemas ``$ref`` Core's ``units.schema.json`` (``frame_ref``, RFC-0007) and
    ``messages.schema.json`` (``AdmissibleDirectives.tasks`` → ``TaskKind``, RFC-0004
    Amendment 2) by their absolute ``$id`` (RFC-0009 §1) — the one public name for each.
    **datamodel-codegen resolves ``$ref``s from disk, not from a registry**, and nothing serves
    ``schemas.astro-mine.org`` (the URIs are nominal; resolution is always offline — RFC-0009 §1),
    so it cannot follow those ``$id``s.

    So the staged *copy* fed to the generator rewrites each ``$id`` to a sibling filename, and the
    installed Core schemas are dropped next to it — **including their own** ``$id`` refs, since
    ``messages.schema.json`` reaches ``units.schema.json`` the same way and the generator has to be
    able to follow that second hop too. This is a workaround for a code generator that only speaks
    file paths; the **shipped** schema keeps the ``$id``s, which is what the contract is about. The
    rewrite lives here rather than in the schema precisely so that Guard stops encoding Core's
    directory layout — the previous staging tree existed to make a
    ``../../../core/units/schema/...`` path-arithmetic ref land somewhere, which is the bug
    RFC-0009 removed (astro-mine-core#54 retired the URI it resolved to).

    Schemas with no Core ref (the audit verdict) are fed from their real location unchanged.
    """
    text = component.schema.read_text(encoding="utf-8")
    if not any(schema_id in text for schema_id in CORE_SCHEMA_IDS):
        return component.schema
    staging.mkdir(parents=True, exist_ok=True)
    for pkg, filename in _CORE_SCHEMAS:
        src = resources.files(pkg).joinpath(f"schema/{filename}").read_text(encoding="utf-8")
        (staging / filename).write_text(_rewrite_core_ids(src), encoding="utf-8")
    staged = staging / component.schema.name
    staged.write_text(_rewrite_core_ids(text), encoding="utf-8")
    return staged


def _generate(schema: Path, root_class: str, name: str, out: Path) -> ModuleType:
    """Generate a throwaway model module from the schema and import it."""
    cmd = [
        "datamodel-codegen",
        "--input",
        str(schema),
        "--input-file-type",
        "jsonschema",
        "--output-model-type",
        "pydantic_v2.BaseModel",
        "--use-standard-collections",
        "--use-union-operator",
        "--target-python-version",
        "3.12",
        "--disable-timestamp",
        "--class-name",
        root_class,
        "--output",
        str(out),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    spec = importlib.util.spec_from_file_location(f"_drift_{name}", out)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _models(module: ModuleType, keep: set[str] | None = None) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for name in dir(module):
        if keep is not None and name not in keep:
            continue
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, BaseModel) and obj is not BaseModel:
            out[name] = set(obj.model_fields)
    return out


def _enums(module: ModuleType, keep: set[str] | None = None) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    bases = {enum.Enum, enum.IntEnum, enum.Flag, enum.IntFlag, enum.StrEnum}
    for name in dir(module):
        if keep is not None and name not in keep:
            continue
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, enum.Enum) and obj not in bases:
            out[name] = {str(m.value) for m in obj}
    return out


def check(component: Component, tmp: Path) -> list[str]:
    staging = tmp / f"stage_{component.name}"
    schema_path = _staged_schema(component, staging)
    gen = _generate(schema_path, component.root_class, component.name, tmp / f"{component.name}.py")
    hand_model = importlib.import_module(component.model_module)
    hand_enums = importlib.import_module(component.enums_module)

    defs = set(json.loads(component.schema.read_text(encoding="utf-8")).get("$defs", {}))
    keep = defs | {component.root_class}

    gen_models, hand_models = _models(gen, keep), _models(hand_model)
    gen_enums = _enums(gen, keep)
    hand_enum_set = _enums(hand_enums)

    problems: list[str] = []
    for cls, fields in gen_models.items():
        if cls not in hand_models:
            problems.append(
                f"class {cls!r} is in the schema but missing from {component.model_module}"
            )
        elif fields != hand_models[cls]:
            missing = fields - hand_models[cls]
            extra = hand_models[cls] - fields
            detail = []
            if missing:
                detail.append(f"schema has fields not in model: {sorted(missing)}")
            if extra:
                detail.append(f"model has fields not in schema: {sorted(extra)}")
            problems.append(f"class {cls!r} field drift — " + "; ".join(detail))
    for en, values in gen_enums.items():
        if en not in hand_enum_set:
            problems.append(
                f"enum {en!r} is in the schema but missing from {component.enums_module}"
            )
        elif values != hand_enum_set[en]:
            problems.append(
                f"enum {en!r} value drift — schema {sorted(values)} "
                f"vs model {sorted(hand_enum_set[en])}"
            )
    return problems


def main() -> int:
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))
    all_problems: dict[str, list[str]] = {}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for component in COMPONENTS:
            problems = check(component, tmp)
            if problems:
                all_problems[component.name] = problems
    if all_problems:
        print("MODEL DRIFT DETECTED — hand-written models diverge from the JSON Schema:\n")
        for comp, problems in all_problems.items():
            print(f"[{comp}]")
            for p in problems:
                print(f"  - {p}")
        print("\nUpdate the model/enums to match the schema (or vice versa), then re-run.")
        return 1
    print("No model drift: hand-written models match the canonical JSON Schemas.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
