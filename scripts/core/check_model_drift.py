#!/usr/bin/env python
"""Drift guard: the hand-written Pydantic models must not diverge from the canonical
JSON Schemas (RM-P0-CORE-07).

The JSON Schema is the source of truth; the Pydantic models in
``astro_mine.core.<comp>.model`` / ``.enums`` are hand-written (so they keep their
curated docstrings and exact semantics). This script regenerates each model from its
schema with ``datamodel-code-generator`` and fails if the *structure* drifts:

- every generated model class must exist in the hand-written module with the **same
  field-name set** (catches a schema field added/removed/renamed but not mirrored,
  and vice-versa);
- every generated enum must have the **same value set** as the hand-written enum.

Cosmetic generator-vs-hand differences (optionality/defaults, docstrings, base class,
enum member *names*) are deliberately ignored — only real structural drift fails.
Hand-written classes with no schema counterpart (the Cap'n Proto hot-path family in
``messages``) are allowed.

Usage: ``uv run python scripts/check_model_drift.py`` (run in CI). Exit 1 on drift.
"""

from __future__ import annotations

import enum
import importlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


_UNITS_SCHEMA = SRC / "astro_mine/core/units/schema/units.schema.json"

#: The URI messages/mission ``$ref`` the units vocabulary by — its own ``$id`` (RFC-0007).
_UNITS_REF_URI = json.loads(_UNITS_SCHEMA.read_text(encoding="utf-8"))["$id"]


@dataclass(frozen=True)
class Component:
    name: str
    schema: Path
    root_class: str | None  # --class-name for rooted schemas; None for a $defs catalog


COMPONENTS = [
    Component("sadf", SRC / "astro_mine/core/sadf/schema/sadf.schema.json", "SadfDocument"),
    Component(
        "objective",
        SRC / "astro_mine/core/objective/schema/objective.schema.json",
        "ObjectiveDocument",
    ),
    Component("messages", SRC / "astro_mine/core/messages/schema/messages.schema.json", None),
    Component(
        "registry",
        SRC / "astro_mine/core/registry/schema/manifest.schema.json",
        "ManifestDocument",
    ),
    Component(
        "provenance",
        SRC / "astro_mine/core/provenance/schema/run_provenance.schema.json",
        "RunProvenanceDocument",
    ),
    Component(
        "policy",
        SRC / "astro_mine/core/policy/schema/policy_package.schema.json",
        "PolicyPackageDocument",
    ),
    Component(
        "mission",
        SRC / "astro_mine/core/mission/schema/mission.schema.json",
        "MissionDocument",
    ),
    Component("units", SRC / "astro_mine/core/units/schema/units.schema.json", None),
    Component("plan", SRC / "astro_mine/core/plan/schema/plan.schema.json", "PlanDocument"),
]


def _stage_for_codegen(schema: Path, workdir: Path) -> Path:
    """Copy ``schema`` into ``workdir`` with its cross-file units ``$ref`` made local.

    messages/mission ``$ref`` the units vocabulary by its absolute ``$id`` (RFC-0007) —
    correct JSON Schema, and what lets a bundle consumer resolve it. datamodel-codegen,
    though, resolves a ``$ref`` as a *filesystem path*, so an ``https://`` ref would send
    it to the network (and it would fail, offline or not). Rewriting the URI to a sibling
    filename keeps codegen hermetic: same schema content, resolvable on disk, no
    ``--allow-remote-refs`` and therefore no way to silently fetch anything."""
    units_name = _UNITS_SCHEMA.name
    shutil.copyfile(_UNITS_SCHEMA, workdir / units_name)
    staged = workdir / schema.name
    text = schema.read_text(encoding="utf-8")
    if schema != _UNITS_SCHEMA:
        text = text.replace(_UNITS_REF_URI, units_name)
    staged.write_text(text, encoding="utf-8")
    return staged


def _generate(component: Component, out: Path) -> ModuleType:
    """Generate a throwaway model module from the schema and import it."""
    workdir = out.parent / f"_stage_{component.name}"
    workdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "datamodel-codegen",
        "--input",
        str(_stage_for_codegen(component.schema, workdir)),
        "--input-file-type",
        "jsonschema",
        "--output-model-type",
        "pydantic_v2.BaseModel",
        "--use-standard-collections",
        "--use-union-operator",
        "--target-python-version",
        "3.12",
        "--disable-timestamp",
        "--output",
        str(out),
    ]
    if component.root_class:
        cmd += ["--class-name", component.root_class]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    spec = importlib.util.spec_from_file_location(f"_drift_{component.name}", out)
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
    # Exclude the enum base classes themselves (imported into generated modules).
    bases = {enum.Enum, enum.IntEnum, enum.Flag, enum.IntFlag, enum.StrEnum}
    for name in dir(module):
        if keep is not None and name not in keep:
            continue
        obj = getattr(module, name)
        if isinstance(obj, type) and issubclass(obj, enum.Enum) and obj not in bases:
            out[name] = {str(m.value) for m in obj}
    return out


def check(component: Component, tmp: Path) -> list[str]:
    gen = _generate(component, tmp / f"{component.name}.py")
    hand_model = importlib.import_module(f"astro_mine.core.{component.name}.model")
    # A component whose schema declares no enums ships no ``enums`` module (``plan``).
    # Absent is not drift — but if the schema *does* declare an enum, the comparison
    # below still catches the missing module, and says so in schema terms.
    try:
        hand_enums = importlib.import_module(f"astro_mine.core.{component.name}.enums")
    except ModuleNotFoundError:
        hand_enums = ModuleType(f"astro_mine.core.{component.name}.enums")

    # Only compare symbols the schema actually declares (its $defs), plus the named
    # root. This drops dcg synthetic artifacts (a title-derived root wrapper / RootModel
    # for a $defs-only catalog) and imported base classes.
    defs = set(json.loads(component.schema.read_text(encoding="utf-8")).get("$defs", {}))
    keep = defs | ({component.root_class} if component.root_class else set())

    gen_models, hand_models = _models(gen, keep), _models(hand_model)
    gen_enums = _enums(gen, keep)
    hand_enum_set = _enums(hand_enums)

    problems: list[str] = []
    for cls, fields in gen_models.items():
        if cls not in hand_models:
            problems.append(
                f"class {cls!r} is in the schema but missing from {component.name}.model"
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
            problems.append(f"enum {en!r} is in the schema but missing from {component.name}.enums")
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
