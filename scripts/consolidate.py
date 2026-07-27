#!/usr/bin/env python3
"""Consolidate the 18 astro-mine component repos into this single-package repo.

Mechanical by design: files are *copied* (never moved) from the sibling clones
under ../, byte-for-byte except for the explicit, reviewable transforms below
(each cross-referenced to docs/CONSOLIDATION_PLAN.md §6). Re-runnable.

Usage:
    python scripts/consolidate.py [--src-root ..] [--analyze-only]
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import shutil
import sys
import tomllib
from pathlib import Path

PLATFORM_ROOT = Path(__file__).resolve().parent.parent

COMPONENTS = [
    "core", "spice", "seal", "worlds", "prospect", "link", "fleet", "sim",
    "bench", "cloud", "surrogate", "mind", "learn", "allocate", "guard",
    "hub", "studio", "cli",
]

# --- §6.1: REST/HTTP-server code is NOT migrated -----------------------------
EXCLUDE_PACKAGE_PATHS: dict[str, list[str]] = {
    "hub": ["hub/api"],
    "studio": ["studio/api"],
    "cloud": ["cloud/serve"],
    # Only the FastAPI route module; the leaderboard *library* stays.
    "bench": ["bench/leaderboard/_app.py"],
}

EXCLUDE_TEST_PATHS: dict[str, list[str]] = {
    "bench": [
        "test_leaderboard.py", "test_leaderboard_hosted.py",
        "test_leaderboard_security.py", "test_submit.py",
    ],
    "hub": ["test_api.py", "test_admission.py", "test_artifact_kind.py"],
    "studio": [
        "test_api.py", "test_api_catalog.py", "test_api_hub.py",
        "test_seed_fixture.py", "test_ui_schema.py",
    ],
    "cloud": ["test_serve.py"],
}

# --- §2: repo-root items kept verbatim at the platform root ------------------
ROOT_PRESERVED: dict[str, list[str]] = {
    "core": [
        "schemas", "codegen", "validator", "buf.yaml", "buf.gen.yaml",
        "buf.gen.langs.yaml", "dist/schema-bundle",
    ],
    "worlds": ["validation"],
    "bench": ["embargo", "policy", "deploy", "TRUST_BOUNDARY.md", "Dockerfile"],
    "sim": ["docker"],
    "cloud": ["platform"],
    "allocate": ["benchmarks"],
    "hub": ["anchor-signing.pub"],
}

EXAMPLES_OWNERS = ["core", "guard"]
DOCS_OWNERS = ["guard", "hub"]
SCHEMA_OWNERS_NONCORE = ["prospect", "surrogate", "allocate", "guard"]

# Scripts read by tests: flat at scripts/ under original names (§6.9).
TEST_COUPLED_SCRIPTS: dict[str, list[str]] = {
    "core": ["build_schema_bundle.py"],
    "bench": ["determinism_gate.py"],
    "sim": ["render_dockerfile.py", "gen_proto.py", "gen_surrogate_fixture.py"],
}

# --- §6.3: repos whose tests use absolute `from tests.x import ...` ----------
TESTS_ABS_IMPORT = [
    "cloud", "mind", "learn", "allocate", "guard", "surrogate",
    "seal", "bench", "sim",
]

# --- §6.4: test files that resolved the repo root via Path(__file__) climbs.
# Tests moved one level deeper, so every parents[N] in these files gets N+1
# (and the three parent.parent climbers gain one .parent).
PARENTS_BUMP_FILES: dict[str, list[str]] = {
    "core": [
        "test_cli.py", "test_schema_bundle.py", "test_schema_digest.py",
        "test_validator_rust_parity.py", "test_sadf.py", "test_mission.py",
        "test_objective.py", "test_plan.py", "test_policy.py",
        "test_provenance.py", "test_registry.py", "test_umbrella.py",
        "test_sadf_consistency.py", "test_mission_consistency.py",
        "test_objective_consistency.py", "test_provenance_consistency.py",
        "test_registry_consistency.py",
    ],
    "worlds": ["test_gravity.py", "test_validation_psr.py"],
    "bench": [
        "test_zoo_anchor.py", "_factories.py", "test_eval.py",
        "test_sandbox.py", "test_telemetry.py",
    ],
    "cloud": ["test_platform_charts.py", "cluster/conftest.py"],
    "allocate": ["test_scale_benchmark.py"],
    "guard": ["test_schema_compat.py", "test_cli.py"],
    "learn": ["test_env_factory_seam.py"],
    "mind": ["test_cli.py"],
    "cli": ["test_new.py", "test_installed_provider.py"],
}
PARENT_PARENT_BUMP_FILES: dict[str, list[str]] = {
    "allocate": ["test_schema.py"],
    "surrogate": ["test_schema.py"],
    "sim": ["test_bench_runner.py", "test_packaging.py", "test_service.py"],
    "bench": ["_policies_hostile.py", "test_determinism_gate.py",
              "test_fidelity_crossover.py"],
}

# --- §6.2: dist-name version lookups -> the platform distribution ------------
_COMP_ALT = "|".join(COMPONENTS)
VERSION_CALL_RE = re.compile(
    rf"version\(\s*([\"'])astro-mine-(?:{_COMP_ALT})\1\s*\)"
)

ALWAYS_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo", ".pytest_cache", ".ruff_cache",
    ".hypothesis", "*.egg-info", ".DS_Store", ".venv", "node_modules",
    "target",  # Rust build outputs (rust/, validator/rust/)
)


def repo_dir(src_root: Path, comp: str) -> Path:
    return src_root / f"astro-mine-{comp}"


def _copy_tree(src: Path, dst: Path, excludes: list[str], base: Path,
               skipped: list[str]) -> None:
    def ignore(dirpath: str, names: list[str]) -> set[str]:
        out: set[str] = set()
        rel_dir = Path(dirpath).relative_to(base)
        for name in names:
            rel = (rel_dir / name).as_posix()
            for pat in excludes:
                if rel == pat or fnmatch.fnmatch(rel, pat):
                    out.add(name)
                    skipped.append(rel)
        return out | ALWAYS_IGNORE(dirpath, names)

    shutil.copytree(src, dst, ignore=ignore, dirs_exist_ok=True)


def _fresh(dst: Path) -> Path:
    if dst.is_dir():
        shutil.rmtree(dst)
    elif dst.exists():
        dst.unlink()
    return dst


def copy_component(src_root: Path, comp: str, report: dict) -> None:
    repo = repo_dir(src_root, comp)
    if not repo.is_dir():
        sys.exit(f"missing source repo: {repo}")
    skipped: list[str] = []

    _copy_tree(repo / "src" / "astro_mine" / comp,
               _fresh(PLATFORM_ROOT / "src" / "astro_mine" / comp),
               EXCLUDE_PACKAGE_PATHS.get(comp, []),
               base=repo / "src" / "astro_mine", skipped=skipped)

    if (repo / "tests").is_dir():
        _copy_tree(repo / "tests", _fresh(PLATFORM_ROOT / "tests" / comp),
                   EXCLUDE_TEST_PATHS.get(comp, []),
                   base=repo / "tests", skipped=skipped)

    for item in ROOT_PRESERVED.get(comp, []):
        src = repo / item
        if not src.exists():
            sys.exit(f"expected {src}")
        dst = _fresh(PLATFORM_ROOT / item)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            _copy_tree(src, dst, [], base=src, skipped=skipped)
        else:
            shutil.copy2(src, dst)

    if comp in SCHEMA_OWNERS_NONCORE and (repo / "schemas").is_dir():
        _copy_tree(repo / "schemas", _fresh(PLATFORM_ROOT / "schemas" / comp),
                   [], base=repo / "schemas", skipped=skipped)
        for buf in ("buf.yaml", "buf.gen.yaml"):
            if (repo / buf).exists():
                shutil.copy2(repo / buf, PLATFORM_ROOT / "schemas" / comp / buf)

    if comp in EXAMPLES_OWNERS and (repo / "examples").is_dir():
        _copy_tree(repo / "examples", PLATFORM_ROOT / "examples", [],
                   base=repo / "examples", skipped=skipped)

    if comp in DOCS_OWNERS and (repo / "docs").is_dir():
        _copy_tree(repo / "docs", _fresh(PLATFORM_ROOT / "docs" / comp), [],
                   base=repo / "docs", skipped=skipped)

    # Per-repo README/ARCHITECTURE, kept for reference (and for the few tests
    # that assert against their own repo's README).
    comp_docs = PLATFORM_ROOT / "docs" / "components" / comp
    comp_docs.mkdir(parents=True, exist_ok=True)
    for f in ("README.md", "ARCHITECTURE.md"):
        if (repo / f).exists():
            shutil.copy2(repo / f, comp_docs / f)

    if (repo / "scripts").is_dir():
        flat = set(TEST_COUPLED_SCRIPTS.get(comp, []))
        for f in sorted((repo / "scripts").iterdir()):
            if f.name in ("__pycache__",):
                continue
            dst = (PLATFORM_ROOT / "scripts" / f.name if f.name in flat
                   else PLATFORM_ROOT / "scripts" / comp / f.name)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if f.is_dir():
                _copy_tree(f, _fresh(dst), [], base=f, skipped=skipped)
            else:
                shutil.copy2(f, dst)

    report[comp] = {"excluded": skipped}


def copy_guard_rust(src_root: Path) -> None:
    src = repo_dir(src_root, "guard") / "rust"
    shutil.copytree(src, _fresh(PLATFORM_ROOT / "rust"),
                    ignore=shutil.ignore_patterns("target", "__pycache__"))


def rewrite_version_calls() -> list[str]:
    """§6.2: version("astro-mine-<comp>") -> version("astro-mine-platform")."""
    changed: list[str] = []
    for py in sorted((PLATFORM_ROOT / "src" / "astro_mine").rglob("*.py")):
        text = py.read_text()
        new, n = VERSION_CALL_RE.subn('version("astro-mine-platform")', text)
        if n:
            py.write_text(new)
            changed.append(f"{py.relative_to(PLATFORM_ROOT)} ({n})")
    return changed


def rewrite_tests_imports() -> list[str]:
    """§6.3: `from tests.x` -> `from tests.<comp>.x` in the six absolute-import repos."""
    changed: list[str] = []
    pat = re.compile(r"^(\s*)(from|import)\s+tests\.", re.MULTILINE)
    for comp in TESTS_ABS_IMPORT:
        for py in sorted((PLATFORM_ROOT / "tests" / comp).rglob("*.py")):
            text = py.read_text()
            new, n = pat.subn(rf"\1\2 tests.{comp}.", text)
            if n:
                py.write_text(new)
                changed.append(f"{py.relative_to(PLATFORM_ROOT)} ({n})")
    return changed


def bump_parents() -> list[str]:
    """§6.4: tests moved one dir deeper -> parents[N] climbs need N+1."""
    changed: list[str] = []
    pat = re.compile(r"parents\[(\d+)\]")
    for comp, files in PARENTS_BUMP_FILES.items():
        for rel in files:
            path = PLATFORM_ROOT / "tests" / comp / rel
            text = path.read_text()
            new, n = pat.subn(lambda m: f"parents[{int(m.group(1)) + 1}]", text)
            if n:
                path.write_text(new)
                changed.append(f"{path.relative_to(PLATFORM_ROOT)} ({n})")
    for comp, files in PARENT_PARENT_BUMP_FILES.items():
        for rel in files:
            path = PLATFORM_ROOT / "tests" / comp / rel
            text = path.read_text()
            new, n = re.subn(r"\.parent\.parent\b(?!\.parent)",
                             ".parent.parent.parent", text)
            if n:
                path.write_text(new)
                changed.append(f"{path.relative_to(PLATFORM_ROOT)} ({n})")
    return changed


def make_test_packages() -> None:
    """§6.3: tests/ becomes a package with per-component subpackages."""
    top = PLATFORM_ROOT / "tests" / "__init__.py"
    if not top.exists():
        top.write_text("")
    for comp in COMPONENTS:
        d = PLATFORM_ROOT / "tests" / comp
        if d.is_dir() and not (d / "__init__.py").exists():
            (d / "__init__.py").write_text("")


def analyze_deps(src_root: Path) -> dict:
    out: dict = {"deps": {}, "scripts": {}, "entry_points": {}, "conflicts": []}
    for comp in COMPONENTS:
        pp = repo_dir(src_root, comp) / "pyproject.toml"
        proj = tomllib.loads(pp.read_text())["project"]
        for dep in proj.get("dependencies", []):
            name = re.split(r"[><=!\[;]", dep)[0].strip().lower()
            if not name.startswith("astro-mine"):
                out["deps"].setdefault(name, {})[comp] = dep
        for name, target in proj.get("scripts", {}).items():
            if out["scripts"].setdefault(name, target) != target:
                out["conflicts"].append(f"script {name}")
        for group, eps in proj.get("entry-points", {}).items():
            for name, target in eps.items():
                key = f"{group}:{name}"
                if out["entry_points"].setdefault(key, target) != target:
                    out["conflicts"].append(f"entry point {key}")
    for name, users in out["deps"].items():
        if len(set(users.values())) > 1:
            out["conflicts"].append(f"dep {name}: {users}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-root", default=str(PLATFORM_ROOT.parent))
    ap.add_argument("--analyze-only", action="store_true")
    args = ap.parse_args()
    src_root = Path(args.src_root).resolve()

    report: dict = {"components": {}}
    if not args.analyze_only:
        for comp in COMPONENTS:
            copy_component(src_root, comp, report["components"])
            print(f"copied {comp}", file=sys.stderr)
        copy_guard_rust(src_root)
        make_test_packages()
        report["version_call_rewrites"] = rewrite_version_calls()
        report["tests_import_rewrites"] = rewrite_tests_imports()
        report["parents_bumps"] = bump_parents()
    report["analysis"] = analyze_deps(src_root)
    out = PLATFORM_ROOT / "build" / "consolidation-report.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True))
    print(f"report: {out}", file=sys.stderr)
    for c in report["analysis"]["conflicts"]:
        print(f"CONFLICT: {c}", file=sys.stderr)


if __name__ == "__main__":
    main()
