"""The content-addressed schema bundle is complete and reproducible (RM-P0-CORE-08).

``scripts/build_schema_bundle.py`` stages the canonical schemas into a bundle whose
``schema_digest`` is the identity a Bench run pins (``docs/VERSIONING.md`` §5). These
tests guard the three properties that make that pin trustworthy: the digest is
**reproducible** (same schemas → same digest, independent of version/commit/clients), the
bundle is **complete** (every declared schema is present and re-hashes correctly), and it
is **self-sufficient** — a consumer can deep-validate from the bundle alone, with a stock
JSON Schema validator and no Core-specific wiring (#53).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "build_schema_bundle.py"


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_build_schema_bundle", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Register before exec: dataclass field resolution looks the module up in sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def builder() -> ModuleType:
    return _load_builder()


def test_digest_is_reproducible(builder: ModuleType, tmp_path: Path) -> None:
    d1 = builder.build(tmp_path / "a")
    d2 = builder.build(tmp_path / "b")
    assert d1 == d2
    assert d1.startswith("sha256:")
    assert len(d1) == len("sha256:") + 64


def test_all_declared_schemas_present(builder: ModuleType, tmp_path: Path) -> None:
    builder.build(tmp_path / "bundle")
    out = tmp_path / "bundle"
    expected = {
        "schemas/json/sadf.schema.json",
        "schemas/json/objective.schema.json",
        "schemas/json/messages.schema.json",
        "schemas/json/manifest.schema.json",
        "schemas/json/run_provenance.schema.json",
        "schemas/json/policy_package.schema.json",
        "schemas/json/mission.schema.json",
        "schemas/json/units.schema.json",
        "schemas/json/plan.schema.json",
        "schemas/capnp/observation.capnp",
        "schemas/conformance/conformance.json",
    }
    for rel in expected:
        assert (out / rel).is_file(), f"missing {rel}"
    assert list(out.glob("schemas/proto/**/*.proto")), "no proto sources staged"


def test_no_schema_source_is_undeclared(builder: ModuleType) -> None:
    """Every schema source in the tree reaches the bundle — or says why it doesn't.

    The bundle's schema lists are hand-maintained, so a schema added to Core can miss the
    bundle in total silence: ``plan.schema.json`` (RFC-0006) did exactly that, and shipped
    a release cycle outside the published set before anyone noticed. This reconciles the
    lists against the tree so the next one can't."""
    # Anchor the guard first: a glob that silently matched nothing would leave the
    # assertion below vacuously true — green, while guarding exactly nothing.
    discovered = builder.discover_schema_sources()
    missing = builder.declared_schema_sources() - discovered
    assert not missing, f"declared schemas the discovery glob cannot see: {sorted(missing)}"

    undeclared = builder.undeclared_schema_sources()
    assert not undeclared, (
        f"schema sources not in the bundle: {sorted(undeclared)}\n"
        "Add each to JSON_SCHEMAS / CAPNP_SCHEMAS / CONFORMANCE_VECTORS in "
        "scripts/build_schema_bundle.py — or, to keep it out of the bundle on purpose, "
        "to BUNDLE_EXCLUDED with the reason."
    )


def test_bundle_metadata(builder: ModuleType, tmp_path: Path) -> None:
    import astro_mine.core as core
    from astro_mine.core.compat import CORE_INTERFACE_VERSIONS

    digest = builder.build(tmp_path / "bundle", commit="deadbeef")
    meta = json.loads((tmp_path / "bundle" / "bundle.json").read_text(encoding="utf-8"))

    assert meta["schema_digest"] == digest
    assert meta["core_version"] == core.__version__
    assert meta["source_commit"] == "deadbeef"
    assert meta["interface_versions"] == dict(CORE_INTERFACE_VERSIONS)
    # Every schema file is recorded with a hash; clients are empty without --clients-dir.
    assert {f["path"] for f in meta["files"]} >= {
        "schemas/json/sadf.schema.json",
        "schemas/json/manifest.schema.json",
    }
    assert meta["clients"] == []


def test_version_override_does_not_change_digest(builder: ModuleType, tmp_path: Path) -> None:
    base = builder.build(tmp_path / "base")
    pinned = builder.build(tmp_path / "pinned", version="9.9.9", commit="cafe")
    assert pinned == base  # digest is over schema content, not package version
    meta = json.loads((tmp_path / "pinned" / "bundle.json").read_text(encoding="utf-8"))
    assert meta["core_version"] == "9.9.9"


def test_manifest_re_hashes(builder: ModuleType, tmp_path: Path) -> None:
    builder.build(tmp_path / "bundle")
    out = tmp_path / "bundle"
    for line in (out / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        sha, rel = line.split("  ", 1)
        actual = hashlib.sha256((out / rel).read_bytes()).hexdigest()
        assert actual == sha, f"hash mismatch for {rel}"


def test_readme_documents_every_declared_schema(builder: ModuleType) -> None:
    """`schemas/json/README.md` names every schema the bundle declares (#52).

    The README's list is the third hand-maintained inventory of the schema set, after the
    bundle's lists and the model-drift check's — and it had rotted to three of nine. The
    declared lists are the source of truth; this reconciles the prose against them so it
    cannot silently fall behind again."""
    readme = (ROOT / "schemas" / "json" / "README.md").read_text(encoding="utf-8")
    undocumented = sorted(rel for rel in builder.declared_schema_sources() if rel not in readme)
    assert not undocumented, (
        f"schema sources not documented in schemas/json/README.md: {undocumented}\n"
        "Add each to the canonical-set table (path as written in build_schema_bundle.py)."
    )


def _maneuver(scale: str) -> dict[str, object]:
    """A Maneuver whose ``epoch`` is a units-typed ``Epoch`` — the cross-file ``$ref``."""
    return {
        "epoch_tdb_s": 0.0,
        "delta_v_mps": 1.0,
        "direction": {"x": 1.0, "y": 0.0, "z": 0.0},
        "maneuver_type": "impulsive",
        "epoch": {"tdb_seconds": 0.0, "scale": scale},
    }


def test_bundle_is_self_sufficient_for_a_stock_validator(
    builder: ModuleType, tmp_path: Path
) -> None:
    """A consumer can deep-validate from the bundle alone — no Core wiring (#53).

    This is the test whose absence let the bug ship. messages/mission ``$ref`` the units
    vocabulary across files by absolute ``$id`` (RFC-0007); those URIs are nominal, so a
    consumer needs the bundle's ``schema_index`` to map ``$id`` → file. Everything here is
    built from ``bundle.json`` and stock ``jsonschema``/``referencing`` — **nothing is
    imported from astro_mine**, which is exactly the position a non-Python binding is in.

    It also *descends into* a units-typed field. `$ref` resolution is lazy: a test that
    merely constructs the validator passes even when the ref is unresolvable.
    """
    from jsonschema import Draft202012Validator
    from referencing import Registry, Resource

    builder.build(tmp_path / "bundle")
    out = tmp_path / "bundle"
    meta = json.loads((out / "bundle.json").read_text(encoding="utf-8"))
    index = meta["schema_index"]

    # Register every bundled schema under its own $id — the generic move, no Core knowledge.
    registry: Registry = Registry().with_resources(
        [
            (schema_id, Resource.from_contents(json.loads((out / rel).read_text("utf-8"))))
            for schema_id, rel in index.items()
        ]
    )
    mission_id = "https://schemas.astro-mine.org/core/mission/v0.1/mission.schema.json"
    assert mission_id in index, "mission schema is not in the bundle index"
    validator = Draft202012Validator({"$ref": f"{mission_id}#/$defs/Maneuver"}, registry=registry)

    # Resolves *through* the cross-file units ref: "tdb" is a valid TimeScale, "utc" is not.
    # An unresolvable ref would raise here instead of yielding a clean validation verdict.
    assert not list(validator.iter_errors(_maneuver("tdb")))
    errors = list(validator.iter_errors(_maneuver("utc")))
    assert errors, "bad TimeScale accepted — the units $ref was never actually enforced"


def test_every_bundled_schema_is_in_the_index(builder: ModuleType, tmp_path: Path) -> None:
    """The index covers every bundled JSON Schema, and each entry points at a real file."""
    builder.build(tmp_path / "bundle")
    out = tmp_path / "bundle"
    index = json.loads((out / "bundle.json").read_text(encoding="utf-8"))["schema_index"]

    bundled = {p.relative_to(out).as_posix() for p in out.glob("schemas/json/*.schema.json")}
    assert set(index.values()) == bundled, "schema_index and bundled JSON Schemas disagree"
    for schema_id, rel in index.items():
        doc = json.loads((out / rel).read_text(encoding="utf-8"))
        assert doc["$id"] == schema_id, f"{rel} is indexed under an $id it does not declare"


def test_clients_excluded_from_schema_digest(builder: ModuleType, tmp_path: Path) -> None:
    # Stage a fake generated client tree and confirm it rides along in the bundle but
    # does not perturb the pinnable schema digest.
    clients = tmp_path / "codegen"
    fake = clients / "rust" / "generated"
    fake.mkdir(parents=True)
    (fake / "lib.rs").write_text("// generated\n", encoding="utf-8")

    with_clients = builder.build(tmp_path / "bundle", clients_dir=clients)
    without = builder.build(tmp_path / "plain")
    assert with_clients == without

    out = tmp_path / "bundle"
    assert (out / "clients" / "rust" / "lib.rs").is_file()
    meta = json.loads((out / "bundle.json").read_text(encoding="utf-8"))
    assert any(c["path"] == "clients/rust/lib.rs" for c in meta["clients"])
