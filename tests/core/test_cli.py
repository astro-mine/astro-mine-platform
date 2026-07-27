"""``astro-mine-core validate`` — the CLI over the Core-authored formats (G2.5, astro-mine-core#61).

The tests pin the properties the issue is about:

* dispatch is **derived from the schema registry**, not a hand-maintained ``{kind: filename}`` map
  (the #50/#52/#53 drift), and adding a tenth schema needs no change to the CLI;
* it validates every authored document format, JSON and YAML;
* it never validates against a guessed schema — an ambiguous/unknown document fails with the list
  of known kinds;
* cross-file ``$ref``s (``mission`` → ``units``) resolve **offline, from a built wheel** — the
  #55 failure mode, where a fifth of the hashed inputs lived outside ``src/``.
"""

from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from astro_mine.core import cli
from astro_mine.core.schemas import CORE_JSON_SCHEMAS, core_schema

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"

# (kind slug, example file) — one authored document per format the CLI validates.
DOCUMENT_EXAMPLES = [
    ("objective", EXAMPLES / "objectives" / "lunar-polar-ice-prospecting.objective.yaml"),
    ("mission", EXAMPLES / "mission" / "lunar-surface-single-phase.mission.yaml"),
    ("mission", EXAMPLES / "mission" / "neo-sample-return-multiphase.mission.yaml"),
    ("plan", EXAMPLES / "plan" / "lunar-prospecting-contingent.plan.yaml"),
    ("plan", EXAMPLES / "plan" / "standing-control.plan.yaml"),
    ("manifest", EXAMPLES / "plugins" / "greedy-prospecting-baseline.manifest.yaml"),
    ("policy_package", EXAMPLES / "policy" / "minimal.policy-package.yaml"),
    ("run_provenance", EXAMPLES / "run-provenance" / "minimal.run-provenance.yaml"),
    ("sadf", EXAMPLES / "assets" / "lunar-scout-rover.sadf.yaml"),
]


# --------------------------------------------------------------------------- dispatch derivation


def test_kinds_are_derived_from_the_registry() -> None:
    """Every kind's schema is a Core registry schema resolved by its own ``$id`` — nothing else.

    This is the contract: the CLI does not keep its own inventory of the schema set. A kind exists
    because a Core schema (and, for documents, its loader) exists.
    """
    expected_ids = {str(core_schema(pkg, name)["$id"]) for pkg, name in CORE_JSON_SCHEMAS}
    documents = {id(core_schema(pkg, name)) for pkg, name in CORE_JSON_SCHEMAS}
    for kind in cli.iter_kinds():
        assert kind.schema_id in expected_ids
        # The kind carries the *registry's own* schema object, not a private copy.
        assert id(kind.schema) in documents


def test_no_kind_to_filename_map_in_the_source() -> None:
    """No schema *filename* is hard-coded in the CLI — that is the fourth-inventory drift trap.

    The kind list must come from the registry; a ``{"objective": "objective.schema.json"}`` dict
    would be the #50/#52/#53 mistake with a new home.
    """
    source = (ROOT / "src" / "astro_mine" / "core" / "cli.py").read_text(encoding="utf-8")
    for _pkg, filename in CORE_JSON_SCHEMAS:
        assert filename not in source, f"{filename} is hard-coded in cli.py — derive it instead"


def test_every_document_schema_becomes_a_kind() -> None:
    """A tenth schema needs no CLI edit: coverage is exactly instance-constraining-or-has-loader.

    The two ``$defs``-only vocabularies with no document loader (``units``) are deliberately absent
    — validating a file against a vocabulary would pass anything, a false certification.
    """
    slugs = {k.slug for k in cli.iter_kinds()}
    assert {
        "sadf",
        "objective",
        "mission",
        "plan",
        "manifest",
        "policy_package",
        "run_provenance",
    } <= slugs
    assert {"action_batch", "contact_plan"} <= slugs  # the message catalog's authored documents
    assert "units" not in slugs  # a referenced vocabulary, not a standalone document format


# --------------------------------------------------------------------------- validation


@pytest.mark.parametrize(
    ("kind", "path"), DOCUMENT_EXAMPLES, ids=lambda v: v.name if isinstance(v, Path) else v
)
def test_examples_validate(kind: str, path: Path) -> None:
    resolved, issues = cli.validate_source(path.read_text(encoding="utf-8"), kind)
    assert resolved.slug == kind
    assert issues == [], f"{path.name}: {[i.render() for i in issues]}"


def test_yaml_and_json_are_equivalent() -> None:
    import yaml

    path = EXAMPLES / "objectives" / "lunar-polar-ice-prospecting.objective.yaml"
    as_yaml = path.read_text(encoding="utf-8")
    as_json = json.dumps(yaml.safe_load(as_yaml))
    assert cli.validate_source(as_yaml, "objective")[1] == []
    assert cli.validate_source(as_json, "objective")[1] == []


def test_self_describing_document_dispatches_on_its_schema() -> None:
    """A document may carry ``$schema``; it is dispatch metadata and does not fail the schema."""
    doc = {
        "$schema": "https://schemas.astro-mine.org/core/objective/v0.1/objective.schema.json",
        "objective_version": "0.1",
        "objective": {
            "id": "o",
            "name": "O",
            "success_criteria": [
                {
                    "id": "c1",
                    "binding": {
                        "metric": "m",
                        "unit": "kg",
                        "direction": "higher_better",
                        "target": 1.0,
                        "tolerance": 0.1,
                    },
                }
            ],
        },
    }
    kind, issues = cli.validate_source(json.dumps(doc), None)
    assert kind.slug == "objective"
    assert issues == []


# --------------------------------------------------------------------------- honest failure


def test_unknown_kind_lists_the_known_kinds() -> None:
    with pytest.raises(cli.KindError) as excinfo:
        cli.resolve_kind({"a": 1}, "units")
    assert "known kinds:" in str(excinfo.value)
    assert "objective" in str(excinfo.value)


def test_undetermined_kind_is_refused_not_guessed() -> None:
    with pytest.raises(cli.KindError) as excinfo:
        cli.resolve_kind({"objective_version": "0.1"}, None)  # looks like an objective; not guessed
    assert "--kind" in str(excinfo.value)


def test_ambiguous_catalog_schema_requires_a_kind() -> None:
    doc = {"$schema": "https://schemas.astro-mine.org/core/messages/v0.1/messages.schema.json"}
    with pytest.raises(cli.KindError) as excinfo:
        cli.resolve_kind(doc, None)
    assert "action_batch" in str(excinfo.value) and "contact_plan" in str(excinfo.value)


def test_schema_error_names_pointer_value_and_expectation() -> None:
    doc = {
        "objective_version": "0.1",
        "objective": {
            "id": "o",
            "name": "O",
            "success_criteria": [
                {
                    "id": "c1",
                    "binding": {
                        "metric": "m",
                        "unit": "kg",
                        "direction": "sideways",  # not in the enum
                        "target": 1.0,
                        "tolerance": 0.1,
                    },
                }
            ],
        },
    }
    _kind, issues = cli.validate_source(json.dumps(doc), "objective")
    assert len(issues) == 1
    issue = issues[0]
    assert issue.layer == "schema"
    assert issue.pointer == "/objective/success_criteria/0/binding/direction"
    assert issue.expected.startswith("enum")
    assert "sideways" in issue.render()


def test_model_layer_catches_what_schema_cannot() -> None:
    """A schema-valid but semantically-invalid document is rejected, and the layer is named."""
    doc = {
        "objective_version": "0.1",
        "objective": {
            "id": "o",
            "name": "O",
            "success_criteria": [
                {
                    "id": "dup",
                    "binding": {
                        "metric": "m",
                        "unit": "kg",
                        "direction": "higher_better",
                        "target": 1.0,
                        "tolerance": 0.1,
                    },
                },
                {
                    "id": "dup",  # duplicate id — a semantic rule, not expressible in JSON Schema
                    "binding": {
                        "metric": "n",
                        "unit": "kg",
                        "direction": "higher_better",
                        "target": 2.0,
                        "tolerance": 0.1,
                    },
                },
            ],
        },
    }
    _kind, issues = cli.validate_source(json.dumps(doc), "objective")
    assert issues and issues[0].layer == "model"
    assert "duplicate" in issues[0].message


# --------------------------------------------------------------------------- CLI surface


def test_validate_exit_code_and_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = tmp_path / "good.json"
    good.write_text(
        (EXAMPLES / "objectives" / "lunar-polar-ice-prospecting.objective.yaml").read_text(),
        encoding="utf-8",
    )
    bad = tmp_path / "bad.yaml"
    bad.write_text("objective_version: '0.1'\nobjective: {}\n", encoding="utf-8")

    # One good, one bad → non-zero exit because *any* file failed.
    code = cli.main(["--json", "validate", "--kind", "objective", str(good), str(bad)])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    by_valid = {row["file"].endswith("good.json"): row["valid"] for row in payload}
    assert by_valid[True] is True and by_valid[False] is False

    assert cli.main(["validate", "--kind", "objective", str(good)]) == 0


def test_kinds_command_lists_from_registry(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["--json", "kinds"]) == 0
    rows = json.loads(capsys.readouterr().out)
    listed = {row["kind"]: row["schema_id"] for row in rows}
    assert listed == {k.slug: k.schema_id for k in cli.iter_kinds()}


def test_cross_file_ref_resolves_offline() -> None:
    """A ``mission`` document ``$ref``s the units vocabulary; it must validate with no network."""
    path = EXAMPLES / "mission" / "neo-sample-return-multiphase.mission.yaml"
    _kind, issues = cli.validate_source(path.read_text(encoding="utf-8"), "mission")
    assert issues == []


# --------------------------------------------------------------------------- the wheel boundary


def test_wheel_carries_the_cli_and_its_schemas(tmp_path: Path) -> None:
    """The CLI + every schema resource survive into a built wheel, and cross-refs resolve from it.

    Consumers install a **wheel**, not this source tree. #55 shipped a plausible-but-wrong digest
    because a fifth of the hashed inputs was absent from the wheel; a CLI that resolved schemas by
    a path relative to ``__file__`` would work here and break for every real user. This builds a
    real wheel, asserts the code, the console-script entry point, and all nine schemas are in it,
    and runs a cross-ref (mission → units) validation loading the code **from the wheel** offline.
    """
    try:
        subprocess.run(
            ["uv", "build", "--wheel", "--out-dir", str(tmp_path)],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - env-dependent
        pytest.skip(f"could not build a wheel: {exc}")

    wheels = list(tmp_path.glob("*.whl"))
    assert len(wheels) == 1, f"expected exactly one wheel, got {wheels}"

    extracted = tmp_path / "site"
    with zipfile.ZipFile(wheels[0]) as whl:
        names = set(whl.namelist())
        assert "astro_mine/core/cli.py" in names, "the CLI module is missing from the wheel"
        for _pkg, filename in CORE_JSON_SCHEMAS:
            assert any(n.endswith(f"schema/{filename}") for n in names), (
                f"{filename} is not packaged — an installed CLI could not resolve it"
            )
        entry_points = next(n for n in names if n.endswith(".dist-info/entry_points.txt"))
        # maturin writes `name=target` without spaces; normalize before comparing.
        registered = whl.read(entry_points).decode().replace(" ", "")
        assert "astro-mine-core=astro_mine.core.cli:main" in registered
        whl.extractall(extracted)

    # Run the CLI loaded **from the extracted wheel** (deps come from this interpreter, offline).
    mission = EXAMPLES / "mission" / "neo-sample-return-multiphase.mission.yaml"
    program = (
        "import sys; from astro_mine.core import cli;\n"
        f"assert {str(extracted)!r} in cli.__file__, cli.__file__\n"
        "sys.exit(cli.main(['validate', '--kind', 'mission', sys.argv[1]]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", program, str(mission)],
        env={"PYTHONPATH": str(extracted), "PATH": ""},
        capture_output=True,
        text=True,
    )
    if f"{extracted}" not in result.stderr and result.returncode == 3:  # pragma: no cover
        pytest.skip("could not force-load the wheel copy over an editable install")
    assert result.returncode == 0, f"wheel CLI failed: {result.stdout}\n{result.stderr}"


# ------------------------------------------------- self-identification without a $schema pointer


def _minimal_sadf() -> dict[str, object]:
    """The document `fleet new` writes, reduced to its required root."""
    return {
        "sadf_version": "0.1",
        "asset": {
            "identity": {
                "id": "example.rover",
                "name": "Example Rover",
                "version": "0.1.0",
                "kind": "rover",
            },
            "core_interface_versions": {"sadf": "0.1.0"},
            "root_frame": "base",
        },
    }


def test_a_document_that_identifies_itself_completely_is_routed() -> None:
    """The case the ``$schema`` route cannot reach.

    SADF's schema is ``additionalProperties: false`` at the root, so a SADF document may not carry
    a ``$schema`` key — the pointer that would identify it is the one key its own format forbids.
    Before this, no SADF file on disk could be routed by `astro-mine validate` at all.
    """
    assert cli.resolve_kind(_minimal_sadf(), None).slug == "sadf"


def test_resemblance_is_still_not_identification() -> None:
    """The line this rule is drawn on. A document must carry the *whole* required root, not just
    the discriminator: `{"sadf_version": "0.1"}` says what it would be without being it."""
    with pytest.raises(cli.KindError) as excinfo:
        cli.resolve_kind({"sadf_version": "0.1"}, None)
    assert "--kind" in str(excinfo.value)


def test_a_version_the_format_does_not_declare_is_not_a_match() -> None:
    """The discriminator is checked against the schema's ``const``, not merely for presence — a
    v9.9 document is not a v0.1 document, and validating it as one would certify a lie."""
    document = _minimal_sadf() | {"sadf_version": "9.9"}
    with pytest.raises(cli.KindError):
        cli.resolve_kind(document, None)


def test_an_explicit_kind_still_wins() -> None:
    """Precedence is unchanged: the user's word beats the document's."""
    assert cli.resolve_kind(_minimal_sadf(), "objective").slug == "objective"


def test_a_self_identifying_document_validates_end_to_end() -> None:
    """Routing is only useful if the document it routes then passes — this is the path
    `astro-mine validate <a scaffolded asset>` takes (RFC-0011 §6/§7)."""
    kind, issues = cli.validate_source(json.dumps(_minimal_sadf()), None)
    assert kind.slug == "sadf"
    assert issues == []
