"""The shipped `WorldSpec` example, and the `astro-mine-worlds validate` checker (G2.11).

Worlds has been able to read a WorldSpec from YAML since the model existed and shipped no YAML: the
only real spec was authored in Python inside the anchor build script, so a user had nothing to copy
and nothing to check what they wrote against (UC-C5).

An example that is not exercised rots into a lie, which is worse than none — so these tests hold it
to the two things it claims: that it is a valid WorldSpec, and that it is the *same* document
`astro-mine new world` writes. The checker is held to reporting every failure rather than the first,
because a user fixing an authoring mistake wants the whole list.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from astro_mine.worlds.cli import main
from astro_mine.worlds.spec import (
    EXAMPLE_RESOURCE,
    JSON_SCHEMA_DIALECT,
    WORLDSPEC_SCHEMA_ID,
    WorldSpec,
    example_world_spec_text,
    published_json_schema,
    published_json_schema_text,
)

#: Pinned so a change to the example is a deliberate act with a visible diff, not a side effect.
#: `spec_hash` is what `world_hash` is built on, so a silent move here would silently re-address
#: every world built from this document.
EXAMPLE_SPEC_HASH = "sha256:b5d02c80be0e4d3b73a686186e9519fe7cac54ec8c0db0fde78ad4c5a5554113"


def test_the_example_is_a_valid_worldspec() -> None:
    """The whole point. An example a user copies must load — including through the explicit-CRS
    gate, which rejects an implicit Earth datum on a lunar body."""
    spec = WorldSpec.from_yaml_text(example_world_spec_text())
    assert spec.world_id == "example-polar-basin"
    assert spec.crs.body == "MOON"
    assert spec.spec_hash == EXAMPLE_SPEC_HASH


def test_the_example_round_trips_through_its_own_serializer() -> None:
    """`from_yaml` / `to_yaml` are the documented front door in both directions; a document that
    survives one pass but not the reverse would make the spec unstable under any tool that edits
    and rewrites it."""
    spec = WorldSpec.from_yaml_text(example_world_spec_text())
    assert WorldSpec.from_yaml_text(spec.to_yaml()) == spec
    assert WorldSpec.from_yaml_text(spec.to_yaml()).spec_hash == EXAMPLE_SPEC_HASH


def test_the_example_resolves_as_package_data() -> None:
    """Shipped in the wheel, not merely in the repo. A file outside the wheel reaches repo cloners
    and nobody else — the gap Guard's anchor.safety.yaml had (G2.7) — and `astro-mine new world`
    reads this through `importlib.resources`, so an installed user gets nothing if it is missing."""
    assert EXAMPLE_RESOURCE.endswith(".world.yaml")
    assert example_world_spec_text().lstrip().startswith("#")


def test_the_example_documents_itself() -> None:
    """It is meant to be read, not just parsed. The comments are the reason it is authored by hand
    rather than dumped from the model — a bare `to_yaml()` would be valid and teach nothing."""
    text = example_world_spec_text()
    assert "#" in text
    # The two things a user most needs told: that the CRS must be explicit, and that the
    # illumination/PSR knobs define the mask rather than tune it.
    assert "require_crs" in text or "implicit Earth datum" in text
    assert "PSR" in text


def test_substituting_an_identity_changes_only_the_identity() -> None:
    """What the scaffold does. Anchored line replacement rather than a blind `str.replace`, so a
    value that also appears in a comment or a description is not silently rewritten."""
    spec = WorldSpec.from_yaml_text(
        example_world_spec_text(world_id="acme-crater", version="2.1.0")
    )
    assert spec.world_id == "acme-crater"
    assert spec.version == "2.1.0"

    baseline = WorldSpec.from_yaml_text(example_world_spec_text())
    assert spec.region == baseline.region
    assert spec.crs == baseline.crs
    assert spec.layers == baseline.layers
    assert spec.source_dem == baseline.source_dem


def test_the_synthetic_source_is_marked_as_synthetic() -> None:
    """`content_hash: null` is the model's documented marker for an illustrative source. A worked
    example that pinned a fake digest would teach exactly the wrong habit about reproducibility."""
    spec = WorldSpec.from_yaml_text(example_world_spec_text())
    assert spec.source_dem.content_hash is None
    assert "synthetic" in (spec.source_dem.description or "").lower()


# --------------------------------------------------------------------------- the checker


def test_validate_accepts_the_shipped_example(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "example.world.yaml"
    out.write_text(example_world_spec_text(), encoding="utf-8")
    assert main(["validate", str(out)]) == 0
    assert "valid WorldSpec example-polar-basin" in capsys.readouterr().out


def test_validate_reports_every_failure_not_just_the_first(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A user fixing an authoring mistake wants the whole list. Stopping at the first turns one
    editing pass into as many passes as there are errors."""
    good = tmp_path / "good.world.yaml"
    good.write_text(example_world_spec_text(), encoding="utf-8")
    bad_a = tmp_path / "a.world.yaml"
    bad_a.write_text("world_id: x\n", encoding="utf-8")
    bad_b = tmp_path / "b.world.yaml"
    bad_b.write_text("{{{ not yaml", encoding="utf-8")

    assert main(["validate", str(good), str(bad_a), str(bad_b)]) == 1
    captured = capsys.readouterr()
    assert "valid WorldSpec" in captured.out
    assert str(bad_a) in captured.err
    assert str(bad_b) in captured.err
    # An authoring error is a message, never a traceback out of our own frames.
    assert "Traceback" not in captured.err


def test_validate_reports_a_missing_file_rather_than_raising(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["validate", str(tmp_path / "nope.world.yaml")]) == 1
    assert "Traceback" not in capsys.readouterr().err


def test_validate_emits_machine_readable_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--json` is what a CI gate reads; the text form is for a person. Both report the same
    verdict, and the JSON carries the spec hash so a pipeline can record what it checked."""
    good = tmp_path / "good.world.yaml"
    good.write_text(example_world_spec_text(), encoding="utf-8")
    bad = tmp_path / "bad.world.yaml"
    bad.write_text("world_id: x\n", encoding="utf-8")

    assert main(["validate", str(good), str(bad), "--json"]) == 1
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is False
    assert report["valid"][0]["hash"] == EXAMPLE_SPEC_HASH
    assert report["invalid"][0]["path"] == str(bad)


# --------------------------------------------------------------------------- the published schema


def test_the_schema_is_self_identifying() -> None:
    """A published schema without an `$id` is a file, not a contract: two copies at two paths are
    indistinguishable and neither can be `$ref`-ed. The namespace is the one Worlds' illumination
    schemas already use, so the platform has one shape rather than three (RFC-0009 rule 1)."""
    schema = WorldSpec.json_schema()
    assert schema["$id"] == WORLDSPEC_SCHEMA_ID
    assert schema["$schema"] == JSON_SCHEMA_DIALECT
    assert WORLDSPEC_SCHEMA_ID.startswith("https://schemas.astro-mine.org/worlds/")


def test_the_shipped_schema_is_what_the_model_generates() -> None:
    """**The anti-drift guard.**

    The model is the source of truth and the schema is derived, so a shipped copy can silently fall
    behind the code it claims to describe — which is worse than shipping none, because a consumer
    would be validating against a contract nobody honours. Regenerate rather than hand-edit:

        astro-mine-worlds schema > src/astro_mine/worlds/spec/schema/worldspec.schema.json
    """
    assert published_json_schema() == WorldSpec.json_schema(), (
        "the shipped WorldSpec schema is stale; regenerate it with "
        "`astro-mine-worlds schema > src/astro_mine/worlds/spec/schema/worldspec.schema.json`"
    )


def test_the_shipped_example_validates_against_the_shipped_schema() -> None:
    """The claim publishing a schema actually makes: that it and the model agree about a real
    document. Checked with a real Draft 2020-12 validator, against the *shipped* bytes — so this
    is the same check a non-Python consumer would run, not a Pydantic round-trip in disguise."""
    import jsonschema

    schema = published_json_schema()
    document = yaml.safe_load(example_world_spec_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(document)


def test_the_schema_rejects_what_the_model_rejects(tmp_path: Path) -> None:
    """A schema that accepted everything would validate nothing. The extent rule is a model-level
    validator and legitimately outside JSON Schema's reach, so the case pinned here is one the
    schema itself must catch: a missing required root member."""
    import jsonschema

    document = yaml.safe_load(example_world_spec_text())
    del document["world_id"]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(published_json_schema()).validate(document)


def test_the_schema_command_prints_the_shipped_bytes(capsys: pytest.CaptureFixture[str]) -> None:
    """What a user pipes to a file must be exactly what the wheel publishes under that `$id` —
    printing a regenerated document instead would make the command a second source of truth."""
    assert main(["schema"]) == 0
    printed = capsys.readouterr().out
    assert json.loads(printed) == published_json_schema()
    assert published_json_schema_text().rstrip("\n") == printed.rstrip("\n")
