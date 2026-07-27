"""Core's validator for the federated `astro-mine validate` (RFC-0011 §6).

Core no longer registers the `validate` **verb** — that moved to the umbrella, because routing a
document to the component that owns its format is something no single component can do without
importing its siblings. Core contributes what it actually owns: `$id`-keyed dispatch over the nine
Core-authored formats.

Asserted here, at home, **without importing the umbrella** — `astro-mine-cli` is not a dependency
of this package and must never become one (``conventions.md §1.1``): Core is the narrow waist, and
that edge would point the wrong way through the whole platform.
"""

from __future__ import annotations

import json
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from astro_mine.core import umbrella

VALIDATOR_GROUP = "astro_mine.cli.validators"
EXAMPLES = Path(__file__).parents[2] / "examples"


def test_validator_satisfies_the_structural_contract() -> None:
    for member in ("name", "claims", "validate"):
        assert hasattr(umbrella.validator, member), f"missing {member!r}"
    assert callable(umbrella.validator.claims)
    assert callable(umbrella.validator.validate)
    assert umbrella.validator.name == "core"


def test_the_entry_point_resolves_into_the_validators_group() -> None:
    """A typo in pyproject.toml is invisible until a user runs the verb — unless this runs."""
    (ours,) = [
        ep
        for ep in entry_points(group=VALIDATOR_GROUP)
        if ep.value.startswith("astro_mine.core.umbrella:")
    ]
    assert ours.name == "core"
    assert ours.load().name == "core"


def test_core_no_longer_registers_the_validate_verb() -> None:
    """The verb is the umbrella's. If Core also advertised it the umbrella would refuse to run at
    all — a built-in shadowed by a distribution is a hard error there, not a silent winner."""
    verbs = {ep.name for ep in entry_points(group="astro_mine.cli")}
    assert "validate" not in verbs


def test_claims_a_self_describing_core_document(tmp_path: Path) -> None:
    document = tmp_path / "objective.yaml"
    document.write_text(
        json.dumps(
            {"$schema": "https://schemas.astro-mine.org/core/objective/v0.1/objective.schema.json"}
        )
    )
    assert umbrella.validator.claims(str(document)) == "objective"


def test_declines_a_document_it_does_not_own(tmp_path: Path) -> None:
    """A SafetySpec is Guard's. Core must say "not mine" rather than fail it — claiming is how the
    umbrella finds the owner, so a false claim would steal another component's document."""
    spec = tmp_path / "anchor.safety.yaml"
    spec.write_text("safety_version: '0.1'\nsafety:\n  id: x\n")
    assert umbrella.validator.claims(str(spec)) is None


@pytest.mark.parametrize(
    ("name", "content"),
    [
        ("missing.yaml", None),
        ("unparseable.yaml", "{{{ not yaml"),
        ("not-a-mapping.yaml", "- just\n- a list\n"),
        ("no-schema.yaml", "some_key: value\n"),
    ],
)
def test_declines_rather_than_raises(tmp_path: Path, name: str, content: str | None) -> None:
    """Claiming runs against every file the umbrella routes, including other components'. It has
    to be total: raising here would turn someone else's malformed document into a Core traceback."""
    path = tmp_path / name
    if content is not None:
        path.write_text(content)
    assert umbrella.validator.claims(str(path)) is None


def test_validate_runs_the_same_checker_as_the_native_cli(tmp_path: Path) -> None:
    """Not a second implementation — the same `_cmd_validate` the native CLI dispatches to.

    Note what this test cannot use: the shipped `examples/` documents declare **no `$schema`**, so
    the validator cannot claim them and federated routing cannot reach them. On the native CLI
    that is a nuisance you work around with `--kind`; under federation it is load-bearing, because
    the umbrella has to identify the owner *before* anyone can be told the kind. Giving the
    examples a `$schema` is tracked in astro-mine/docs#39 (and astro-mine/docs#38 promises
    "a real example that validates").
    """
    document = tmp_path / "manifest.yaml"
    source = (EXAMPLES / "plugins/greedy-prospecting-baseline.manifest.yaml").read_text()
    document.write_text(
        '$schema: "https://schemas.astro-mine.org/core/registry/v0.1/manifest.schema.json"\n'
        + source
    )
    assert umbrella.validator.claims(str(document)) == "manifest"
    assert umbrella.validator.validate([str(document)], as_json=False) == 0


def test_an_invalid_document_returns_non_zero(tmp_path: Path) -> None:
    """The status is what the umbrella turns into the process exit code."""
    broken = tmp_path / "broken.json"
    broken.write_text(
        json.dumps(
            {
                "$schema": "https://schemas.astro-mine.org/core/objective/v0.1/objective.schema.json",
                "objective_id": "missing-everything-else",
            }
        )
    )
    assert umbrella.validator.validate([str(broken)], as_json=False) != 0


def test_claims_a_document_that_identifies_itself_without_a_schema_pointer(tmp_path: Path) -> None:
    """A SADF document — the format that cannot carry a ``$schema`` key, because its own schema is
    ``additionalProperties: false`` at the root.

    Until this, `astro-mine validate` could not route a single SADF file: the pointer Core claimed
    on was the one key SADF forbids, so every asset came back *"no installed validator recognizes
    it"*. That is the whole reason this route exists, and `astro-mine new asset` (RFC-0011 §7) is
    what made it visible — a scaffold whose output the platform's own validator refused.
    """
    asset = tmp_path / "rover.sadf.yaml"
    asset.write_text(
        "sadf_version: '0.1'\n"
        "asset:\n"
        "  identity: {id: example.rover, name: Example Rover, version: '0.1.0', kind: rover}\n"
        "  core_interface_versions: {sadf: '0.1.0'}\n"
        "  root_frame: base\n",
        encoding="utf-8",
    )
    assert umbrella.validator.claims(str(asset)) == "sadf"


def test_still_declines_a_sibling_format_that_identifies_itself(tmp_path: Path) -> None:
    """The risk this route introduces, closed. Guard and Mind identify their formats the same way
    (``safety_version``, ``stack_spec_version``), so a Core claim keyed on a version alone would
    start stealing their documents. Requiring the schema's whole required root is what keeps the
    federation's owners distinct — and two claimants would be a hard error, not a silent winner."""
    stack = tmp_path / "stack.yaml"
    stack.write_text("stack_spec_version: '0.1'\nstack_spec:\n  id: x\n", encoding="utf-8")
    assert umbrella.validator.claims(str(stack)) is None
