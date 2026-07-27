"""Worlds's adapter for the umbrella CLI (`astro-mine worlds`).

Asserted here, at home, **without importing the umbrella** — `astro-mine-cli` is not a dependency
of this package and must not become one (``conventions.md §1.1``) — so these tests check the same
structural shape the umbrella checks at dispatch.

This is a *passthrough* adapter, so there is no flag list to keep in sync: the tail goes to
the same ``main`` `worlds` runs. What is worth pinning is that the tail arrives
**intact** — flags included, unparsed by the umbrella — and that the exit status survives
the hand-off.
"""

from __future__ import annotations

import argparse
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from astro_mine.worlds import umbrella
from astro_mine.worlds.cli import main
from astro_mine.worlds.spec import WorldSpec, example_world_spec_text

UMBRELLA_GROUP = "astro_mine.cli"


def test_adapter_satisfies_the_structural_contract() -> None:
    for member in ("name", "help", "add_arguments", "run"):
        assert hasattr(umbrella.worlds, member), f"missing {member!r}"
    assert callable(umbrella.worlds.add_arguments)
    assert callable(umbrella.worlds.run)
    assert umbrella.worlds.name == "worlds"
    assert umbrella.worlds.help


def test_the_entry_point_resolves_and_matches_its_name() -> None:
    """A typo in pyproject.toml is invisible until a user types the verb — unless this runs."""
    (ours,) = [
        ep
        for ep in entry_points(group=UMBRELLA_GROUP)
        if ep.value.startswith("astro_mine.worlds.umbrella:")
    ]
    assert ours.name == "worlds"
    assert ours.load().name == ours.name


def test_the_tail_arrives_intact() -> None:
    """The point of a passthrough: the umbrella parses the verb and nothing after it, so flags
    reach Worlds's own parser unmangled — including ones the umbrella has never heard of."""
    parser = argparse.ArgumentParser(prog="astro-mine worlds")
    umbrella.worlds.add_arguments(parser)
    args = parser.parse_args(["publish", "--some-flag", "value", "positional"])
    assert args.tail == ["publish", "--some-flag", "value", "positional"]


def test_the_exit_status_survives_the_handoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run` must *return* the status. A passthrough that dropped it would turn a failing command
    into a passing script."""
    monkeypatch.setattr(umbrella, "main", lambda argv: 3, raising=True)
    parser = argparse.ArgumentParser(prog="astro-mine worlds")
    umbrella.worlds.add_arguments(parser)
    assert umbrella.worlds.run(parser.parse_args(["publish"])) == 3


def test_a_usage_error_becomes_a_status_not_an_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """argparse inside Worlds raises SystemExit; the umbrella's contract is a returned status, so
    the adapter converts it. The code itself is unchanged."""

    def _exit(argv: list[str]) -> int:
        raise SystemExit(2)

    monkeypatch.setattr(umbrella, "main", _exit, raising=True)
    parser = argparse.ArgumentParser(prog="astro-mine worlds")
    umbrella.worlds.add_arguments(parser)
    assert umbrella.worlds.run(parser.parse_args(["publish"])) == 2


def test_a_clean_systemexit_is_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """`raise SystemExit()` with no code means success everywhere in Python; it must not become a
    spurious failure just because it crossed an adapter."""

    def _exit(argv: list[str]) -> int:
        raise SystemExit

    monkeypatch.setattr(umbrella, "main", _exit, raising=True)
    parser = argparse.ArgumentParser(prog="astro-mine worlds")
    umbrella.worlds.add_arguments(parser)
    assert umbrella.worlds.run(parser.parse_args(["publish"])) == 0


# --- the federated validator and the `world` scaffold (G2.11; RFC-0011 §6, §7) ----


def _scaffold(output: Path, *argv: str) -> int:
    """Drive the scaffold through the parser the umbrella would build for it.

    `output` and `--force` are declared here because the umbrella declares them before handing the
    parser over — reproducing that is what makes this a test of the real surface rather than of a
    namespace we invented.
    """
    parser = argparse.ArgumentParser(prog="astro-mine new world")
    parser.add_argument("output")
    parser.add_argument("--force", action="store_true")
    umbrella.world_scaffold.add_arguments(parser)
    return int(umbrella.world_scaffold.run(parser.parse_args([str(output), *argv])))


def test_the_validator_satisfies_the_structural_contract() -> None:
    for member in ("name", "claims", "validate"):
        assert hasattr(umbrella.validator, member), f"missing {member!r}"
    assert umbrella.validator.name == "worlds"


def test_the_scaffold_satisfies_the_structural_contract() -> None:
    """The same four members a verb has — RFC-0011 §7's group binds to the same contract."""
    for member in ("name", "help", "add_arguments", "run"):
        assert hasattr(umbrella.world_scaffold, member), f"missing {member!r}"
    assert umbrella.world_scaffold.name == "world"
    assert umbrella.world_scaffold.help


@pytest.mark.parametrize(
    ("group", "attribute", "entry_name"),
    [
        ("astro_mine.cli.validators", "validator", "worlds"),
        ("astro_mine.cli.scaffolds", "world_scaffold", "world"),
    ],
)
def test_the_entry_points_resolve_and_match_their_names(
    group: str, attribute: str, entry_name: str
) -> None:
    """A typo in pyproject.toml is invisible until a user types the command — unless this runs.
    For the scaffold the entry-point NAME is the kind as typed, so a wrong one is unreachable."""
    (ours,) = [
        ep
        for ep in entry_points(group=group)
        if ep.value == f"astro_mine.worlds.umbrella:{attribute}"
    ]
    assert ours.name == entry_name
    assert ours.load() is getattr(umbrella, attribute)


def test_the_claim_key_is_exactly_the_models_required_root() -> None:
    """The rule and the model must not drift apart.

    A WorldSpec is claimed by its required root rather than by a schema discriminator, because
    adding one would move `spec_hash` — hence `world_hash` — for every world already built,
    including the published anchor and the Bench scenarios pinned to it. That makes the claim key a
    derived fact, so it is derived here and compared: adding a required field to WorldSpec without
    revisiting the claim fails at this line rather than silently narrowing what gets recognized.
    """
    required = {
        name
        for name, field in WorldSpec.model_fields.items()
        if field.is_required()  # no default → part of the identifying root
    }
    assert required == umbrella.REQUIRED_ROOT


def test_it_claims_a_worldspec(tmp_path: Path) -> None:
    path = tmp_path / "example.world.yaml"
    path.write_text(example_world_spec_text(), encoding="utf-8")
    assert umbrella.validator.claims(str(path)) == "world_spec"


@pytest.mark.parametrize(
    ("name", "content"),
    [
        # Guard's and Mind's formats, which claim on their own version keys. If Worlds claimed
        # these too the umbrella would raise a collision — loud, but still a bug here.
        ("anchor.safety.yaml", "safety_version: '0.1'\nsafety:\n  id: x\n"),
        ("stack.yaml", "stack_spec_version: '0.1'\nstack_spec:\n  id: x\n"),
        # A Core document: self-describing by $schema, not by our root.
        ("objective.yaml", "objective_version: '0.1'\nobjective:\n  id: o\n"),
        # And the shapes a claim must survive without raising.
        ("partial.yaml", "world_id: x\ncrs: {}\n"),
        ("list.yaml", "- not\n- a mapping\n"),
        ("unparseable.yaml", "{{{ not yaml"),
    ],
)
def test_it_declines_what_is_not_its_own(tmp_path: Path, name: str, content: str) -> None:
    """Claiming is how the umbrella finds an owner, so a false claim steals another component's
    document — and claiming must be total: an unreadable or malformed file belongs to whoever owns
    it, and raising here would turn their problem into a Worlds traceback."""
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    assert umbrella.validator.claims(str(path)) is None


def test_it_declines_a_file_that_does_not_exist(tmp_path: Path) -> None:
    assert umbrella.validator.claims(str(tmp_path / "absent.yaml")) is None


def test_the_validator_runs_the_same_checker_the_cli_runs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Not a second implementation: the two surfaces cannot disagree about what is valid."""
    good = tmp_path / "good.world.yaml"
    good.write_text(example_world_spec_text(), encoding="utf-8")
    bad = tmp_path / "bad.world.yaml"
    bad.write_text("world_id: x\ncrs: {}\nregion: {}\nsource_dem: {}\n", encoding="utf-8")

    assert umbrella.validator.validate([str(good)], as_json=False) == 0
    assert "valid WorldSpec" in capsys.readouterr().out
    assert umbrella.validator.validate([str(bad)], as_json=False) == 1


def test_a_scaffolded_world_validates_with_no_hand_editing(tmp_path: Path) -> None:
    """**The acceptance criterion** for RFC-0011 §7. Checked through Worlds's own CLI, which is the
    same code path `astro-mine validate` routes a WorldSpec to."""
    out = tmp_path / "my.world.yaml"
    assert _scaffold(out) == 0
    assert main(["validate", str(out)]) == 0


def test_the_scaffold_writes_the_shipped_example_under_a_new_identity(tmp_path: Path) -> None:
    """One document, not two. A scaffold that emitted something other than the documented example
    would be a second thing to keep valid, and the two would drift."""
    out = tmp_path / "my.world.yaml"
    assert _scaffold(out, "--id", "acme-crater", "--world-version", "2.1.0") == 0
    spec = WorldSpec.from_yaml(out)
    assert (spec.world_id, spec.version) == ("acme-crater", "2.1.0")
    assert spec.region == WorldSpec.from_yaml_text(example_world_spec_text()).region
    # The comments come along: the scaffold's value is the explanation, not just the shape.
    assert "#" in out.read_text(encoding="utf-8")


def test_the_scaffold_refuses_to_overwrite_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "my.world.yaml"
    assert _scaffold(out) == 0
    out.write_text("# hand-edited\n", encoding="utf-8")
    assert _scaffold(out) == 1
    assert "file exists" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "# hand-edited\n"
    assert _scaffold(out, "--force") == 0
    assert "# hand-edited" not in out.read_text(encoding="utf-8")
