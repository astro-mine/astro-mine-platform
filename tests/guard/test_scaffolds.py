"""Guard's scaffold for the umbrella CLI (`astro-mine new safety`).

Asserted here, at home, **without importing the umbrella** — `astro-mine-cli` is not a dependency
of this package and must not become one (``conventions.md §1.1``) — so these tests check the same
structural shape the umbrella checks at dispatch.

This is where **the acceptance criterion for RFC-0011 §7 lives** for Guard's kind: *a scaffolded
document must validate with no hand-editing*. No test in `astro-mine-cli` can check that — it
installs no components by design — so it is checked by the package that owns both the template and
the checker.

For a safety contract, validating is the floor rather than the bar. The template makes three claims
a user would otherwise have to learn by being burned — that silence in ``admissible_directives``
grants nothing, that ``on_uncertain`` can never be ``passthrough``, and that a signal is declared
once and referenced by key — and each is pinned below against the model, so the template cannot
drift into teaching something false.
"""

from __future__ import annotations

import argparse
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from astro_mine.guard import scaffolds
from astro_mine.guard.cli import main
from astro_mine.guard.spec.loader import load_safety_spec

SCAFFOLD_GROUP = "astro_mine.cli.scaffolds"


def _scaffold(output: Path, *argv: str) -> int:
    """Drive the scaffold through the parser the umbrella would build for it.

    ``output`` and ``--force`` are declared here because the umbrella declares them before handing
    the parser over — reproducing that is what makes this a test of the real surface rather than of
    a namespace we invented.
    """
    parser = argparse.ArgumentParser(prog="astro-mine new safety")
    parser.add_argument("output")
    parser.add_argument("--force", action="store_true")
    scaffolds.safety_scaffold.add_arguments(parser)
    return int(scaffolds.safety_scaffold.run(parser.parse_args([str(output), *argv])))


def test_the_scaffold_satisfies_the_structural_contract() -> None:
    """The same four members a verb has: RFC-0011 §7's group binds to the same contract, so a
    component writes the object it already knows how to write."""
    for member in ("name", "help", "add_arguments", "run"):
        assert hasattr(scaffolds.safety_scaffold, member), f"missing {member!r}"
    assert scaffolds.safety_scaffold.name == "safety"
    assert scaffolds.safety_scaffold.help


def test_the_entry_point_resolves_and_matches_its_name() -> None:
    """The entry-point NAME is the kind the user types, so a typo here is a kind nobody can reach —
    invisible until someone runs `astro-mine new safety`, unless this runs."""
    (ours,) = [
        ep
        for ep in entry_points(group=SCAFFOLD_GROUP)
        if ep.value.startswith("astro_mine.guard.scaffolds:")
    ]
    assert ours.name == "safety"
    assert ours.load().name == "safety"


def test_a_scaffolded_spec_validates_with_no_hand_editing(tmp_path: Path) -> None:
    """**The acceptance criterion**, run through Guard's own CLI — the same code path
    `astro-mine validate` routes a SafetySpec to."""
    out = tmp_path / "my.safety.yaml"
    assert _scaffold(out) == 0
    assert main(["validate", str(out)]) == 0


def test_a_scaffolded_spec_compiles(tmp_path: Path) -> None:
    """Validation says the document is well-formed; compilation is what the trusted core actually
    enforces. A contract that validates and then fails to lower would have cost the user the hour
    they spent authoring against it."""
    out = tmp_path / "my.safety.yaml"
    assert _scaffold(out) == 0
    assert main(["compile", str(out), "--out", str(tmp_path / "ir.json")]) == 0


def test_it_grants_directives_explicitly_because_silence_grants_nothing(tmp_path: Path) -> None:
    """The trap the template exists to close.

    An absent or empty `admissible_directives` certifies **no** directive, whatever deployment
    configuration says — the deliberate asymmetry with `kinematic_limit`, where an absent authored
    limit lets the configured one stand. A scaffold that omitted the block would emit a contract
    that validates cleanly and then silently refuses every directive the user's stack sends, and
    they would go looking in the stack.
    """
    out = tmp_path / "my.safety.yaml"
    assert _scaffold(out) == 0
    spec = load_safety_spec(out.read_text(encoding="utf-8")).safety
    assert spec.admissible_directives is not None
    assert spec.admissible_directives.modes
    assert spec.admissible_directives.tasks


def test_no_constraint_can_fail_open(tmp_path: Path) -> None:
    """`on_uncertain` is written out at its default rather than left implicit, because the one
    place an author will want to let something through is the one place they cannot. Pinned against
    the model so the template cannot drift into showing a value the schema forbids."""
    out = tmp_path / "my.safety.yaml"
    assert _scaffold(out) == 0
    spec = load_safety_spec(out.read_text(encoding="utf-8")).safety
    assert spec.constraints
    for constraint in spec.constraints:
        assert constraint.on_uncertain != "passthrough"


def test_every_constraint_reads_a_declared_signal(tmp_path: Path) -> None:
    """The vocabulary and the constraint are separate on purpose, and a constraint referencing an
    undeclared key is the first mistake an author makes. The template has to model it correctly."""
    out = tmp_path / "my.safety.yaml"
    assert _scaffold(out) == 0
    spec = load_safety_spec(out.read_text(encoding="utf-8")).safety
    declared = {signal.key for signal in spec.signals}
    for constraint in spec.constraints:
        block = getattr(constraint, str(constraint.kind))
        assert block.signal in declared


def test_the_identity_is_the_users_to_choose(tmp_path: Path) -> None:
    out = tmp_path / "my.safety.yaml"
    assert _scaffold(out, "--id", "acme-safety", "--scenario-ref", "s") == 0
    spec = load_safety_spec(out.read_text(encoding="utf-8")).safety
    assert spec.id == "acme-safety"
    assert spec.scenario_ref == "s"


def test_it_refuses_to_overwrite_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A safety contract is exactly the file a user will have edited before re-running a command
    from their shell history. `--force` is declared by the umbrella; honouring it is required."""
    out = tmp_path / "my.safety.yaml"
    assert _scaffold(out) == 0
    out.write_text("# hand-edited\n", encoding="utf-8")
    assert _scaffold(out) == 1
    assert "file exists" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "# hand-edited\n"
    assert _scaffold(out, "--force") == 0
    assert "# hand-edited" not in out.read_text(encoding="utf-8")
