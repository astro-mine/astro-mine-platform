"""Fleet's contributions to the umbrella CLI — the `astro-mine fleet` verb and `new asset`.

Asserted here, at home, **without importing the umbrella** — `astro-mine-cli` is not a dependency
of this package and must not become one (``conventions.md §1.1``) — so these tests check the same
structural shape the umbrella checks at dispatch.

The verb is a *passthrough* adapter, so there is no flag list to keep in sync: the tail goes to
the same ``main`` `fleet` runs. What is worth pinning is that the tail arrives
**intact** — flags included, unparsed by the umbrella — and that the exit status survives
the hand-off.

The scaffold is where **the acceptance criterion for RFC-0011 §7 actually lives**: *a scaffolded
document must validate with no hand-editing*. No test in `astro-mine-cli` can check that — it
installs no components by design — so it is checked here, by the package that owns both the
template and the checker. That is the whole argument for federating rather than centralizing.
"""

from __future__ import annotations

import argparse
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from astro_mine.core.sadf import validate_sadf
from astro_mine.fleet import umbrella

UMBRELLA_GROUP = "astro_mine.cli"
SCAFFOLD_GROUP = "astro_mine.cli.scaffolds"


def test_adapter_satisfies_the_structural_contract() -> None:
    for member in ("name", "help", "add_arguments", "run"):
        assert hasattr(umbrella.fleet, member), f"missing {member!r}"
    assert callable(umbrella.fleet.add_arguments)
    assert callable(umbrella.fleet.run)
    assert umbrella.fleet.name == "fleet"
    assert umbrella.fleet.help


def test_the_entry_point_resolves_and_matches_its_name() -> None:
    """A typo in pyproject.toml is invisible until a user types the verb — unless this runs."""
    (ours,) = [
        ep
        for ep in entry_points(group=UMBRELLA_GROUP)
        if ep.value.startswith("astro_mine.fleet.umbrella:")
    ]
    assert ours.name == "fleet"
    assert ours.load().name == ours.name


def test_the_tail_arrives_intact() -> None:
    """The point of a passthrough: the umbrella parses the verb and nothing after it, so flags
    reach Fleet's own parser unmangled — including ones the umbrella has never heard of."""
    parser = argparse.ArgumentParser(prog="astro-mine fleet")
    umbrella.fleet.add_arguments(parser)
    args = parser.parse_args(["validate", "--some-flag", "value", "positional"])
    assert args.tail == ["validate", "--some-flag", "value", "positional"]


def test_a_successful_command_returns_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fleet's `main` is typed ``-> None``: unlike every other component's, it does not return a
    status at all. It signals success by returning and failure by raising ``SystemExit`` (see
    ``cli.main``), so the adapter maps a clean return to 0 — and the SystemExit tests below are
    where Fleet's *failure* path is actually pinned."""
    called: list[list[str]] = []
    monkeypatch.setattr(umbrella, "main", called.append, raising=True)
    parser = argparse.ArgumentParser(prog="astro-mine fleet")
    umbrella.fleet.add_arguments(parser)
    assert umbrella.fleet.run(parser.parse_args(["validate", "asset.yaml"])) == 0
    assert called == [["validate", "asset.yaml"]]


def test_a_failing_command_propagates_its_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fleet's real failure path: a non-zero handler result becomes ``SystemExit(code)``, and the
    adapter has to turn that back into the status the umbrella will exit with. A passthrough that
    dropped it would turn a failing command into a passing script."""

    def _fail(argv: list[str]) -> None:
        raise SystemExit(3)

    monkeypatch.setattr(umbrella, "main", _fail, raising=True)
    parser = argparse.ArgumentParser(prog="astro-mine fleet")
    umbrella.fleet.add_arguments(parser)
    assert umbrella.fleet.run(parser.parse_args(["validate"])) == 3


def test_a_usage_error_becomes_a_status_not_an_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """argparse inside Fleet raises SystemExit; the umbrella's contract is a returned status, so
    the adapter converts it. The code itself is unchanged."""

    def _exit(argv: list[str]) -> int:
        raise SystemExit(2)

    monkeypatch.setattr(umbrella, "main", _exit, raising=True)
    parser = argparse.ArgumentParser(prog="astro-mine fleet")
    umbrella.fleet.add_arguments(parser)
    assert umbrella.fleet.run(parser.parse_args(["validate"])) == 2


def test_a_clean_systemexit_is_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """`raise SystemExit()` with no code means success everywhere in Python; it must not become a
    spurious failure just because it crossed an adapter."""

    def _exit(argv: list[str]) -> int:
        raise SystemExit

    monkeypatch.setattr(umbrella, "main", _exit, raising=True)
    parser = argparse.ArgumentParser(prog="astro-mine fleet")
    umbrella.fleet.add_arguments(parser)
    assert umbrella.fleet.run(parser.parse_args(["validate"])) == 0


# --- the `asset` scaffold (`astro-mine new asset`) --------------------------------


def _scaffold(tmp_path: Path, *argv: str) -> tuple[int, Path]:
    """Drive the scaffold through the parser the umbrella would build for it.

    ``output`` and ``--force`` are declared here because the umbrella declares them before handing
    the parser over — reproducing that is what makes this a test of the real surface rather than of
    a namespace we invented.
    """
    parser = argparse.ArgumentParser(prog="astro-mine new asset")
    parser.add_argument("output")
    parser.add_argument("--force", action="store_true")
    umbrella.asset_scaffold.add_arguments(parser)
    out = tmp_path / "scaffolded.sadf.yaml"
    return umbrella.asset_scaffold.run(parser.parse_args([str(out), *argv])), out


def test_the_scaffold_satisfies_the_structural_contract() -> None:
    """The same four members a verb has — RFC-0011 §7's group binds to the same contract, so a
    component writes the object it already knows how to write."""
    for member in ("name", "help", "add_arguments", "run"):
        assert hasattr(umbrella.asset_scaffold, member), f"missing {member!r}"
    assert callable(umbrella.asset_scaffold.add_arguments)
    assert callable(umbrella.asset_scaffold.run)
    assert umbrella.asset_scaffold.name == "asset"
    assert umbrella.asset_scaffold.help


def test_the_scaffold_entry_point_resolves_and_matches_its_name() -> None:
    """The entry-point NAME is the kind the user types, so a typo here is a kind nobody can reach.
    Invisible until someone runs `astro-mine new asset` — unless this runs."""
    (ours,) = [
        ep
        for ep in entry_points(group=SCAFFOLD_GROUP)
        if ep.value.startswith("astro_mine.fleet.umbrella:")
    ]
    assert ours.name == "asset"
    assert ours.load().name == ours.name


def test_a_scaffolded_asset_is_valid_sadf_with_no_hand_editing(tmp_path: Path) -> None:
    """**The acceptance criterion.** Everything else about this feature is plumbing: if the
    document a user is handed does not pass the gate, the scaffold has cost them time instead of
    saving it. Checked against Core's own validator, which is the same gate `astro-mine validate`
    routes SADF to."""
    status, out = _scaffold(tmp_path)
    assert status == 0
    validate_sadf(out.read_text(encoding="utf-8"))


def test_it_produces_exactly_what_fleet_new_produces(tmp_path: Path) -> None:
    """One template, two roads to it. The adapter calls the same handler rather than re-rendering,
    so the umbrella's output cannot drift from the component's — a drift that would be invisible,
    because both paths would keep producing valid documents while producing *different* ones."""
    from astro_mine.fleet.cli import main

    direct = tmp_path / "direct.sadf.yaml"
    main(["new", "rover", str(direct)])
    _, through_umbrella = _scaffold(tmp_path)
    assert through_umbrella.read_text(encoding="utf-8") == direct.read_text(encoding="utf-8")


def test_the_kind_and_identity_are_the_users_to_choose(tmp_path: Path) -> None:
    """`fleet new` takes the kind as a positional; under the umbrella the first positional is
    already `output`, so it becomes an option — with a default, so the bare command still works."""
    status, out = _scaffold(tmp_path, "--kind", "excavator", "--id", "acme.digger")
    assert status == 0
    text = out.read_text(encoding="utf-8")
    assert '"excavator"' in text
    assert '"acme.digger"' in text
    validate_sadf(text)


def test_it_refuses_to_overwrite_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Inherited from the handler, and asserted rather than assumed: `--force` is declared by the
    umbrella, so a scaffold that ignored it would silently destroy a user's authored file."""
    status, out = _scaffold(tmp_path)
    assert status == 0
    out.write_text("# hand-edited\n", encoding="utf-8")

    assert _scaffold(tmp_path)[0] == 1
    assert "file exists" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "# hand-edited\n"

    assert _scaffold(tmp_path, "--force")[0] == 0
    assert "# hand-edited" not in out.read_text(encoding="utf-8")
