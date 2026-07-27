"""`astro-mine plugin new` — scaffolding a package that extends the platform (RFC-0011 §7).

The verb is the same engine `astro-mine new` uses over a second entry-point group, so the routing
tests live in `test_new.py` and what is pinned here is the half that is specific: the action level
(`plugin new`, not `plugin`), and the one kind the umbrella owns.

**Why `cli` is a built-in at all.** Every other kind belongs to the component that hosts the
extension group; `astro_mine.cli` is hosted here, so its scaffold has nowhere else to live — the
same reasoning that makes `validate` a built-in verb. It also makes the acceptance claim testable
in this repo: the emitted package is checked here for shape, and installed for real in
`test_installed_provider.py`, where the verb it registers actually runs.
"""

from __future__ import annotations

import argparse
import tomllib
from pathlib import Path

import pytest
from _verbs import make_entry_point

from astro_mine.cli import PLUGIN_SCAFFOLD_GROUP
from astro_mine.cli._new import plugin
from astro_mine.cli._protocol import check_subcommand
from astro_mine.cli._templates import CLI_PLUGIN_SCAFFOLD


def _scaffold(*argv: str) -> int:
    return plugin.run(argparse.Namespace(action="new", rest=["cli", *argv]))


def _emitted(target: Path, module: str = "my-verb") -> tuple[dict, str]:
    """The generated packaging metadata (parsed) and the generated module source."""
    manifest = tomllib.loads((target / "pyproject.toml").read_text(encoding="utf-8"))
    package = module.replace("-", "_")
    return manifest, (target / "src" / package / "__init__.py").read_text(encoding="utf-8")


def test_bare_plugin_lists_the_kinds(capsys: pytest.CaptureFixture[str]) -> None:
    assert plugin.run(argparse.Namespace(action=None, rest=[])) == 0
    out = capsys.readouterr().out
    assert "usage: astro-mine plugin new <kind>" in out
    assert "cli" in out
    # In astro-mine-platform every first-party owner ships in this distribution, so `solver`
    # is a *registered* kind (the original env listed it as "not installed, provided by
    # astro-mine-allocate" — a state that no longer exists).
    assert "solver" in out


def test_plugin_new_with_no_kind_lists_rather_than_erroring(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert plugin.run(argparse.Namespace(action="new", rest=[])) == 0
    assert "usage: astro-mine plugin new <kind>" in capsys.readouterr().out


@pytest.mark.parametrize("flag", ["-h", "--help"])
def test_asking_for_help_lists_the_kinds_rather_than_reading_it_as_one(
    flag: str, capsys: pytest.CaptureFixture[str]
) -> None:
    """`--help` must not arrive as a kind name.

    The tail is an `argparse.REMAINDER`, so argparse hands the flag through instead of acting on
    it; before this was claimed explicitly the verb answered `unknown kind '--help'` and exited 2,
    while its sibling `astro-mine new --help` printed help and exited 0.
    """
    assert plugin.run(argparse.Namespace(action="new", rest=[flag])) == 0
    captured = capsys.readouterr()
    assert "usage: astro-mine plugin new <kind>" in captured.out
    assert captured.err == ""


def test_a_kinds_own_help_still_reaches_the_kind(capsys: pytest.CaptureFixture[str]) -> None:
    """Only a *leading* help flag is the verb's. `plugin new cli --help` is the scaffold's."""
    with pytest.raises(SystemExit) as exit_info:
        plugin.run(argparse.Namespace(action="new", rest=["cli", "--help"]))
    assert exit_info.value.code == 0
    assert "astro-mine plugin new cli" in capsys.readouterr().out


def test_an_unknown_action_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    """`new` is the only action today. An unrecognized one is a usage error, not a silent no-op."""
    assert plugin.run(argparse.Namespace(action="delete", rest=[])) == 2
    err = capsys.readouterr().err
    assert "unknown action 'delete'" in err
    assert "Traceback" not in err


def test_the_scaffold_writes_a_package_that_registers_a_verb(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The entry point is the whole contract: if it is not in the metadata, nothing else is."""
    target = tmp_path / "my-verb"
    assert _scaffold(str(target)) == 0
    manifest, _ = _emitted(target)
    assert manifest["project"]["entry-points"]["astro_mine.cli"] == {"my-verb": "my_verb:my_verb"}
    assert manifest["project"]["name"] == "my-verb"
    assert "wrote" in capsys.readouterr().out


def test_the_generated_package_does_not_depend_on_the_umbrella(tmp_path: Path) -> None:
    """The layering rule, taught by example. A scaffold that quietly added `astro-mine-cli` to the
    generated dependencies would teach every third-party author to invert it (`conventions.md
    §1.1`) — and the umbrella would become a dependency of the ecosystem it dispatches."""
    target = tmp_path / "my-verb"
    assert _scaffold(str(target)) == 0
    manifest, module = _emitted(target)
    assert manifest["project"]["dependencies"] == []
    # The module *names* the umbrella in prose — that is the lesson. What it must never do is
    # import it, which is the difference between explaining the contract and depending on it.
    assert "import astro_mine" not in module
    assert "from astro_mine" not in module


def test_the_generated_module_is_valid_python_and_satisfies_the_contract(tmp_path: Path) -> None:
    """A scaffold whose output does not run is worse than no scaffold: the user debugs *our*
    template before writing a line of their own. So the emitted module is executed and the object
    the entry point names is put through the same checker dispatch would use."""
    target = tmp_path / "my-verb"
    assert _scaffold(str(target)) == 0
    _, module = _emitted(target)
    namespace: dict[str, object] = {}
    exec(compile(module, "generated", "exec"), namespace)

    subcommand = check_subcommand(namespace["my_verb"], verb="my-verb")
    assert subcommand.name == "my-verb"
    parser = argparse.ArgumentParser()
    subcommand.add_arguments(parser)
    assert subcommand.run(parser.parse_args(["--name", "moon"])) == 0


def test_names_can_be_chosen_independently_of_the_directory(tmp_path: Path) -> None:
    """The defaults derive everything from the directory name, which is right for the common case
    and wrong as soon as the distribution, the import package and the verb should differ."""
    target = tmp_path / "somewhere"
    assert (
        _scaffold(
            str(target), "--distribution", "acme-tools", "--module", "acme_tools", "--verb", "acme"
        )
        == 0
    )
    manifest, module = _emitted(target, "acme_tools")
    assert manifest["project"]["name"] == "acme-tools"
    assert manifest["project"]["entry-points"]["astro_mine.cli"] == {"acme": "acme_tools:acme"}
    assert 'name = "acme"' in module


def test_it_refuses_to_overwrite_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "my-verb"
    assert _scaffold(str(target)) == 0
    (target / "pyproject.toml").write_text("# hand-edited\n", encoding="utf-8")

    assert _scaffold(str(target)) == 1
    assert "file exists" in capsys.readouterr().err
    assert (target / "pyproject.toml").read_text(encoding="utf-8") == "# hand-edited\n"

    assert _scaffold(str(target), "--force") == 0
    assert "# hand-edited" not in (target / "pyproject.toml").read_text(encoding="utf-8")


@pytest.mark.parametrize(
    ("argument", "expected"),
    [
        ("--distribution=not a name!", "not a usable distribution name"),
        # `--verb=-sneaky` rather than `--verb -sneaky`: the separated form never reaches us,
        # because argparse reads the value as an option. The `=` form does, and a verb starting
        # with `-` is the case worth the regex on its own — it would install fine and then be
        # permanently unreachable, since argparse would read it as an option there too.
        ("--verb=-sneaky", "not a usable verb"),
        ("--module=class", "not a usable Python package name"),
    ],
)
def test_unusable_names_are_refused_before_anything_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], argument: str, expected: str
) -> None:
    """Caught here rather than by the user's build backend three commands later."""
    target = tmp_path / "my-verb"
    assert _scaffold(str(target), argument) == 2
    assert expected in capsys.readouterr().err
    assert not target.exists()


def test_a_distribution_may_not_silently_shadow_the_built_in_kind(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The same stance the verb level takes on a shadowed built-in: a hard error naming the
    claimant. Letting an installed package quietly redefine `plugin new cli` would mean the command
    that teaches the contract is the one command whose behaviour you cannot predict."""
    entries = {"cli": make_entry_point("cli", "ECHO", PLUGIN_SCAFFOLD_GROUP)}
    monkeypatch.setattr("astro_mine.cli._new.discover_scaffolds", lambda group: entries)
    assert plugin.run(argparse.Namespace(action="new", rest=["cli", "out"])) == 2
    err = capsys.readouterr().err
    assert "shadows a kind the umbrella owns" in err
    assert "astro_mine.cli.plugin_scaffolds" in err
    assert "Traceback" not in err


def test_the_verb_parses_only_the_action_and_leaves_the_tail_alone() -> None:
    parser = argparse.ArgumentParser(prog="astro-mine plugin")
    plugin.add_arguments(parser)
    args = parser.parse_args(["new", "cli", "./pkg", "--verb", "greet"])
    assert args.action == "new"
    assert args.rest == ["cli", "./pkg", "--verb", "greet"]


def test_the_built_in_kind_declares_the_contract_it_is_checked_against() -> None:
    """It is discovered through the same path a component's scaffold is, so it has to satisfy the
    same contract — a built-in that could skip the check would be a built-in that drifts from it."""
    assert check_subcommand(CLI_PLUGIN_SCAFFOLD, verb="cli").name == "cli"
