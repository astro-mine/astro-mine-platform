"""Bench's plugin scaffold (`astro-mine plugin new runner`).

Asserted here, at home, **without importing the umbrella** — `astro-mine-cli` is not a dependency
of this package and must not become one (``conventions.md §1.1``) — so these tests check the same
structural shape the umbrella checks at dispatch.

This group is the platform's best worked example of "contribute once, use everywhere": Bench scores
against real physics without importing Sim, so the Sim-backed runner lives in Sim and Bench finds it
by name. The three claims pinned below are the consequences an author cannot guess — that the id is
provenance, that a built-in id can never be reached, and that a runner answers *two* surfaces, the
second of which is the determinism gate.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import tomllib
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from astro_mine.bench import scaffolds
from astro_mine.bench.baseline._registry import _BUILTINS, BenchRunnerProvider

PLUGIN_SCAFFOLD_GROUP = "astro_mine.cli.plugin_scaffolds"


def _run(output: Path, *argv: str) -> int:
    """Drive the scaffold through the parser the umbrella would build for it — `output` and
    `--force` included, because the umbrella declares those before handing the parser over."""
    parser = argparse.ArgumentParser(prog="astro-mine plugin new runner")
    parser.add_argument("output")
    parser.add_argument("--force", action="store_true")
    scaffolds.runner_scaffold.add_arguments(parser)
    return int(scaffolds.runner_scaffold.run(parser.parse_args([str(output), *argv])))


def _emitted(target: Path, module: str = "demo_runner") -> tuple[dict, str]:
    packaging = tomllib.loads((target / "pyproject.toml").read_text(encoding="utf-8"))
    return packaging, (target / "src" / module / "__init__.py").read_text(encoding="utf-8")


def test_the_scaffold_satisfies_the_structural_contract() -> None:
    for member in ("name", "help", "add_arguments", "run"):
        assert hasattr(scaffolds.runner_scaffold, member), f"missing {member!r}"
    assert scaffolds.runner_scaffold.name == "runner"
    assert scaffolds.runner_scaffold.help


def test_the_entry_point_resolves_and_matches_its_name() -> None:
    """The entry-point NAME is the kind the user types; a typo is a kind nobody can reach."""
    (ours,) = [
        ep
        for ep in entry_points(group=PLUGIN_SCAFFOLD_GROUP)
        if ep.value.startswith("astro_mine.bench.scaffolds:")
    ]
    assert ours.name == "runner"
    assert ours.load().name == "runner"


def test_the_entry_point_name_is_the_runner_id(tmp_path: Path) -> None:
    """The id is what `--runner` selects, what the scorecard is stamped with, and what its content
    hash folds in — so a third-party run is distinguishable by provenance, not only by numbers."""
    target = tmp_path / "demo-runner"
    assert _run(target, "--runner", "acme") == 0
    packaging, _ = _emitted(target)
    assert packaging["project"]["entry-points"]["astro_mine.bench.runners"] == {
        "acme": "demo_runner:acme_runner_provider"
    }


@pytest.mark.parametrize("builtin", sorted(_BUILTINS))
def test_a_built_in_id_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], builtin: str
) -> None:
    """A built-in always wins resolution, so a plugin claiming its id would install, register, and
    never be reached — the worst failure shape there is, because everything looks fine. Caught at
    scaffold time instead. Parameterized over the live built-in set so a new one is covered."""
    target = tmp_path / "demo-runner"
    assert _run(target, "--runner", builtin) == 2
    assert "built-in runner id" in capsys.readouterr().err
    assert not target.exists()


def test_the_emitted_provider_satisfies_the_runner_contract(tmp_path: Path) -> None:
    """**Both surfaces, not one.** `episode_runner` is the scoring path and `harness_runner` is the
    determinism gate; a provider with only the first installs and registers cleanly and then fails
    at the gate — the run that matters most. Checked against Bench's own runtime-checkable
    Protocol, which is the same check resolution performs."""
    target = tmp_path / "demo-runner"
    assert _run(target) == 0
    sys.path.insert(0, str(target / "src"))
    try:
        module = importlib.import_module("demo_runner")
        provider = module.demo_runner_provider
        assert isinstance(provider, BenchRunnerProvider)
        assert provider.runner_id == "demo"
        assert provider.episode_runner() is not None
        assert provider.harness_runner() is not None
    finally:
        sys.path.remove(str(target / "src"))
        sys.modules.pop("demo_runner", None)


def test_the_store_argument_stays_untyped(tmp_path: Path) -> None:
    """Bench types `store` as `object` precisely so it never names a Sim type. A template that
    narrowed it would put an engine import in a package Bench discovers, dragging that engine onto
    every machine that installed the plugin."""
    target = tmp_path / "demo-runner"
    assert _run(target) == 0
    _, source = _emitted(target)
    assert "store: object | None = None" in source
    assert "import astro_mine.sim" not in source


def test_the_generated_package_does_not_depend_on_the_umbrella(tmp_path: Path) -> None:
    target = tmp_path / "demo-runner"
    assert _run(target) == 0
    packaging, source = _emitted(target)
    assert "astro-mine-cli" not in packaging["project"]["dependencies"]
    assert "import astro_mine.cli" not in source


def test_an_unusable_runner_id_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The id is typed after `--runner` on a command line, so it has to look like one."""
    target = tmp_path / "demo-runner"
    assert _run(target, "--runner", "Not A Runner") == 2
    assert "not a usable runner id" in capsys.readouterr().err
    assert not target.exists()


def test_it_refuses_to_overwrite_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "demo-runner"
    assert _run(target) == 0
    (target / "pyproject.toml").write_text("# hand-edited\n", encoding="utf-8")
    assert _run(target) == 1
    assert "file exists" in capsys.readouterr().err
    assert (target / "pyproject.toml").read_text(encoding="utf-8") == "# hand-edited\n"
    assert _run(target, "--force") == 0
    assert "# hand-edited" not in (target / "pyproject.toml").read_text(encoding="utf-8")


def test_an_unusable_module_name_is_refused_before_anything_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "demo-runner"
    assert _run(target, "--module", "class") == 2
    assert "not a usable Python package name" in capsys.readouterr().err
    assert not target.exists()
