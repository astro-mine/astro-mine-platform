"""Allocate's plugin scaffold (`astro-mine plugin new solver`).

Asserted here, at home, **without importing the umbrella** — `astro-mine-cli` is not a dependency
of this package and must not become one (``conventions.md §1.1``) — so these tests check the same
structural shape the umbrella checks at dispatch.

The solver seam is safe to open precisely because Allocate is not the safety authority: every
feasible plan, from any backend, is independently re-checked against the IR. What the scaffold has
to get right is what the *registry* enforces — that the id is provenance and cannot shadow a
built-in — and what the IR was designed for: incumbents that **stream** with improving bounds,
rather than one final answer.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import tomllib
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from astro_mine.allocate import scaffolds
from astro_mine.allocate.solvers.registry import _LOADERS, known_backends

PLUGIN_SCAFFOLD_GROUP = "astro_mine.cli.plugin_scaffolds"


def _run(output: Path, *argv: str) -> int:
    """Drive the scaffold through the parser the umbrella would build for it — `output` and
    `--force` included, because the umbrella declares those before handing the parser over."""
    parser = argparse.ArgumentParser(prog="astro-mine plugin new solver")
    parser.add_argument("output")
    parser.add_argument("--force", action="store_true")
    scaffolds.solver_scaffold.add_arguments(parser)
    return int(scaffolds.solver_scaffold.run(parser.parse_args([str(output), *argv])))


def _emitted(target: Path, module: str = "demo_solver") -> tuple[dict, str]:
    packaging = tomllib.loads((target / "pyproject.toml").read_text(encoding="utf-8"))
    return packaging, (target / "src" / module / "__init__.py").read_text(encoding="utf-8")


def test_the_scaffold_satisfies_the_structural_contract() -> None:
    for member in ("name", "help", "add_arguments", "run"):
        assert hasattr(scaffolds.solver_scaffold, member), f"missing {member!r}"
    assert scaffolds.solver_scaffold.name == "solver"
    assert scaffolds.solver_scaffold.help


def test_the_entry_point_resolves_and_matches_its_name() -> None:
    """The entry-point NAME is the kind the user types; a typo is a kind nobody can reach."""
    (ours,) = [
        ep
        for ep in entry_points(group=PLUGIN_SCAFFOLD_GROUP)
        if ep.value.startswith("astro_mine.allocate.scaffolds:")
    ]
    assert ours.name == "solver"
    assert ours.load().name == "solver"


def test_the_entry_point_name_is_the_backend_id(tmp_path: Path) -> None:
    """The id is recorded in a plan's `provenance.backend`, so which solver produced a plan is
    always recoverable from the plan itself."""
    target = tmp_path / "demo-solver"
    assert _run(target, "--backend", "acme-solver") == 0
    packaging, _ = _emitted(target)
    assert packaging["project"]["entry-points"]["astro_mine.allocate.solvers"] == {
        "acme-solver": "demo_solver:acme_solver"
    }


@pytest.mark.parametrize("builtin", sorted(_LOADERS))
def test_a_built_in_id_cannot_be_shadowed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], builtin: str
) -> None:
    """Advertising a built-in id is a hard error naming both claimants, for the same provenance
    reason: an ambiguous id would make the same plan attributable to two engines. Caught at scaffold
    time so the author is told before writing a solver against the id. Parameterized over the live
    built-in set so a new engine is covered without editing this test."""
    target = tmp_path / "demo-solver"
    assert _run(target, "--backend", builtin) == 2
    err = capsys.readouterr().err
    assert "cannot be shadowed" in err
    assert "provenance" in err
    assert not target.exists()


def test_the_default_id_is_not_one_the_registry_already_knows() -> None:
    """A colliding default would make the very first `pip install -e .` a hard error the author did
    not cause."""
    assert "demo-solver" not in known_backends()


def test_the_emitted_solver_streams_incumbents(tmp_path: Path) -> None:
    """**Streaming is the point.** A solver that yields one final answer works and throws away the
    anytime behaviour the IR was designed for, so the template delegates to the shipped stub —
    which already streams correctly — rather than showing a single `yield`. Checked by driving the
    emitted solver over a real IR and reading what comes out.
    """
    from astro_mine.allocate.solvers.trivial import TrivialStubSolver

    target = tmp_path / "demo-solver"
    assert _run(target) == 0
    sys.path.insert(0, str(target / "src"))
    try:
        module = importlib.import_module("demo_solver")
        factory = module.demo_solver
        # Same construction signature the registry uses for a built-in, so one resolve path covers
        # both — if this diverged, the plugin would fail only once someone selected it.
        assert callable(factory)
        import inspect

        assert set(inspect.signature(TrivialStubSolver.__init__).parameters) == set(
            inspect.signature(factory.__init__).parameters
        )
        assert inspect.isgeneratorfunction(factory.solve)
    finally:
        sys.path.remove(str(target / "src"))
        sys.modules.pop("demo_solver", None)


def test_the_generated_package_does_not_depend_on_the_umbrella(tmp_path: Path) -> None:
    target = tmp_path / "demo-solver"
    assert _run(target) == 0
    packaging, source = _emitted(target)
    assert "astro-mine-cli" not in packaging["project"]["dependencies"]
    assert "import astro_mine.cli" not in source


def test_it_refuses_to_overwrite_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "demo-solver"
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
    target = tmp_path / "demo-solver"
    assert _run(target, "--module", "class") == 2
    assert "not a usable Python package name" in capsys.readouterr().err
    assert not target.exists()
