"""Learn's plugin scaffolds (`astro-mine plugin new algorithm | curriculum`).

Asserted here, at home, **without importing the umbrella** — `astro-mine-cli` is not a dependency
of this package and must not become one (``conventions.md §1.1``) — so these tests check the same
structural shape the umbrella checks at dispatch.

**The claim worth testing is not "it registers" but "it registers under the name the author
asked for."** Both of Learn's groups name a plugin by something other than its entry-point name —
an algorithm by ``spec.capability_tag``, a curriculum by ``spec.name`` when the callable returns a
spec — and getting that wrong is silent: the package installs, the registry loads it, and the id
the author put in their config resolves to nothing. So each emitted package is imported for real
and pushed through Learn's own registry, and the *key it lands under* is what is asserted.
"""

from __future__ import annotations

import argparse
import importlib
import sys
import tomllib
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from astro_mine.learn import scaffolds

PLUGIN_SCAFFOLD_GROUP = "astro_mine.cli.plugin_scaffolds"


def _run(scaffold: object, output: Path, *argv: str) -> int:
    """Drive a scaffold through the parser the umbrella would build for it.

    ``output`` and ``--force`` are declared here because the umbrella declares them before handing
    the parser over — reproducing that is what makes this a test of the real surface.
    """
    parser = argparse.ArgumentParser(prog="astro-mine plugin new")
    parser.add_argument("output")
    parser.add_argument("--force", action="store_true")
    scaffold.add_arguments(parser)  # type: ignore[attr-defined]
    return int(scaffold.run(parser.parse_args([str(output), *argv])))  # type: ignore[attr-defined]


def _emitted(target: Path, module: str) -> tuple[dict, str]:
    packaging = tomllib.loads((target / "pyproject.toml").read_text(encoding="utf-8"))
    return packaging, (target / "src" / module / "__init__.py").read_text(encoding="utf-8")


def _import_emitted(target: Path, module: str) -> object:
    """Import the generated package the way its own wheel layout would place it."""
    sys.path.insert(0, str(target / "src"))
    try:
        return importlib.import_module(module)
    finally:
        sys.path.remove(str(target / "src"))
        sys.modules.pop(module, None)


@pytest.mark.parametrize(
    ("scaffold", "kind"),
    [
        (scaffolds.algorithm_scaffold, "algorithm"),
        (scaffolds.curriculum_scaffold, "curriculum"),
    ],
)
def test_the_scaffolds_satisfy_the_structural_contract(scaffold: object, kind: str) -> None:
    """The same four members a verb has: RFC-0011 §7's groups bind to the same contract."""
    for member in ("name", "help", "add_arguments", "run"):
        assert hasattr(scaffold, member), f"missing {member!r}"
    assert scaffold.name == kind  # type: ignore[attr-defined]
    assert scaffold.help  # type: ignore[attr-defined]


@pytest.mark.parametrize("kind", ["algorithm", "curriculum"])
def test_the_entry_points_resolve_and_match_their_names(kind: str) -> None:
    """The entry-point NAME is the kind the user types; a typo here is a kind nobody can reach."""
    ours = {
        ep.name: ep
        for ep in entry_points(group=PLUGIN_SCAFFOLD_GROUP)
        if ep.value.startswith("astro_mine.learn.scaffolds:")
    }
    assert kind in ours
    assert ours[kind].load().name == kind


# --------------------------------------------------------------------------- algorithm


def test_an_algorithm_registers_under_its_capability_tag_not_its_entry_point(
    tmp_path: Path,
) -> None:
    """**The trap this scaffold exists to close.**

    Learn's registry keys an algorithm by `spec.capability_tag`, read off the loaded object, and
    ignores the entry-point name entirely. An author who assumed otherwise gets a package that
    installs, loads, and registers under a name they did not choose — so the id in their
    TrainConfig resolves to nothing, far from the cause.
    """
    target = tmp_path / "demo-algo"
    assert _run(scaffolds.algorithm_scaffold, target, "--tag", "marl.acme.custom") == 0
    module = _import_emitted(target, "demo_algo")

    algorithm = module.build()  # type: ignore[attr-defined]
    assert algorithm.spec.capability_tag == "marl.acme.custom"


def test_an_algorithm_does_not_collide_with_a_shipped_baseline(tmp_path: Path) -> None:
    """The default tag has to be one no built-in already claims, or the very first
    `pip install -e .` produces a registry collision the author did not cause."""
    from astro_mine.learn.algos.registry import AlgorithmRegistry

    target = tmp_path / "demo-algo"
    assert _run(scaffolds.algorithm_scaffold, target) == 0
    module = _import_emitted(target, "demo_algo")

    builtin_tags = {spec.capability_tag for spec in AlgorithmRegistry().specs()}
    assert module.build().spec.capability_tag not in builtin_tags  # type: ignore[attr-defined]


def test_the_algorithm_spec_declares_the_paradigm_asked_for(tmp_path: Path) -> None:
    """`ctde` builds a centralized critic and exposes it on the trainer; `decentralized` does not.
    Declaring the wrong one produces a trainer that silently lacks what the algorithm needs."""
    target = tmp_path / "demo-algo"
    assert _run(scaffolds.algorithm_scaffold, target, "--paradigm", "ctde") == 0
    module = _import_emitted(target, "demo_algo")
    spec = module.build().spec  # type: ignore[attr-defined]
    assert spec.paradigm == "ctde"
    assert spec.is_ctde


def test_the_algorithm_entry_point_points_at_what_was_written(tmp_path: Path) -> None:
    target = tmp_path / "demo-algo"
    assert _run(scaffolds.algorithm_scaffold, target) == 0
    packaging, _ = _emitted(target, "demo_algo")
    assert packaging["project"]["entry-points"]["astro_mine.learn.algorithms"] == {
        "marl_demo_algorithm": "demo_algo:build"
    }


# --------------------------------------------------------------------------- curriculum


def test_a_curriculum_registers_under_its_spec_name_not_its_entry_point(tmp_path: Path) -> None:
    """The second half of the same trap, with a twist: which name wins depends on what the callable
    *returns*. This template returns a spec, so `spec.name` is the key and the entry-point name is
    ignored — a factory would be the other way round."""
    target = tmp_path / "demo-curriculum"
    assert _run(scaffolds.curriculum_scaffold, target, "--name-it", "acme_ladder") == 0
    module = _import_emitted(target, "demo_curriculum")

    spec = module.build()  # type: ignore[attr-defined]
    assert spec.name == "acme_ladder"


def test_a_scaffolded_curriculum_is_a_valid_ladder(tmp_path: Path) -> None:
    """A curriculum with no stages, or with a stage that can never promote, validates as a document
    and then never advances. The emitted ladder has to be a real one."""
    target = tmp_path / "demo-curriculum"
    assert _run(scaffolds.curriculum_scaffold, target) == 0
    spec = _import_emitted(target, "demo_curriculum").build()  # type: ignore[attr-defined]
    assert len(spec.stages) >= 2
    assert [stage.name for stage in spec.stages] == ["clear", "degraded"]
    # Every stage promotes on sustained progress rather than one lucky iteration.
    assert all(stage.advance.patience >= 2 for stage in spec.stages)
    # It round-trips through its own schema — its content hash is part of a run's repro key.
    assert type(spec).model_validate_json(spec.model_dump_json()) == spec


def test_the_curriculum_entry_point_and_spec_name_agree(tmp_path: Path) -> None:
    """Kept equal on purpose, so the spec-vs-factory split cannot bite before it is meant to."""
    target = tmp_path / "demo-curriculum"
    assert _run(scaffolds.curriculum_scaffold, target, "--name-it", "acme_ladder") == 0
    packaging, _ = _emitted(target, "demo_curriculum")
    assert packaging["project"]["entry-points"]["astro_mine.learn.curricula"] == {
        "acme_ladder": "demo_curriculum:build"
    }


# --------------------------------------------------------------------------- shared


@pytest.mark.parametrize(
    ("scaffold", "module"),
    [
        (scaffolds.algorithm_scaffold, "demo_algo"),
        (scaffolds.curriculum_scaffold, "demo_curriculum"),
    ],
)
def test_the_generated_packages_do_not_depend_on_the_umbrella(
    tmp_path: Path, scaffold: object, module: str
) -> None:
    """The layering rule, taught by example: the umbrella loads a plugin; it is not a dependency
    of one. A scaffold that added it would teach every third-party author to invert it."""
    target = tmp_path / module.replace("_", "-")
    assert _run(scaffold, target) == 0
    packaging, source = _emitted(target, module)
    assert "astro-mine-cli" not in packaging["project"]["dependencies"]
    assert "astro_mine.cli" not in source


def test_an_unusable_module_name_is_refused_before_anything_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "demo-algo"
    assert _run(scaffolds.algorithm_scaffold, target, "--module", "class") == 2
    assert "not a usable Python package name" in capsys.readouterr().err
    assert not target.exists()


def test_it_refuses_to_overwrite_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "demo-algo"
    assert _run(scaffolds.algorithm_scaffold, target) == 0
    (target / "pyproject.toml").write_text("# hand-edited\n", encoding="utf-8")
    assert _run(scaffolds.algorithm_scaffold, target) == 1
    assert "file exists" in capsys.readouterr().err
    assert (target / "pyproject.toml").read_text(encoding="utf-8") == "# hand-edited\n"
    assert _run(scaffolds.algorithm_scaffold, target, "--force") == 0
    assert "# hand-edited" not in (target / "pyproject.toml").read_text(encoding="utf-8")
