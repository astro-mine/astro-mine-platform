"""Worlds' plugin scaffold (`astro-mine plugin new field-model`).

Asserted here, at home, **without importing the umbrella** — `astro-mine-cli` is not a dependency
of this package and must not become one (``conventions.md §1.1``) — so these tests check the same
structural shape the umbrella checks at dispatch.

The three claims that make this scaffold worth having are the three the registry enforces and none
of which an author can guess: the entry-point name is the backend id (selectable from a WorldSpec),
a built-in id may not be shadowed because an id is provenance, and the per-azimuth horizon map
survives — it is the always-present LOS product Link queries, not an implementation detail of the
default backend.
"""

from __future__ import annotations

import argparse
import tomllib
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from astro_mine.worlds import scaffolds
from astro_mine.worlds.illumination._registry import _BUILTINS, known_backends

PLUGIN_SCAFFOLD_GROUP = "astro_mine.cli.plugin_scaffolds"


def _run(output: Path, *argv: str) -> int:
    """Drive the scaffold through the parser the umbrella would build for it — `output` and
    `--force` included, because the umbrella declares those before handing the parser over."""
    parser = argparse.ArgumentParser(prog="astro-mine plugin new field-model")
    parser.add_argument("output")
    parser.add_argument("--force", action="store_true")
    scaffolds.field_model_scaffold.add_arguments(parser)
    return int(scaffolds.field_model_scaffold.run(parser.parse_args([str(output), *argv])))


def _emitted(target: Path, module: str = "demo_illum") -> tuple[dict, str]:
    packaging = tomllib.loads((target / "pyproject.toml").read_text(encoding="utf-8"))
    return packaging, (target / "src" / module / "__init__.py").read_text(encoding="utf-8")


def test_the_scaffold_satisfies_the_structural_contract() -> None:
    for member in ("name", "help", "add_arguments", "run"):
        assert hasattr(scaffolds.field_model_scaffold, member), f"missing {member!r}"
    assert scaffolds.field_model_scaffold.name == "field-model"
    assert scaffolds.field_model_scaffold.help


def test_the_entry_point_resolves_and_matches_its_name() -> None:
    """The entry-point NAME is the kind the user types; a typo is a kind nobody can reach."""
    (ours,) = [
        ep
        for ep in entry_points(group=PLUGIN_SCAFFOLD_GROUP)
        if ep.value.startswith("astro_mine.worlds.scaffolds:")
    ]
    assert ours.name == "field-model"
    assert ours.load().name == "field-model"


def test_the_entry_point_name_is_the_backend_id(tmp_path: Path) -> None:
    """The id is what a WorldSpec's `layers.illumination_backend` selects — which is how a whole
    world bundle picks up a backend, not just a direct call."""
    target = tmp_path / "demo-illum"
    assert _run(target, "--backend", "acme-illum") == 0
    packaging, _ = _emitted(target)
    assert packaging["project"]["entry-points"]["astro_mine.field_models"] == {
        "acme-illum": "demo_illum:build"
    }


@pytest.mark.parametrize("builtin", sorted(_BUILTINS))
def test_a_built_in_id_cannot_be_shadowed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], builtin: str
) -> None:
    """**The claim worth enforcing at scaffold time.** Advertising a built-in id is a hard error at
    resolution, naming both claimants — because an id is provenance: it is folded into
    `illumination_hash` and stamped into the published `field_model` manifest, so an ambiguous id
    would mis-attribute which model produced an illumination product.

    Catching it here means the author is told before they have written a backend against the id,
    rather than at the first resolve. Parameterized over the live built-in set so a new built-in is
    covered without editing this test.
    """
    target = tmp_path / "demo-illum"
    assert _run(target, "--backend", builtin) == 2
    err = capsys.readouterr().err
    assert "cannot be shadowed" in err
    assert "provenance" in err
    assert not target.exists()


def test_the_default_id_is_not_one_the_registry_already_knows() -> None:
    """A default that collided would make the very first `pip install -e .` a hard error the author
    did not cause."""
    assert "demo-illum" not in known_backends()


def test_the_emitted_model_inherits_the_horizon_map(tmp_path: Path) -> None:
    """A finer Sun-visibility backend does not remove the per-azimuth skyline Link queries for
    occlusion — it is the always-present LOS product. Subclassing the shipped model is how the
    template inherits it, so a scaffold that started from scratch would quietly drop it."""
    target = tmp_path / "demo-illum"
    assert _run(target) == 0
    _, source = _emitted(target)
    assert "IlluminationModel" in source
    assert "from astro_mine.worlds.illumination import IlluminationModel" in source


def test_the_emitted_factory_has_the_registrys_signature(tmp_path: Path) -> None:
    """`(terrain, **kwargs) -> SunVisibilityModel`. Extra kwargs flow through unchanged, so
    selecting this backend is a drop-in for the default."""
    target = tmp_path / "demo-illum"
    assert _run(target) == 0
    _, source = _emitted(target)
    assert "def build(terrain: Any, **kwargs: Any)" in source


def test_the_generated_package_does_not_depend_on_the_umbrella(tmp_path: Path) -> None:
    target = tmp_path / "demo-illum"
    assert _run(target) == 0
    packaging, source = _emitted(target)
    assert "astro-mine-cli" not in packaging["project"]["dependencies"]
    assert "import astro_mine.cli" not in source


def test_it_refuses_to_overwrite_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "demo-illum"
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
    target = tmp_path / "demo-illum"
    assert _run(target, "--module", "class") == 2
    assert "not a usable Python package name" in capsys.readouterr().err
    assert not target.exists()
