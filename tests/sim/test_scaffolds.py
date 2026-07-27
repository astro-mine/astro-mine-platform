"""Sim's plugin scaffold (`astro-mine plugin new provider`).

Asserted here, at home, **without importing the umbrella** — `astro-mine-cli` is not a dependency
of this package and must not become one (``conventions.md §1.1``) — so these tests check the same
structural shape the umbrella checks at dispatch.

Two claims carry this scaffold, and both are about failures that do not point at their cause: that
the entry-point name is a Core ``PluginKind`` value rather than a label (an invented one is
discovered and then never matched to a pin, so content resolves to no provider in silence), and
that the emitted package does not import ``astro_mine.sim`` (which would close the very dependency
cycle the rebuild-from-bundle seam exists to prevent).
"""

from __future__ import annotations

import argparse
import importlib
import sys
import tomllib
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from astro_mine.core.registry.enums import PluginKind
from astro_mine.sim import scaffolds

PLUGIN_SCAFFOLD_GROUP = "astro_mine.cli.plugin_scaffolds"


def _run(output: Path, *argv: str) -> int:
    """Drive the scaffold through the parser the umbrella would build for it — `output` and
    `--force` included, because the umbrella declares those before handing the parser over."""
    parser = argparse.ArgumentParser(prog="astro-mine plugin new provider")
    parser.add_argument("output")
    parser.add_argument("--force", action="store_true")
    scaffolds.provider_scaffold.add_arguments(parser)
    return int(scaffolds.provider_scaffold.run(parser.parse_args([str(output), *argv])))


def _emitted(target: Path, module: str = "demo_provider") -> tuple[dict, str]:
    packaging = tomllib.loads((target / "pyproject.toml").read_text(encoding="utf-8"))
    return packaging, (target / "src" / module / "__init__.py").read_text(encoding="utf-8")


def test_the_scaffold_satisfies_the_structural_contract() -> None:
    for member in ("name", "help", "add_arguments", "run"):
        assert hasattr(scaffolds.provider_scaffold, member), f"missing {member!r}"
    assert scaffolds.provider_scaffold.name == "provider"
    assert scaffolds.provider_scaffold.help


def test_the_entry_point_resolves_and_matches_its_name() -> None:
    """The entry-point NAME is the kind the user types; a typo is a kind nobody can reach."""
    (ours,) = [
        ep
        for ep in entry_points(group=PLUGIN_SCAFFOLD_GROUP)
        if ep.value.startswith("astro_mine.sim.scaffolds:")
    ]
    assert ours.name == "provider"
    assert ours.load().name == "provider"


def test_every_offered_kind_is_a_real_core_plugin_kind() -> None:
    """**The trap this scaffold closes.** This is the one group whose entry-point names come from
    Core's closed vocabulary. A factory registered under an invented name is discovered and then
    never matched against a content pin — the provider stays `None` and nothing says why. Pinned
    against Core's enum so the choice list cannot drift away from it."""
    vocabulary = {kind.value for kind in PluginKind}
    assert set(scaffolds.PROVIDER_KINDS) <= vocabulary


def test_the_entry_point_name_is_the_plugin_kind(tmp_path: Path) -> None:
    target = tmp_path / "demo-provider"
    assert _run(target, "--kind", "comms_model") == 0
    packaging, _ = _emitted(target)
    assert packaging["project"]["entry-points"]["astro_mine.providers"] == {
        "comms_model": "demo_provider:from_bundle"
    }


def test_an_invented_kind_is_refused(tmp_path: Path) -> None:
    """Refused by argparse before anything is written, rather than emitted and discovered dead."""
    target = tmp_path / "demo-provider"
    with pytest.raises(SystemExit):
        _run(target, "--kind", "not_a_plugin_kind")
    assert not target.exists()


def test_the_emitted_package_does_not_import_sim(tmp_path: Path) -> None:
    """**The other trap.** Sim reconstructs a provider from pulled bytes *without importing the
    producer*; a producer that imported Sim back would close the cycle the seam exists to prevent.
    A template doing it for one convenience type would teach every producer to."""
    target = tmp_path / "demo-provider"
    assert _run(target) == 0
    packaging, source = _emitted(target)
    # The module *names* Sim in prose — that is the lesson. What it must never do is import it,
    # which is the difference between explaining the rule and breaking it.
    assert "import astro_mine.sim" not in source
    assert "from astro_mine.sim" not in source
    assert "astro-mine-sim" not in packaging["project"]["dependencies"]
    # ...and not the umbrella either: it loads this package, it is not a dependency of it.
    assert "astro-mine-cli" not in packaging["project"]["dependencies"]
    assert "import astro_mine.cli" not in source


def test_the_emitted_factory_answers_the_contract(tmp_path: Path) -> None:
    """`(manifest, layers) -> provider`. A scaffold whose factory does not run is worse than none:
    the author debugs our template before writing a line of their own."""
    target = tmp_path / "demo-provider"
    assert _run(target) == 0
    sys.path.insert(0, str(target / "src"))
    try:
        module = importlib.import_module("demo_provider")
        provider = module.from_bundle("a-manifest", {"application/octet-stream": b"bytes"})
        assert provider.manifest == "a-manifest"
        assert provider.layers == {"application/octet-stream": b"bytes"}
    finally:
        sys.path.remove(str(target / "src"))
        sys.modules.pop("demo_provider", None)


def test_it_refuses_to_overwrite_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "demo-provider"
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
    target = tmp_path / "demo-provider"
    assert _run(target, "--module", "class") == 2
    assert "not a usable Python package name" in capsys.readouterr().err
    assert not target.exists()
