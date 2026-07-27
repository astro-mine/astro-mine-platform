"""Mind's scaffolds for the umbrella CLI (`astro-mine new stack`, `plugin new tier`).

Asserted here, at home, **without importing the umbrella** — `astro-mine-cli` is not a dependency
of this package and must not become one (``conventions.md §1.1``) — so these tests check the same
structural shape the umbrella checks at dispatch.

This is where **the acceptance criterion for RFC-0011 §7 actually lives** for Mind's two kinds: *a
scaffolded document must validate with no hand-editing*. No test in `astro-mine-cli` can check it —
that package installs no components by design — so it is checked by the package that owns both the
template and the checker.

For a stack spec the bar is higher than schema validity. `astro-mine-mind validate` also asks the
registry whether every plugin the stack binds is discoverable, which is the error users actually
hit; a scaffold that emitted a schema-valid spec naming plugins nobody registers would be a worked
example of the mistake it exists to prevent.
"""

from __future__ import annotations

import argparse
import tomllib
from importlib.metadata import entry_points
from pathlib import Path

import pytest

from astro_mine.mind import scaffolds
from astro_mine.mind.cli import main
from astro_mine.mind.compose.composer import compose
from astro_mine.mind.spec.loader import load_stack_spec

SCAFFOLD_GROUP = "astro_mine.cli.scaffolds"
PLUGIN_SCAFFOLD_GROUP = "astro_mine.cli.plugin_scaffolds"


def _run(scaffold: object, output: Path, *argv: str) -> int:
    """Drive a scaffold through the parser the umbrella would build for it.

    ``output`` and ``--force`` are declared here because the umbrella declares them before handing
    the parser over — reproducing that is what makes this a test of the real surface rather than of
    a namespace we invented.
    """
    parser = argparse.ArgumentParser(prog="astro-mine new")
    parser.add_argument("output")
    parser.add_argument("--force", action="store_true")
    scaffold.add_arguments(parser)  # type: ignore[attr-defined]
    return int(scaffold.run(parser.parse_args([str(output), *argv])))  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("scaffold", "kind", "group"),
    [
        (scaffolds.stack_scaffold, "stack", SCAFFOLD_GROUP),
        (scaffolds.tier_scaffold, "tier", PLUGIN_SCAFFOLD_GROUP),
    ],
)
def test_the_scaffolds_satisfy_the_structural_contract(
    scaffold: object, kind: str, group: str
) -> None:
    """The same four members a verb has: RFC-0011 §7's groups bind to the same contract, so a
    component writes the object it already knows how to write."""
    for member in ("name", "help", "add_arguments", "run"):
        assert hasattr(scaffold, member), f"missing {member!r}"
    assert scaffold.name == kind  # type: ignore[attr-defined]
    assert scaffold.help  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("group", "kind"), [(SCAFFOLD_GROUP, "stack"), (PLUGIN_SCAFFOLD_GROUP, "tier")]
)
def test_the_entry_points_resolve_and_match_their_names(group: str, kind: str) -> None:
    """The entry-point NAME is the kind the user types, so a typo here is a kind nobody can reach —
    invisible until someone runs the command, unless this runs."""
    (ours,) = [
        ep for ep in entry_points(group=group) if ep.value.startswith("astro_mine.mind.scaffolds:")
    ]
    assert ours.name == kind
    assert ours.load().name == kind


# --------------------------------------------------------------------------- `new stack`


def test_a_scaffolded_stack_passes_minds_full_checker(tmp_path: Path) -> None:
    """**The acceptance criterion**, at the bar that matters.

    Not just schema-valid: `astro-mine-mind validate` also checks that every plugin the stack binds
    is discoverable. Run through Mind's own CLI, which is the same code path `astro-mine validate`
    routes a stack spec to.
    """
    out = tmp_path / "stack.yaml"
    assert _run(scaffolds.stack_scaffold, out) == 0
    assert main(["validate", str(out)]) == 0


def test_a_scaffolded_stack_composes(tmp_path: Path) -> None:
    """Validation says the bindings resolve; composition proves they actually build. A spec that
    validates and then fails to compose would still have cost the user their first hour."""
    out = tmp_path / "stack.yaml"
    assert _run(scaffolds.stack_scaffold, out) == 0
    from astro_mine.mind.registry.registry import TierRegistry

    spec = load_stack_spec(out.read_text(encoding="utf-8"))
    stack = compose(spec, TierRegistry.from_entry_points())
    assert stack is not None


def test_it_binds_only_plugins_a_bare_install_registers(tmp_path: Path) -> None:
    """The constraint behind the template. Binding `guard.shield` would read better and would fail
    for anyone who has not also installed Guard — a scaffold whose output does not validate until
    you install something else is worse than no scaffold."""
    out = tmp_path / "stack.yaml"
    _run(scaffolds.stack_scaffold, out)
    text = out.read_text(encoding="utf-8")
    bound = {line.split("plugin:", 1)[1].strip() for line in text.splitlines() if "plugin:" in line}
    registered = {ep.name for ep in entry_points(group="astro_mine.mind.tier_plugins")}
    assert bound <= registered, f"scaffold binds unregistered plugin(s): {bound - registered}"


def test_the_identity_is_the_users_to_choose(tmp_path: Path) -> None:
    out = tmp_path / "stack.yaml"
    assert _run(scaffolds.stack_scaffold, out, "--id", "acme-stack", "--scenario-ref", "s") == 0
    document = load_stack_spec(out.read_text(encoding="utf-8"))
    assert document.stack_spec.id == "acme-stack"
    assert document.stack_spec.scenario_ref == "s"


def test_it_refuses_to_overwrite_without_force(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "stack.yaml"
    assert _run(scaffolds.stack_scaffold, out) == 0
    out.write_text("# hand-edited\n", encoding="utf-8")
    assert _run(scaffolds.stack_scaffold, out) == 1
    assert "file exists" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "# hand-edited\n"
    assert _run(scaffolds.stack_scaffold, out, "--force") == 0
    assert "# hand-edited" not in out.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- `plugin new tier`


def _emitted(target: Path, module: str) -> tuple[dict, str, str]:
    manifest = tomllib.loads((target / "pyproject.toml").read_text(encoding="utf-8"))
    package = target / "src" / module
    return (
        manifest,
        (package / "__init__.py").read_text(encoding="utf-8"),
        (package / "manifest.yaml").read_text(encoding="utf-8"),
    )


def test_the_tier_scaffold_registers_into_minds_group(tmp_path: Path) -> None:
    """The entry point is the whole contract — if it is not in the metadata, nothing else is."""
    target = tmp_path / "demo-tier"
    assert _run(scaffolds.tier_scaffold, target) == 0
    packaging, _, _ = _emitted(target, "demo_tier")
    assert packaging["project"]["entry-points"]["astro_mine.mind.tier_plugins"] == {
        "demo.control": "demo_tier:demo_control_plugin"
    }


def test_the_emitted_manifest_is_a_valid_core_plugin_manifest(tmp_path: Path) -> None:
    """The manifest is the plugin's public face and the thing Mind's registry gates on. A scaffold
    that emitted one Core rejects would fail at `pip install -e .`, after the author had already
    written their tier against it."""
    from astro_mine.core.registry import load_manifest

    target = tmp_path / "demo-tier"
    assert _run(scaffolds.tier_scaffold, target) == 0
    _, _, manifest_yaml = _emitted(target, "demo_tier")
    manifest = load_manifest(manifest_yaml).manifest
    assert manifest.name == "demo.control"
    # `policy` even for an allocator or a shield — the requirement authors get wrong, which is why
    # the scaffold hard-codes it and this pins it.
    assert manifest.kind == "policy"
    assert manifest.attributes["tier"] == "control"


def test_the_emitted_module_is_valid_python_and_provides_the_contract(tmp_path: Path) -> None:
    """A scaffold whose output does not run is worse than none: the author debugs *our* template
    before writing a line of their own. So the emitted module is compiled and executed, and the
    provider the entry point names is called and its result checked."""
    from astro_mine.mind.registry import TierPlugin

    target = tmp_path / "demo-tier"
    assert _run(scaffolds.tier_scaffold, target) == 0
    _, module_source, _ = _emitted(target, "demo_tier")

    # The provider reads its manifest as package data, so the emitted package has to be importable
    # rather than merely exec'd — install it on sys.path the way its own wheel layout would.
    import sys

    sys.path.insert(0, str(target / "src"))
    try:
        import importlib

        module = importlib.import_module("demo_tier")
        plugin = module.demo_control_plugin()
        assert isinstance(plugin, TierPlugin)
        assert plugin.manifest.name == "demo.control"
        tier = plugin.factory({})
        assert callable(tier.decide)
    finally:
        sys.path.remove(str(target / "src"))
        sys.modules.pop("demo_tier", None)
    assert "import astro_mine.cli" not in module_source


def test_every_tier_role_scaffolds_a_manifest_core_accepts(tmp_path: Path) -> None:
    """`shield` and `allocator` are tiers too, and their manifest kind is still `policy`. Pinned
    across all five because that is the rule an author is most likely to get wrong."""
    from astro_mine.core.registry import load_manifest

    for tier in scaffolds.tier_scaffold.TIERS:
        target = tmp_path / tier
        assert _run(scaffolds.tier_scaffold, target, "--tier", tier) == 0
        _, _, manifest_yaml = _emitted(target, tier)
        manifest = load_manifest(manifest_yaml).manifest
        assert manifest.kind == "policy"
        assert manifest.attributes["tier"] == tier


def test_the_generated_package_does_not_depend_on_the_umbrella(tmp_path: Path) -> None:
    """The layering rule, taught by example. A scaffold that added `astro-mine-cli` to the emitted
    dependencies would teach every third-party author to invert it."""
    target = tmp_path / "demo-tier"
    assert _run(scaffolds.tier_scaffold, target) == 0
    packaging, module_source, _ = _emitted(target, "demo_tier")
    assert "astro-mine-cli" not in packaging["project"]["dependencies"]
    assert "astro_mine.cli" not in module_source


def test_an_unusable_module_name_is_refused_before_anything_is_written(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "demo-tier"
    assert _run(scaffolds.tier_scaffold, target, "--module", "class") == 2
    assert "not a usable Python package name" in capsys.readouterr().err
    assert not target.exists()
