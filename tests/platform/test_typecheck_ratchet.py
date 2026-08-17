"""The type-check ratchet is a set, and a set can rot in two directions (platform#32).

``conventions.md §2`` requires Python to be "type-hinted, checked with ``mypy``/``pyright``". For
the whole life of this repository nothing checked it: ``mypy`` sat in the ``dev`` dependency group
with no ``[tool.mypy]`` section, no ``mypy.ini``, no pre-commit hook and no workflow step. 110,888
lines across 746 files, in the one distribution where a wrong signature is everyone's problem
because no package boundary contains it.

A whole-tree gate is a flag day nobody takes. The adoptable form is a **ratchet**: enrol the
components that already pass, fail the build on a regression in any of them, and widen the set
component by component. That shape has a failure mode a passing CI run cannot show you — the
exemption list quietly growing until the gate covers nothing. So the list is pinned here, and
moving it in either direction means editing this file.

Four properties, and why each is worth a test rather than a comment:

* **The exemption list is exactly what we think it is.** Pinned below. Adding a component to
  ``pyproject.toml``'s exemptions fails here until someone writes it down; enrolling one fails here
  too, which is the same rule read forwards — either way the change is deliberate and visible,
  never a side effect of making CI green. This is the §11 lateral-edge pattern applied to types.
* **Every component is accounted for.** A component added to the tree is enrolled by default, and
  an exemption naming a component that no longer exists is caught rather than left to rot.
* **Core is never exempt.** Core is the narrow waist; every other component's types are stated in
  terms of it, so an exemption there would hollow out the gate for the whole tree while leaving
  twelve components nominally enrolled.
* **We never blind ourselves to our own types.** Exemptions use ``ignore_errors``, which suppresses
  findings *inside* a module while its types stay visible to callers. ``follow_imports = "skip"``
  would turn the module into ``Any`` and silently drop coverage from every consumer — the right
  tool for a third-party package that lies about its own types (``mcap``, ``pyarrow``), never for
  ours.

And then the half that matters most: the checks are **fired**, not just read.
``test_the_gate_fires`` proves the committed configuration rejects an error in an enrolled
component and accepts the same error in an exempt one — running the real ``[tool.mypy]`` table,
over a synthetic tree, so the discrimination the ratchet claims is demonstrated rather than
asserted. Without it, a configuration that had quietly stopped being strict would read as a
passing suite forever.
"""

from __future__ import annotations

import pkgutil
import subprocess
import sys
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "astro_mine"
PYPROJECT = REPO_ROOT / "pyproject.toml"

#: The components not yet enrolled, with the error count each carried when the gate was seeded.
#: These are debts with a number attached, not exemptions — the list may shrink, and every entry
#: removed from it is the point of the exercise. It must never grow silently, which is what pinning
#: it here buys. Keep it in step with the ``[[tool.mypy.overrides]]`` blocks in ``pyproject.toml``;
#: the counts live there, next to the reason each component fails.
NOT_YET_ENROLLED = frozenset({"surrogate", "learn"})

#: A deliberate violation of ``disallow_untyped_defs``, which ``strict = true`` turns on. Chosen
#: over an obviously-wrong assignment because it is the error a migrated tree actually produces in
#: bulk, and because it needs no imports — the synthetic tree below stands alone, so this test is
#: about the configuration and never about whether the real package imports cleanly.
_SYNTHETIC_VIOLATION = "def undertyped(value):\n    return value\n"


def _components() -> frozenset[str]:
    """The component subpackages of *this* distribution's tree.

    Read from the directory rather than from ``astro_mine.__path__``: the namespace package spans
    every installed distribution contributing to it, and a development environment normally has
    ``astro-mine-cli`` installed editable alongside — which would put ``cli`` in the answer and have
    this test demand a mypy exemption for a package that is not in this repository.
    """
    return frozenset(
        module.name for module in pkgutil.iter_modules([str(PACKAGE_ROOT)]) if module.ispkg
    )


def _mypy_config() -> dict[str, object]:
    with PYPROJECT.open("rb") as handle:
        return tomllib.load(handle)["tool"]["mypy"]


def _overrides() -> list[dict[str, object]]:
    config = _mypy_config()
    overrides = config.get("overrides", [])
    assert isinstance(overrides, list)
    return overrides


def _modules(override: dict[str, object]) -> list[str]:
    modules = override["module"]
    return [modules] if isinstance(modules, str) else list(modules)  # type: ignore[arg-type]


def _exempt_components() -> frozenset[str]:
    """Components an ``ignore_errors`` override takes out of the gate.

    Matches ``astro_mine.<component>.*`` only. The generated-bindings exemption
    (``astro_mine.*._proto.*``) is deliberately not a component exemption — it names a kind of file,
    not a unit of ownership, and every component that ships protobuf keeps its hand-written code
    checked.
    """
    exempt = set()
    for override in _overrides():
        if not override.get("ignore_errors"):
            continue
        for module in _modules(override):
            parts = module.split(".")
            if len(parts) == 3 and parts[0] == "astro_mine" and parts[2] == "*":
                exempt.add(parts[1])
    return frozenset(exempt)


def test_the_gate_is_configured_and_strict() -> None:
    """Without this, everything below would pass against a configuration that checks nothing."""
    config = _mypy_config()
    assert config.get("strict") is True, (
        "[tool.mypy] must set strict = true. The ratchet's promise is that an enrolled component "
        "is held to the same bar as astro-mine-cli; relaxing the global table would move the bar "
        "for thirteen components at once, silently."
    )


def test_the_exemption_list_has_not_moved() -> None:
    """The ratchet may only be turned by hand."""
    assert _exempt_components() == NOT_YET_ENROLLED, (
        "The set of components exempt from the mypy gate in pyproject.toml no longer matches the "
        "set pinned in this test. If you enrolled a component, remove it from NOT_YET_ENROLLED "
        "here — that is the ratchet turning, and it should be visible in the diff. If you exempted "
        "one, say why in the pyproject.toml override and add it here, so the next reader can see "
        "the gate got narrower and when."
    )


def test_every_component_is_either_enrolled_or_named() -> None:
    """No component may be exempt by omission, and no exemption may outlive its component."""
    components = _components()
    stale = _exempt_components() - components
    assert not stale, (
        f"pyproject.toml exempts {sorted(stale)} from the mypy gate, but no such component exists "
        "under src/astro_mine. A renamed or removed component left its exemption behind."
    )
    # The converse needs no assertion — a component absent from the exemption list is enrolled, and
    # `uv run mypy src/astro_mine` in CI is what proves it passes. What this file guarantees is
    # that being enrolled is the *default*: a new component cannot arrive unchecked without an
    # entry in pyproject.toml, which `test_the_exemption_list_has_not_moved` would then catch.
    assert components - _exempt_components(), "every component is exempt; the gate checks nothing"


def test_core_is_never_exempt() -> None:
    """Core is the narrow waist (§1). Exempting it would hollow out the gate everywhere."""
    assert "core" not in _exempt_components()
    assert "core" in _components()


def test_our_own_modules_are_never_skipped() -> None:
    """``follow_imports = "skip"`` is for dependencies that lie about their types, not for us.

    Applied to an ``astro_mine`` module it would make that module ``Any`` for every consumer — the
    exemption would leak out of the component that took it and quietly weaken twelve others.
    """
    for override in _overrides():
        if override.get("follow_imports") != "skip":
            continue
        offenders = [module for module in _modules(override) if module.startswith("astro_mine")]
        assert not offenders, (
            f"{offenders} are skipped rather than checked. Use ignore_errors, which suppresses "
            "findings inside the module while keeping its types visible to callers."
        )


def _synthetic_tree(root: Path, components: list[str]) -> None:
    """A namespace tree shaped like ``src/astro_mine``, holding nothing but the violation.

    Namespace packages, with no ``__init__.py``, exactly as the real tree is laid out — the module
    names are what the per-module overrides match on, so they have to come out identical.
    """
    for component in components:
        package = root / "astro_mine" / component
        package.mkdir(parents=True)
        (package / "_probe.py").write_text(_SYNTHETIC_VIOLATION, encoding="utf-8")


def _config_against(root: Path, destination: Path) -> Path:
    """The committed ``[tool.mypy]`` table, retargeted at the synthetic tree.

    Copied rather than re-declared: strictness, the per-module overrides and the exemption list all
    come from ``pyproject.toml``, so this exercises the configuration that actually gates CI. The
    single edit is ``mypy_path``, which has to point at the synthetic tree for the probe modules to
    resolve under their real names.
    """
    config = dict(_mypy_config())
    overrides = config.pop("overrides", [])

    def render(value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, list):
            return "[" + ", ".join(render(item) for item in value) + "]"
        return f'"{value}"'

    config["mypy_path"] = str(root)
    lines = ["[tool.mypy]"]
    lines += [f"{key} = {render(value)}" for key, value in config.items()]
    for override in overrides:
        lines.append("\n[[tool.mypy.overrides]]")
        lines += [f"{key} = {render(value)}" for key, value in override.items()]

    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return destination


def _run_mypy(config: Path, target: Path, cache: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "mypy",
            "--config-file",
            str(config),
            "--cache-dir",
            str(cache),
            str(target),
        ],
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("component", "enrolled"),
    [("core", True), (sorted(NOT_YET_ENROLLED)[0], False)],
    ids=["enrolled-component-fails", "exempt-component-passes"],
)
def test_the_gate_fires(component: str, enrolled: bool, tmp_path: Path) -> None:
    """The same error, in two components, judged differently by the committed configuration.

    This is the half of the suite that cannot be satisfied by a configuration that has stopped
    working. The first case is the issue's acceptance criterion — a deliberate type error in an
    enrolled component fails — and the second proves the exemptions are doing the work rather than
    the gate being off everywhere.
    """
    tree = tmp_path / "tree"
    _synthetic_tree(tree, [component])
    config = _config_against(tree, tmp_path / "mypy.toml")
    probe = tree / "astro_mine" / component / "_probe.py"

    result = _run_mypy(config, probe, tmp_path / "cache")
    report = f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"

    if enrolled:
        assert result.returncode != 0, f"an enrolled component accepted an untyped def\n{report}"
        assert "no-untyped-def" in result.stdout, report
    else:
        assert result.returncode == 0, f"an exempt component was still checked\n{report}"
