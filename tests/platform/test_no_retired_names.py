"""No install hint and no shipped example may name something a user cannot install (platform#6).

Consolidation retired seventeen ``astro-mine-<component>`` distributions in favour of one
``astro-mine-platform`` wheel, and moved each optional extra into that wheel's
``<component>-<extra>`` namespace. The degradation messages did not follow. A user who hits an
optional-dependency guard reads ``pip install 'astro-mine-cloud[s3]'``, runs it, gets "No matching
distribution found", and concludes their environment is broken rather than the message.

That is the same defect ``astro-mine-cli`` found on its own side (astro-mine-cli#19, fixed in #20).
It could not be fixed from there, because the strings live here.

**Three rules, and why each is written the way it is.**

*Install hints* are matched by looking for an install verb and reading forward across the line
break, because the offending distribution is often on the *next* source line — two of the original
28 wrapped exactly that way and a line-oriented scan missed both.

*Extras* are checked against ``pyproject.toml`` rather than a list. Rewriting
``astro-mine-sim[bench]`` to ``astro-mine-platform[sim-bench]`` is only a fix if ``sim-bench`` is a
real extra; a typo would trade one uninstallable name for another, and nothing else would catch it.

*Shipped examples* are checked whole, not only for install lines. An example is package data a
scaffold copies — ``astro-mine new world`` writes ``synthetic_polar.world.yaml`` verbatim — so a
dead command in one is reproduced into every document a user scaffolds from it.

**Deliberately not a blanket source scan.** Backlog URLs
(``https://github.com/astro-mine/astro-mine-worlds/issues/7``) point at repositories that still
exist, and wire constants (``BUNDLE_SCHEMA = "astro-mine-worlds/world/v0.1"``) are values whose
whole purpose is not to change. Both are correct as they stand, and a grep for the bare name would
fail on both.
"""

from __future__ import annotations

import pkgutil
import re
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT = REPO_ROOT / "src" / "astro_mine"

#: The two distributions that do exist (conventions.md §7.1). ``astro-mine-api`` is deliberately
#: absent: the repo is a placeholder, and a message naming an unbuilt thing as unbuilt is honest in
#: a way that an install line for it would not be.
LIVE_DISTRIBUTIONS = frozenset({"astro-mine-platform", "astro-mine-cli"})

#: The distributions consolidation retired, derived from the package layout rather than listed, so
#: a component added or renamed later cannot leave a stale allowlist behind.
#:
#: Derived from *this distribution's* tree, not from ``astro_mine.__path__``. ``astro_mine`` spans
#: every installed distribution that contributes to it, and a development environment normally has
#: ``astro-mine-cli`` installed editable alongside — which puts ``cli`` in the namespace and would
#: have this test declare ``astro-mine-cli``, the one live sibling, retired. ``LIVE_DISTRIBUTIONS``
#: is subtracted as well, so the answer does not depend on what happens to be installed.
RETIRED = sorted(
    name
    for name in (
        f"astro-mine-{module.name}"
        for module in pkgutil.iter_modules([str(PACKAGE_ROOT)])
        if module.ispkg
    )
    if name not in LIVE_DISTRIBUTIONS
)

_INSTALL_VERB = re.compile(r"(?:uv\s+)?pip\s+install|uv\s+add|uv\s+sync")

#: A retired distribution name — as a name, not as an address. Three forms are spared, and all
#: three point at a repository that still exists or a value that must not change: a GitHub URL
#: (``.../astro-mine-worlds/issues/7``, excluded by the leading ``/``), a cross-repo issue
#: reference (``astro-mine-guard#29``), and a schema ``$id`` (``astro-mine-worlds/world/v0.1``,
#: excluded by the trailing ``/``).
_RETIRED_NAME = re.compile(
    r"(?<![/\w-])(" + "|".join(re.escape(name) for name in RETIRED) + r")(?![\w-])(?![/#])"
)

#: ``astro-mine-platform[<extra>]`` — the form every rewritten hint now takes.
_PLATFORM_EXTRA = re.compile(r"astro-mine-platform\[([a-z0-9-]+)\]")

#: How far past an install verb to keep looking. Comfortably clears a wrapped string literal and
#: its indentation without running into the next statement.
_HINT_WINDOW = 160


def _source_files() -> list[Path]:
    return [
        path
        for path in sorted(PACKAGE_ROOT.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in {".py", ".yaml", ".yml", ".json", ".md", ".toml", ".cfg", ".txt"}
    ]


def _shipped_examples() -> list[Path]:
    """The authored-format documents this distribution ships as package data (conventions.md §13).

    Data files only. The ``reference/`` trees also hold the Python that *loads* them, and a
    docstring there may legitimately quote a retired name while explaining why it is retired —
    the same reason ``astro-mine-cli``'s equivalent test is not a source scan. What must be clean
    is the document a scaffold copies verbatim into a user's working directory.
    """
    return [
        path
        for directory in PACKAGE_ROOT.rglob("*")
        if directory.is_dir() and directory.name in {"examples", "reference"}
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix in {".yaml", ".yml", ".json", ".toml", ".md", ".txt"}
    ]


def _declared_extras() -> frozenset[str]:
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return frozenset(data["project"]["optional-dependencies"])


def test_the_retired_set_is_derived_not_listed() -> None:
    """If this ever shrinks to nothing the other tests pass vacuously, so assert it is populated."""
    assert len(RETIRED) >= 17
    assert "astro-mine-sim" in RETIRED
    assert LIVE_DISTRIBUTIONS.isdisjoint(RETIRED)


def test_no_install_hint_names_a_retired_distribution() -> None:
    """Acceptance criterion 1: no ``pip install`` hint names a distribution that cannot resolve."""
    offences: list[str] = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8")
        for verb in _INSTALL_VERB.finditer(text):
            window = text[verb.start() : verb.start() + _HINT_WINDOW]
            for name in sorted(set(_RETIRED_NAME.findall(window))):
                line = text.count("\n", 0, verb.start()) + 1
                offences.append(f"{path.relative_to(REPO_ROOT)}:{line}: install hint names {name}")
    assert offences == []


def test_every_platform_extra_named_in_source_is_declared() -> None:
    """Acceptance criterion 2: a rewritten extra must name one ``astro-mine-platform`` declares."""
    declared = _declared_extras()
    offences: list[str] = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8")
        for match in _PLATFORM_EXTRA.finditer(text):
            extra = match.group(1)
            if extra in declared:
                continue
            line = text.count("\n", 0, match.start()) + 1
            offences.append(
                f"{path.relative_to(REPO_ROOT)}:{line}: astro-mine-platform[{extra}] is not a "
                f"declared extra"
            )
    assert offences == []


def test_no_shipped_example_names_a_retired_distribution() -> None:
    """Acceptance criterion 3a: an example is copied, so a dead name in one multiplies."""
    offences = [
        f"{path.relative_to(REPO_ROOT)}: names {name}"
        for path in _shipped_examples()
        for name in sorted(set(_RETIRED_NAME.findall(path.read_text(encoding="utf-8"))))
    ]
    assert offences == []


def test_no_shipped_example_names_a_retired_binary() -> None:
    """Acceptance criterion 3b: ``astro-mine new world`` must stop writing a dead command.

    conventions.md §13 retired every ``astro-mine-<component>`` script along with the distributions;
    the one address is ``astro-mine <component> <verb>``. The example carried
    ``astro-mine-worlds validate <path>``, and it is the text the scaffold writes, so every
    scaffolded WorldSpec inherited it.
    """
    binary = re.compile(
        r"(?<![/\w-])(" + "|".join(re.escape(name) for name in RETIRED) + r")\s+[a-z]"
    )
    offences = [
        f"{path.relative_to(REPO_ROOT)}: invokes {name} as a command"
        for path in _shipped_examples()
        for name in sorted(set(binary.findall(path.read_text(encoding="utf-8"))))
    ]
    assert offences == []


@pytest.mark.parametrize(
    "hint",
    [
        "pip install 'astro-mine-cloud[s3]'",
        "uv pip install astro-mine-sim",
        "uv add 'astro-mine-learn[jax]'",
        # The wrapped form two of the original 28 used, which a line-oriented scan misses.
        "install the 'cluster' extra: pip install \n            'astro-mine-cloud[cluster]'",
    ],
)
def test_the_install_hint_check_fires(hint: str) -> None:
    verb = _INSTALL_VERB.search(hint)
    assert verb is not None
    assert _RETIRED_NAME.findall(hint[verb.start() : verb.start() + _HINT_WINDOW])


@pytest.mark.parametrize(
    "benign",
    [
        # A backlog URL: the repository still exists.
        "Backlog: RM-P0-CLOUD-03 -- https://github.com/astro-mine/astro-mine-cloud/issues/3",
        # A wire constant: its value must not change.
        'BUNDLE_SCHEMA = "astro-mine-worlds/world/v0.1"',
        # A cross-repo issue reference: the repository still exists.
        "shipped as package data via astro_mine.guard.reference since astro-mine-guard#29",
    ],
)
def test_the_retired_name_check_spares_urls_and_wire_constants(benign: str) -> None:
    assert _RETIRED_NAME.findall(benign) == []
