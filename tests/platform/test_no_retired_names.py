"""Nothing may name a distribution a user cannot install, or a binary they cannot run.

Two sibling defects, one file. platform#6 drew its line at strings a user is told to **install**;
platform#8 is the line one over — strings a user is told to **run**. Both send a reader to a shell
that answers "command not found".

Consolidation retired seventeen ``astro-mine-<component>`` distributions in favour of one
``astro-mine-platform`` wheel, and moved each optional extra into that wheel's
``<component>-<extra>`` namespace. The degradation messages did not follow. A user who hits an
optional-dependency guard reads ``pip install 'astro-mine-cloud[s3]'``, runs it, gets "No matching
distribution found", and concludes their environment is broken rather than the message.

That is the same defect ``astro-mine-cli`` found on its own side (astro-mine-cli#19, fixed in #20).
It could not be fixed from there, because the strings live here.

``conventions.md §13`` retired every ``astro-mine-<component>`` *script* along with the
distributions: there is exactly one executable, ``astro-mine``, under one grammar, and "any such
name in a document, a docstring, or a blog post is historical". Three of those names reached a
terminal directly, as ``argparse`` ``prog=`` values printed in ``--help`` and in every usage error.

**Four rules, and why each is written the way it is.**

*Install hints* are matched by looking for an install verb and reading forward across the line
break, because the offending distribution is often on the *next* source line — two of the original
28 wrapped exactly that way and a line-oriented scan missed both.

*Extras* are checked against ``pyproject.toml`` rather than a list. Rewriting
``astro-mine-sim[bench]`` to ``astro-mine-platform[sim-bench]`` is only a fix if ``sim-bench`` is a
real extra; a typo would trade one uninstallable name for another, and nothing else would catch it.
Comma-separated extras are split, because ``astro-mine-platform[sim-dem,sim-hub]`` is one valid
PEP 508 requirement and a single-extra pattern reads straight past it.

*Shipped examples* are checked whole, not only for install lines. An example is package data a
scaffold copies — ``astro-mine new world`` writes ``synthetic_polar.world.yaml`` verbatim — so a
dead command in one is reproduced into every document a user scaffolds from it.

*Invocations* (platform#8) are matched by **position, not by vocabulary**. A verb allowlist would
be the thing this file's ``RETIRED`` set deliberately is not — a list that rots — so the check asks
where the name sits rather than what follows it: a retired name is an offence when it is the first
token of a code span, of a quoted literal, or of a line. That is where a reader is told to *type*
something. It is *not* an offence in running prose, where the name means a package or a repository
and is correct: "the astro-mine-hub client", "The astro-mine-guard code version", "(astro-mine-hub
is git-pinned in pyproject.toml)". Both directions are asserted below, so the rule's edges are
tested rather than assumed.

**Deliberately not a blanket source scan.** Backlog URLs
(``astro-mine-worlds#7``) point at repositories that still
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

#: ``astro-mine-platform[<extra>[,<extra>…]]`` — the form every rewritten hint now takes. The comma
#: alternative is load-bearing: ``[sim-dem,sim-hub,sim-surrogate]`` is one requirement, and a
#: single-extra pattern matches no part of it, so three unchecked extras would ride along silently.
#: F-string forms (``astro-mine-platform[learn-{extra}]``) still do not match, and must not — the
#: extra is not known until runtime, so there is nothing to check it against.
_PLATFORM_EXTRA = re.compile(r"astro-mine-platform\[([a-z0-9,-]+)\]")

#: A retired name in **command position**: the first token of a code span (`` `x` ``/``` ``x`` ```),
#: of a quoted literal (an ``argparse`` ``prog=``, a message, a document field), or of a line (an
#: RST literal block, a ``.. code-block:: console``, a shell script). Followed by a word, which is
#: what makes it an invocation rather than a bare mention. Anywhere else the name is prose about a
#: package and stays.
_INVOCATION = re.compile(
    r"(?:[`\"']|^[ \t]*)"
    r"(" + "|".join(re.escape(name) for name in RETIRED) + r")"
    r"[ \t]+([a-z][a-z0-9-]*)",
    re.MULTILINE,
)

#: An ``argparse`` ``prog=`` naming a retired binary — **with or without a verb after it**.
#:
#: Its own rule, because the one that reached a terminal bare was ``prog="astro-mine-sim"``, and
#: :data:`_INVOCATION` requires a following word to tell a command from a mention. Widening that to
#: "a retired name alone inside quotes" would be wrong: ``writer.start(profile="astro-mine-sim",
#: library="astro-mine-bench")`` writes those into every MCAP header, and a wire value's whole
#: purpose is not to change. ``prog=`` cannot be anything but a command, so it needs no such care.
_PROG_BINARY = re.compile(
    r"prog=[\"'](" + "|".join(re.escape(name) for name in RETIRED) + r")(?![\w-])"
)

#: Files whose retired names are **hash inputs or history**, and must not be edited.
#:
#: A zoo ``pins.json`` ``recipe`` is content-hashed alongside its producer/id/version by
#: :func:`~astro_mine.bench.zoo._provisional.provisional_pin_hash`, and a ``scenario.json``
#: ``description`` is part of the ``model_dump`` behind
#: :attr:`~astro_mine.bench.scenario.ScenarioSpec.spec_hash`. Rewriting either moves a digest,
#: which re-pins the scenario zoo and invalidates every published scorecard that resolves against
#: it. ``PROVENANCE.md`` records how published content was actually built: the command that ran is
#: the command that ran.
#:
#: They are excluded by name rather than left to the position rule to spare, so that tightening
#: that rule later cannot start demanding an edit to content that must not change.
_FROZEN_BY_HASH_OR_HISTORY = frozenset({"pins.json", "scenario.json", "PROVENANCE.md"})

#: No directory is excluded any more, and the reason the last one was is worth keeping.
#:
#: ``examples/downstream-consumer/`` was exempt because its retired distribution name was a
#: **working instruction**: the ``astro-mine-core`` repository and its ``v0.1.0`` tag both existed,
#: so ``uv sync --locked`` resolved. That is the one case this gate cannot judge — it matches a
#: name, not whether a Git remote answers.
#:
#: Deleting the archived repositories made the premise false, and with it the exemption. The
#: example was removed rather than retargeted: "it still resolves" was its only remaining
#: justification, and the history it recorded lives in `git log` and in the repository mirrors kept
#: under `files/archived-repos-backup/`. The gate now covers `examples/` with no holes.

#: How far past an install verb to keep looking. Comfortably clears a wrapped string literal and
#: its indentation without running into the next statement.
_HINT_WINDOW = 160


def _source_files() -> list[Path]:
    """The package, ``scripts/``, **and the two document trees a reader actually meets first**.

    ``scripts/`` was outside platform#6's scan and carried two dead install lines because of it —
    including one telling the reader to ``uv pip install`` three retired distributions. A script a
    contributor is told to run is as user-facing as a degradation message, and having the four
    checks disagree about which tree they cover is the seam this file exists to close.

    ``docs/`` and ``examples/`` were outside platform#8's scan for the same reason and carried
    more than ``src/`` and ``scripts/`` combined: **91 invocations and 25 install hints**
    (platform#10). They are the per-repo READMEs consolidation copied in, each written for a repo
    that shipped its own wheel and its own binary, and both premises are now false. The audience is
    what makes them worth the sweep — a degradation message is read by someone already running the
    code, mid-task; a README is read by someone deciding whether to use the thing at all, which is
    the more expensive place to be wrong.

    The position rule was the open question here: it keys on a retired name being the first token
    of a code span, a quoted literal, or a line, and markdown prose is full of sentences that begin
    with a package name. It over-fired on none of the 4,142 lines swept, so no markdown-specific
    rule was needed — the delimiters that make a command a command are the same in a docstring and
    in a README.
    """
    roots = (PACKAGE_ROOT, REPO_ROOT / "scripts", REPO_ROOT / "docs", REPO_ROOT / "examples")
    return [
        path
        for root in roots
        for path in sorted(root.rglob("*"))
        if path.is_file()
        and "__pycache__" not in path.parts
        and path.suffix in {".py", ".yaml", ".yml", ".json", ".md", ".toml", ".cfg", ".txt", ".sh"}
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
            for extra in match.group(1).split(","):
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


def test_nothing_invokes_a_retired_binary() -> None:
    """Acceptance criterion 4 (platform#8): no docstring, message or ``prog=`` names a dead command.

    The three that reached a terminal were ``argparse`` ``prog=`` values, printed in ``--help`` and
    in every usage error. All three name ``python -m`` entry points, which ``conventions.md §13``
    explicitly keeps and ``cli.md §10`` names — Bench's per-seed ``eval-worker`` and Sim's container
    entrypoint are deliberately *not* verbs — so the honest ``prog`` is the argv the process is
    launched with, not an ``astro-mine bench eval-worker`` that does not exist. The other ~30 were
    docstrings and runtime messages, and each was checked against the CLI's real surface before
    being rewritten: inventing a verb in a docstring is worse than the stale name it replaces.
    """
    offences: list[str] = []
    for path in _source_files():
        if path.name in _FROZEN_BY_HASH_OR_HISTORY:
            continue
        text = path.read_text(encoding="utf-8")
        for match in _INVOCATION.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offences.append(
                f"{path.relative_to(REPO_ROOT)}:{line}: invokes `{match.group(1)} {match.group(2)}`"
            )
        for match in _PROG_BINARY.finditer(text):
            line = text.count("\n", 0, match.start()) + 1
            offences.append(f"{path.relative_to(REPO_ROOT)}:{line}: prog= names {match.group(1)}")
    assert offences == []


@pytest.mark.parametrize(
    "invocation",
    [
        # An RST code span in a docstring, and a markdown/f-string one in a runtime message.
        "resolves the pinned content (``astro-mine-sim run``) from a local store",
        "poll it later with `astro-mine-bench submit --job {job_id} --wait`",
        # An `argparse` prog=, which argparse prints on every usage error.
        'parser = argparse.ArgumentParser(prog="astro-mine-bench eval-worker")',
        # A document field naming the tool that authored it.
        'author="astro-mine-studio serve",',
        # A line in an RST literal block or a shell script, where nothing delimits the command.
        "\n.. code-block:: console\n\n    astro-mine-worlds schema > worldspec.schema.json\n",
        "\n    astro-mine-guard sign my.safety.yaml --verify --pub key.pub.pem\n",
    ],
)
def test_the_invocation_check_fires(invocation: str) -> None:
    assert _INVOCATION.search(invocation) is not None


@pytest.mark.parametrize(
    "prog",
    [
        # The bare form, which has no verb for the position rule to key on.
        'argparse.ArgumentParser(prog="astro-mine-sim", description=__doc__)',
        "argparse.ArgumentParser(prog='astro-mine-learn vector benchmark')",
    ],
)
def test_the_prog_check_fires(prog: str) -> None:
    assert _PROG_BINARY.search(prog) is not None


def test_the_prog_check_spares_wire_values() -> None:
    """A retired name inside quotes is not automatically a command — MCAP headers carry two."""
    header = 'writer.start(profile="astro-mine-sim", library="astro-mine-bench")'
    assert _PROG_BINARY.search(header) is None
    assert _INVOCATION.search(header) is None


@pytest.mark.parametrize(
    "prose",
    [
        # The name of a package, as the subject or object of a sentence. All four are live text in
        # this tree, and all four are correct: they mean the package, not a command.
        "A Hub publish/pull could not be completed (e.g. astro-mine-hub is not installed).",
        '"description": "The astro-mine-guard code version that produced this verdict"',
        "the caller passes a ``publisher`` callable (the astro-mine-hub client adapts to it)",
        "install astro-mine-platform[link-hub] (astro-mine-hub is git-pinned in pyproject.toml)",
        # A repository, named as the place a sibling file lives.
        "# Mirrors astro-mine-core / astro-mine-surrogate scripts/gen_proto.sh.",
        "Usage (from the astro-mine-link project environment):",
        # History: the ContactPlan was computed by that package, and still was.
        "It also contradicted the ContactPlan this scenario pins, which astro-mine-link computed",
    ],
)
def test_the_invocation_check_spares_prose_about_packages(prose: str) -> None:
    assert _INVOCATION.search(prose) is None


def test_the_frozen_files_are_real_and_still_carry_retired_names() -> None:
    """The exclusion must stay load-bearing, or it is an oversight wearing a reason.

    If a future sweep rewrites these anyway, this fails and asks why — rather than leaving a
    silent carve-out for content that no longer needs one.
    """
    frozen = [path for path in _source_files() if path.name in _FROZEN_BY_HASH_OR_HISTORY]
    assert frozen, "no frozen file found; the exclusion no longer matches anything"
    assert [path for path in frozen if _INVOCATION.search(path.read_text(encoding="utf-8"))]


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
        "Backlog: RM-P0-CLOUD-03 -- astro-mine-cloud#3",
        # A wire constant: its value must not change.
        'BUNDLE_SCHEMA = "astro-mine-worlds/world/v0.1"',
        # A cross-repo issue reference: the repository still exists.
        "shipped as package data via astro_mine.guard.reference since astro-mine-guard#29",
    ],
)
def test_the_retired_name_check_spares_urls_and_wire_constants(benign: str) -> None:
    assert _RETIRED_NAME.findall(benign) == []
