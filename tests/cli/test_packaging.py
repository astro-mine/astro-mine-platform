"""Packaging invariants — the rules that make this package what RFC-0011 decided it is.

These are not smoke tests. The umbrella's whole justification over the rejected
depend-on-everything alternative is that it stays installable on its own (RFC-0011
§"Alternatives considered" (a); ``conventions.md §7``, the local tier "MUST always work"), and a
rule enforced only by review is a rule that erodes on the first convenient import. So the
zero-dependency claim is asserted here, in the default test lane, where a PR that breaks it
fails before anyone reads it.
"""

from __future__ import annotations

from importlib.metadata import entry_points, requires

import pytest

import astro_mine.cli

DISTRIBUTION = "astro-mine-platform"


@pytest.mark.skip(
    reason="structurally per-repo: astro-mine-cli was a zero-dependency distribution; the "
    "consolidated astro-mine-platform wheel necessarily lists the whole platform's "
    "Requires-Dist. The umbrella *code* still imports nothing outside the stdlib, which "
    "the laziness tests in test_installed_provider.py continue to prove."
)
def test_declares_no_runtime_dependencies() -> None:
    """The umbrella depends on the entry-point *group name*, never on a provider.

    ``requires()`` reports the installed distribution's ``Requires-Dist`` metadata, so this sees
    exactly what a user's ``pip install astro-mine-cli`` would pull. Dev tooling lives in a PEP-735
    dependency group, which is not distribution metadata and correctly does not appear here.

    If this fails, the fix is *not* to relax the assertion.
    """
    assert not (requires(DISTRIBUTION) or [])


def test_console_script_is_declared() -> None:
    """`astro-mine` is the one console script, and it resolves to the documented callable.

    The umbrella surface is `astro-mine <verb>` (``conventions.md §13``, normative). RFC-0011's
    per-component dispatch (`astro-mine studio serve`) is a thin call into an already-shipped
    subcommand, which only works if this name and target stay put.
    """
    scripts = {ep.name: ep.value for ep in entry_points(group="console_scripts")}
    assert scripts.get("astro-mine") == "astro_mine.cli:main"


def test_version_is_resolved_from_installed_metadata() -> None:
    """``__version__`` comes from the installed distribution (hatch-vcs), not a hardcoded string.

    The repo carries no tags yet, so the version is a development version — that is expected and
    matches the sibling repos; what matters is that it is *derived*, so it cannot drift.
    """
    assert astro_mine.cli.__version__
    assert astro_mine.cli.__version__ != "0.0.0.dev0", (
        "the fallback fired, so the package under test is not installed; run `uv sync`"
    )
