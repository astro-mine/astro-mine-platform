"""The bare-install state — the first thing every user sees, before installing a component.

Regression cover from the standup (#1), kept intact through the dispatcher's arrival: a machine
with no components installed is the normal starting point, and the umbrella's behaviour there is
a product decision, not an edge case.

These tests inject an explicitly empty verb set rather than relying on the ambient environment,
so they keep testing the empty state even in a checkout where a component happens to be installed.
"""

from __future__ import annotations

import pytest

from astro_mine.cli import build_parser, main

EMPTY: dict[str, object] = {}


def test_bare_invocation_prints_help_and_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    """Asking a dispatcher what it can do is a legitimate question with a complete answer."""
    assert main([], verbs=EMPTY) == 0  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert "usage: astro-mine" in out
    assert "Verbs:" in out


def test_validate_is_always_available(capsys: pytest.CaptureFixture[str]) -> None:
    """The umbrella is never truly verb-less any more: it owns `validate` itself, because routing a
    document to the component that owns its format is the one job no component can do (RFC-0011
    §6). With nothing installed the verb is still listed — running it then names the package that
    owns the format at hand, which is the honest failure."""
    main([], verbs=EMPTY)  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert "validate" in out.split("not installed here")[0]


def test_the_empty_state_still_maps_the_platform(capsys: pytest.CaptureFixture[str]) -> None:
    """An empty dispatcher must not read as "the platform has no commands".

    Before the manifest, this screen was a dead end. Now a newcomer with nothing installed can see
    what exists and what to install for it — which is the discovery gap (UC-A3) the umbrella was
    built to close, and it works before they have installed a single component.
    """
    main([], verbs=EMPTY)  # type: ignore[arg-type]
    out = capsys.readouterr().out
    assert "not installed here" in out
    assert "score" in out and "astro-mine-bench" in out
    assert "astro-mine-bench score" in out  # the component CLIs work directly, today


def test_unknown_verb_fails(capsys: pytest.CaptureFixture[str]) -> None:
    """Silence on an unrecognized verb would be the dishonest case — argparse exits 2."""
    with pytest.raises(SystemExit) as excinfo:
        main(["definitely-not-a-verb"], verbs=EMPTY)  # type: ignore[arg-type]
    assert excinfo.value.code == 2
    assert "unknown verb" in capsys.readouterr().err


def test_help_exits_zero() -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"], verbs=EMPTY)  # type: ignore[arg-type]
    assert excinfo.value.code == 0


def test_version_flag_reports_the_package_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"], verbs=EMPTY)  # type: ignore[arg-type]
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.startswith("astro-mine ")


def test_parser_is_built_per_call() -> None:
    """Not a style point: the verb set is read from installed metadata at build time, so a cached
    parser would freeze the environment as it looked at first import."""
    assert build_parser(EMPTY) is not build_parser(EMPTY)  # type: ignore[arg-type]
