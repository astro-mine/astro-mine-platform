"""Bench's adapters for the umbrella CLI (`astro-mine score|fetch|submit|list`).

Two things are worth testing here, and neither is "does argparse work".

**The contract, without importing the umbrella.** `astro-mine-cli` is deliberately not a
dependency of this package (``conventions.md §1.1``), so these tests assert the *shape* the
umbrella binds to — four members, callable — exactly as the umbrella's own structural check does.
If Bench ever drifts from that shape, this fails here rather than in someone's terminal.

**The two surfaces cannot diverge.** `astro-mine score` and `astro-mine-bench score` are supposed
to accept the same flags because they attach the same argument function. That is only true while
nobody "helpfully" re-declares a flag in the adapter, so the equality is asserted rather than
assumed.
"""

from __future__ import annotations

import argparse
import io
from importlib.metadata import entry_points

import pytest

from astro_mine.bench import umbrella
from astro_mine.bench.cli import _build_parser

ADAPTERS = [umbrella.score, umbrella.fetch, umbrella.submit, umbrella.list_scenarios]
UMBRELLA_GROUP = "astro_mine.cli"


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: str(a.name))
def test_adapter_satisfies_the_structural_contract(adapter: object) -> None:
    """The four members the umbrella checks for at dispatch, checked at home instead."""
    for member in ("name", "help", "add_arguments", "run"):
        assert hasattr(adapter, member), f"{adapter} is missing {member!r}"
    assert callable(adapter.add_arguments)  # type: ignore[attr-defined]
    assert callable(adapter.run)  # type: ignore[attr-defined]
    assert isinstance(adapter.name, str) and adapter.name  # type: ignore[attr-defined]
    assert isinstance(adapter.help, str) and adapter.help  # type: ignore[attr-defined]


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: str(a.name))
def test_adapter_name_matches_its_entry_point(adapter: object) -> None:
    """The umbrella routes on the entry-point name and reports errors using the object's ``name``.
    A mismatch would make one of the two a lie in exactly the situation a user needs the truth."""
    declared = {ep.name: ep.value for ep in entry_points(group=UMBRELLA_GROUP)}
    assert adapter.name in declared, f"{adapter.name!r} is not declared in {UMBRELLA_GROUP}"  # type: ignore[attr-defined]
    assert declared[adapter.name].startswith("astro_mine.bench.umbrella:")  # type: ignore[attr-defined]


def test_every_bench_entry_point_resolves() -> None:
    """A typo in pyproject.toml is invisible until a user types the verb — unless this runs."""
    ours = [
        ep
        for ep in entry_points(group=UMBRELLA_GROUP)
        if ep.value.startswith("astro_mine.bench.umbrella:")
    ]
    assert {ep.name for ep in ours} == {"score", "fetch", "submit", "list"}
    for ep in ours:
        assert ep.load() is not None


@pytest.mark.parametrize("verb", ["score", "fetch", "submit"])
def test_both_surfaces_accept_the_same_flags(verb: str) -> None:
    """The anti-drift check.

    Both parsers are built from the same ``add_*_arguments`` function, so their option strings must
    match exactly. This fails the moment someone re-declares a flag in the adapter instead of
    extending the shared function — which is the one way these two surfaces can quietly diverge.
    """
    adapter = {"score": umbrella.score, "fetch": umbrella.fetch, "submit": umbrella.submit}[verb]
    umbrella_parser = argparse.ArgumentParser(prog=f"astro-mine {verb}")
    adapter.add_arguments(umbrella_parser)

    native = _subparser_for(verb)
    assert _option_strings(umbrella_parser) == _option_strings(native)


def test_list_runs_through_the_adapter(capsys: pytest.CaptureFixture[str]) -> None:
    """The end-to-end path for the one verb that needs no content, no network and no extra.

    It also covers the wiring bug this adapter had to fix: Bench's handlers print through
    ``args.stdout``, which only ``cli.main`` sets — so an adapter that just forwarded the namespace
    would raise AttributeError on its first line of output.
    """
    parser = argparse.ArgumentParser(prog="astro-mine list")
    umbrella.list_scenarios.add_arguments(parser)
    assert umbrella.list_scenarios.run(parser.parse_args([])) == 0
    assert "lunar-polar-ice-prospecting-v1" in capsys.readouterr().out


def test_injected_streams_are_preserved() -> None:
    """Defaulting, not overwriting: a caller that supplies buffers still gets them."""
    buffer = io.StringIO()
    parser = argparse.ArgumentParser(prog="astro-mine list")
    umbrella.list_scenarios.add_arguments(parser)
    args = parser.parse_args([])
    args.stdout = buffer
    args.stderr = buffer
    assert umbrella.list_scenarios.run(args) == 0
    assert "lunar-polar-ice-prospecting-v1" in buffer.getvalue()


def _subparser_for(verb: str) -> argparse.ArgumentParser:
    """Dig the native `astro-mine-bench <verb>` parser out of the top-level one."""
    actions = [
        action
        for action in _build_parser()._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    (subcommands,) = actions
    return subcommands.choices[verb]


def _option_strings(parser: argparse.ArgumentParser) -> set[str]:
    return {option for action in parser._actions for option in action.option_strings}
