"""Sim's adapters for the umbrella CLI (`astro-mine run|record`).

The contract is asserted here, at home, **without importing the umbrella** — `astro-mine-cli` is
deliberately not a dependency of this package (``conventions.md §1.1``), so these tests check the
same structural shape the umbrella checks at dispatch.

The anti-drift test is the one that earns its place: both surfaces are built from the same
argument functions, and that stays true only while nobody re-declares a flag in the adapter. It
matters more in Sim than elsewhere because `record`'s `--scenario` alias is what keeps the
container entrypoint working (cloud.md §4) — losing it on one surface would be a silent break.
"""

from __future__ import annotations

import argparse
from importlib.metadata import entry_points

import pytest

from astro_mine.sim import umbrella
from astro_mine.sim.__main__ import _build_parser

ADAPTERS = [umbrella.run, umbrella.record]
UMBRELLA_GROUP = "astro_mine.cli"


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: str(a.name))
def test_adapter_satisfies_the_structural_contract(adapter: object) -> None:
    for member in ("name", "help", "add_arguments", "run"):
        assert hasattr(adapter, member), f"{adapter} is missing {member!r}"
    assert callable(adapter.add_arguments)  # type: ignore[attr-defined]
    assert callable(adapter.run)  # type: ignore[attr-defined]
    assert isinstance(adapter.name, str) and adapter.name  # type: ignore[attr-defined]
    assert isinstance(adapter.help, str) and adapter.help  # type: ignore[attr-defined]


def test_every_sim_entry_point_resolves() -> None:
    """A typo in pyproject.toml is invisible until a user types the verb — unless this runs."""
    ours = [
        ep
        for ep in entry_points(group=UMBRELLA_GROUP)
        if ep.value.startswith("astro_mine.sim.umbrella:")
    ]
    assert {ep.name for ep in ours} == {"run", "record"}
    for ep in ours:
        assert ep.load().name == ep.name  # the object's name must match the routing name


@pytest.mark.parametrize("verb", ["run", "record"])
def test_both_surfaces_accept_the_same_flags(verb: str) -> None:
    """The anti-drift check — including `record --scenario`, the container's alias."""
    adapter = {"run": umbrella.run, "record": umbrella.record}[verb]
    umbrella_parser = argparse.ArgumentParser(prog=f"astro-mine {verb}")
    adapter.add_arguments(umbrella_parser)
    assert _option_strings(umbrella_parser) == _option_strings(_subparser_for(verb))


def test_the_container_alias_survives_on_the_umbrella_surface() -> None:
    """`--scenario` is what Cloud's workload image passes. A laptop run and a cluster run are
    supposed to be the same run, so the alias cannot exist on only one of the two surfaces."""
    parser = argparse.ArgumentParser(prog="astro-mine record")
    umbrella.record.add_arguments(parser)
    args = parser.parse_args(["--scenario", "episode.json"])
    assert args.scenario_file.name == "episode.json"


def test_run_rejects_a_missing_scenario_id() -> None:
    """The umbrella owns the parser, so its errors must still be argparse's, not a traceback."""
    parser = argparse.ArgumentParser(prog="astro-mine run")
    umbrella.run.add_arguments(parser)
    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args([])
    assert excinfo.value.code == 2


def _subparser_for(verb: str) -> argparse.ArgumentParser:
    (subcommands,) = [
        action
        for action in _build_parser()._actions
        if isinstance(action, argparse._SubParsersAction)
    ]
    return subcommands.choices[verb]


def _option_strings(parser: argparse.ArgumentParser) -> set[str]:
    return {option for action in parser._actions for option in action.option_strings}
