"""Learn's adapter for the umbrella CLI (`astro-mine train`).

Asserted here, at home, **without importing the umbrella** — `astro-mine-cli` is not a dependency
of this package and must not become one (``conventions.md §1.1``) — so these tests check the same
structural shape the umbrella checks at dispatch.

Learn's CLI is flat, so the anti-drift check is the strongest of any component's: the umbrella's
parser and `astro-mine-train`'s must accept the *identical* option set, with nothing extra on
either side.
"""

from __future__ import annotations

import argparse
from importlib.metadata import entry_points

from astro_mine.learn import umbrella
from astro_mine.learn.train.run import _parser

UMBRELLA_GROUP = "astro_mine.cli"


def test_adapter_satisfies_the_structural_contract() -> None:
    for member in ("name", "help", "add_arguments", "run"):
        assert hasattr(umbrella.train, member), f"missing {member!r}"
    assert callable(umbrella.train.add_arguments)
    assert callable(umbrella.train.run)
    assert umbrella.train.name == "train"
    assert umbrella.train.help


def test_the_entry_point_resolves_and_matches_its_name() -> None:
    """A typo in pyproject.toml is invisible until a user types the verb — unless this runs."""
    (ours,) = [
        ep
        for ep in entry_points(group=UMBRELLA_GROUP)
        if ep.value.startswith("astro_mine.learn.umbrella:")
    ]
    assert ours.name == "train"
    assert ours.load().name == ours.name


def test_both_surfaces_accept_exactly_the_same_flags() -> None:
    """Learn's CLI is flat, so the whole parser is the verb — the sets must match exactly."""
    umbrella_parser = argparse.ArgumentParser(prog="astro-mine train")
    umbrella.train.add_arguments(umbrella_parser)
    assert _option_strings(umbrella_parser) == _option_strings(_parser())


def test_the_export_flags_are_on_the_umbrella_surface() -> None:
    """`--export` is the reason this verb matters: it produces the commons' unit of exchange
    (G1.4). A umbrella surface that trained but could not export would be a trap."""
    parser = argparse.ArgumentParser(prog="astro-mine train")
    umbrella.train.add_arguments(parser)
    options = _option_strings(parser)
    assert {"--export", "--export-format", "--export-version"} <= options


def test_env_factory_stays_required_on_the_umbrella_surface() -> None:
    """Missing `--env-factory` must fail as an argparse usage error on both surfaces, not reach
    the training code and raise something less legible."""
    parser = argparse.ArgumentParser(prog="astro-mine train")
    umbrella.train.add_arguments(parser)
    try:
        parser.parse_args([])
    except SystemExit as exit_:
        assert exit_.code == 2
    else:  # pragma: no cover - the assertion below reports it
        raise AssertionError("--env-factory should be required")


def _option_strings(parser: argparse.ArgumentParser) -> set[str]:
    return {option for action in parser._actions for option in action.option_strings}
