"""Guard's adapter for the umbrella CLI (`astro-mine guard`).

Asserted here, at home, **without importing the umbrella** — `astro-mine-cli` is not a dependency
of this package and must not become one (``conventions.md §1.1``) — so these tests check the same
structural shape the umbrella checks at dispatch.

This is a *passthrough* adapter, so there is no flag list to keep in sync: the tail goes to
the same ``main`` `astro-mine-guard` runs. What is worth pinning is that the tail arrives
**intact** — flags included, unparsed by the umbrella — and that the exit status survives
the hand-off.
"""

from __future__ import annotations

import argparse
from importlib.metadata import entry_points

import pytest

from astro_mine.guard import umbrella

UMBRELLA_GROUP = "astro_mine.cli"


def test_adapter_satisfies_the_structural_contract() -> None:
    for member in ("name", "help", "add_arguments", "run"):
        assert hasattr(umbrella.guard, member), f"missing {member!r}"
    assert callable(umbrella.guard.add_arguments)
    assert callable(umbrella.guard.run)
    assert umbrella.guard.name == "guard"
    assert umbrella.guard.help


def test_the_entry_point_resolves_and_matches_its_name() -> None:
    """A typo in pyproject.toml is invisible until a user types the verb — unless this runs."""
    (ours,) = [
        ep
        for ep in entry_points(group=UMBRELLA_GROUP)
        if ep.value.startswith("astro_mine.guard.umbrella:")
    ]
    assert ours.name == "guard"
    assert ours.load().name == ours.name


def test_the_tail_arrives_intact() -> None:
    """The point of a passthrough: the umbrella parses the verb and nothing after it, so flags
    reach Guard's own parser unmangled — including ones the umbrella has never heard of."""
    parser = argparse.ArgumentParser(prog="astro-mine guard")
    umbrella.guard.add_arguments(parser)
    args = parser.parse_args(["validate", "--some-flag", "value", "positional"])
    assert args.tail == ["validate", "--some-flag", "value", "positional"]


def test_the_exit_status_survives_the_handoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """`run` must *return* the status. A passthrough that dropped it would turn a failing command
    into a passing script."""
    monkeypatch.setattr(umbrella, "main", lambda argv: 3, raising=True)
    parser = argparse.ArgumentParser(prog="astro-mine guard")
    umbrella.guard.add_arguments(parser)
    assert umbrella.guard.run(parser.parse_args(["validate"])) == 3


def test_a_usage_error_becomes_a_status_not_an_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """argparse inside Guard raises SystemExit; the umbrella's contract is a returned status, so
    the adapter converts it. The code itself is unchanged."""

    def _exit(argv: list[str]) -> int:
        raise SystemExit(2)

    monkeypatch.setattr(umbrella, "main", _exit, raising=True)
    parser = argparse.ArgumentParser(prog="astro-mine guard")
    umbrella.guard.add_arguments(parser)
    assert umbrella.guard.run(parser.parse_args(["validate"])) == 2


def test_a_clean_systemexit_is_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """`raise SystemExit()` with no code means success everywhere in Python; it must not become a
    spurious failure just because it crossed an adapter."""

    def _exit(argv: list[str]) -> int:
        raise SystemExit

    monkeypatch.setattr(umbrella, "main", _exit, raising=True)
    parser = argparse.ArgumentParser(prog="astro-mine guard")
    umbrella.guard.add_arguments(parser)
    assert umbrella.guard.run(parser.parse_args(["validate"])) == 0
