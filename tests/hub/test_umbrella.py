"""Hub's adapters for the umbrella CLI (`astro-mine publish|search|pull|verify`).

Asserted here, at home, **without importing the umbrella** — `astro-mine-cli` is not a dependency
of this package and must not become one (``conventions.md §1.1``) — so these tests check the same
structural shape the umbrella checks at dispatch.

The anti-drift test carries real weight in Hub: `--trusted-key` and `--no-verify` decide what a
*verified* pull means. A surface quietly missing one of them would not be a missing convenience,
it would be a supply-chain footgun, so the option sets are asserted equal rather than assumed.
"""

from __future__ import annotations

import argparse
from importlib.metadata import entry_points

import pytest

from astro_mine.hub import umbrella
from astro_mine.hub.client.cli import _parser

ADAPTERS = [umbrella.publish, umbrella.search, umbrella.pull, umbrella.verify]
UMBRELLA_GROUP = "astro_mine.cli"


@pytest.mark.parametrize("adapter", ADAPTERS, ids=lambda a: str(a.name))
def test_adapter_satisfies_the_structural_contract(adapter: object) -> None:
    for member in ("name", "help", "add_arguments", "run"):
        assert hasattr(adapter, member), f"{adapter} is missing {member!r}"
    assert callable(adapter.add_arguments)  # type: ignore[attr-defined]
    assert callable(adapter.run)  # type: ignore[attr-defined]
    assert isinstance(adapter.name, str) and adapter.name  # type: ignore[attr-defined]
    assert isinstance(adapter.help, str) and adapter.help  # type: ignore[attr-defined]


def test_every_hub_entry_point_resolves_and_matches_its_name() -> None:
    """A typo in pyproject.toml is invisible until a user types the verb — unless this runs."""
    ours = [
        ep
        for ep in entry_points(group=UMBRELLA_GROUP)
        if ep.value.startswith("astro_mine.hub.umbrella:")
    ]
    assert {ep.name for ep in ours} == {"publish", "search", "pull", "verify"}
    for ep in ours:
        assert ep.load().name == ep.name


def test_resolve_and_keygen_are_not_umbrella_verbs() -> None:
    """Deliberate, not forgotten: a pin lookup only means something in Hub's own vocabulary, and
    key material is a Hub-scoped command. Both keep working on `astro-mine-hub`."""
    advertised = {
        ep.name
        for ep in entry_points(group=UMBRELLA_GROUP)
        if ep.value.startswith("astro_mine.hub.umbrella:")
    }
    assert "resolve" not in advertised
    assert "keygen" not in advertised
    native = _subparsers()
    assert {"resolve", "keygen"} <= set(native.choices)


@pytest.mark.parametrize("verb", ["publish", "search", "pull", "verify"])
def test_both_surfaces_accept_the_same_flags(verb: str) -> None:
    adapter = {
        "publish": umbrella.publish,
        "search": umbrella.search,
        "pull": umbrella.pull,
        "verify": umbrella.verify,
    }[verb]
    umbrella_parser = argparse.ArgumentParser(prog=f"astro-mine {verb}")
    adapter.add_arguments(umbrella_parser)
    assert _option_strings(umbrella_parser) == _option_strings(_subparsers().choices[verb])


def test_the_verification_flags_survive_on_the_umbrella_surface() -> None:
    """Named explicitly because these are the ones whose absence would be dangerous rather than
    merely inconvenient: they decide whether a pull is verified and against whose key."""
    parser = argparse.ArgumentParser(prog="astro-mine pull")
    umbrella.pull.add_arguments(parser)
    assert {"--trusted-key", "--no-verify"} <= _option_strings(parser)


def test_publish_still_validates_kind_on_the_umbrella_surface() -> None:
    """`--kind` is a closed vocabulary on publish (and free-form on search). The umbrella must not
    become a way around that check."""
    parser = argparse.ArgumentParser(prog="astro-mine publish")
    umbrella.publish.add_arguments(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--registry",
                "./reg",
                "--name",
                "x",
                "--version",
                "1.0.0",
                "--kind",
                "not-a-real-kind",
                "--manifest",
                "m.json",
                "--key",
                "k.pem",
            ]
        )


def _subparsers() -> argparse._SubParsersAction:  # type: ignore[type-arg]
    (action,) = [a for a in _parser()._actions if isinstance(a, argparse._SubParsersAction)]
    return action


def _option_strings(parser: argparse.ArgumentParser) -> set[str]:
    return {option for action in parser._actions for option in action.option_strings}
