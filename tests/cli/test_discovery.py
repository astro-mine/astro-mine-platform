"""Discovery: what the umbrella learns from metadata, and what it refuses to guess."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from _verbs import make_entry_point

from astro_mine.cli import VERB_ENTRY_POINT_GROUP, VerbCollisionError, discover_verbs
from astro_mine.cli._discovery import describe_provider, load_verb
from astro_mine.cli._protocol import InvalidSubcommandError, check_subcommand


def test_group_name_is_the_documented_one() -> None:
    """The group name is the contract. Components declare it in their own pyproject, so changing
    it silently unregisters every verb on the platform."""
    assert VERB_ENTRY_POINT_GROUP == "astro_mine.cli"


def test_discovers_advertised_verbs_by_name() -> None:
    entries = [make_entry_point("echo", "ECHO"), make_entry_point("quiet", "RETURNS_NONE")]
    assert sorted(discover_verbs(entries)) == ["echo", "quiet"]


def test_empty_environment_is_not_an_error() -> None:
    """A machine with no components installed is the normal first state, not a failure."""
    assert discover_verbs([]) == {}


def test_two_providers_claiming_one_verb_is_a_hard_error() -> None:
    """No precedence rule: a silent winner means the same command does different things on two
    machines, and nothing tells the user which one they have."""
    entries = [make_entry_point("echo", "ECHO"), make_entry_point("echo", "RETURNS_NONE")]
    with pytest.raises(VerbCollisionError) as excinfo:
        discover_verbs(entries)
    message = str(excinfo.value)
    assert "echo" in message
    # Both claimants named, so the user can act without guessing which to uninstall.
    assert message.count("_verbs:") == 2


def test_describe_provider_is_honest_without_a_distribution() -> None:
    """An entry point from a local or namespace install has no resolvable distribution. Report
    the target rather than attributing it to the umbrella."""
    assert describe_provider(make_entry_point("echo", "ECHO")) == "_verbs:ECHO"


def test_describe_provider_names_the_distribution_and_version() -> None:
    """What a user needs to act on: which package to uninstall, upgrade, or file against."""
    entry = _stub_entry(dist_name="am-cli-test-provider", version="0.1.0")
    assert describe_provider(entry) == "am-cli-test-provider 0.1.0 (pkg:verb)"


def test_describe_provider_degrades_to_the_name_alone() -> None:
    """Some installs expose a distribution name but no version; half an answer beats none."""
    assert describe_provider(_stub_entry(dist_name="pkg", version=None)) == "pkg (pkg:verb)"


def test_a_non_conforming_object_reports_the_contract_without_an_entry_point() -> None:
    """`check_subcommand` is also usable off the entry-point path, so it must still say what a
    verb owes rather than referring to a distribution it does not have."""
    with pytest.raises(InvalidSubcommandError) as excinfo:
        check_subcommand(object(), verb="orphan")
    message = str(excinfo.value)
    assert "A verb must provide: name, help, add_arguments, run." in message
    assert "entry point" not in message


def _stub_entry(*, dist_name: str, version: str | None):  # type: ignore[no-untyped-def]
    """The shape `describe_provider` reads, without installing a package to get it."""
    return SimpleNamespace(
        name="verb", value="pkg:verb", dist=SimpleNamespace(name=dist_name, version=version)
    )


def test_load_rejects_a_provider_that_does_not_conform() -> None:
    with pytest.raises(InvalidSubcommandError) as excinfo:
        load_verb(make_entry_point("bogus", "MALFORMED"))
    message = str(excinfo.value)
    assert "'name'" in message  # the first missing member, named
    assert "_verbs:MALFORMED" in message  # and where it came from


def test_load_rejects_a_provider_whose_run_is_not_callable() -> None:
    with pytest.raises(InvalidSubcommandError) as excinfo:
        load_verb(make_entry_point("inert", "NOT_CALLABLE"))
    assert "'run' is not callable" in str(excinfo.value)


def test_load_returns_the_subcommand_for_a_conforming_provider() -> None:
    assert load_verb(make_entry_point("echo", "ECHO")).name == "echo"
