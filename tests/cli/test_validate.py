"""The federated `astro-mine validate` (RFC-0011 §6).

The umbrella owns this verb and validates nothing: it asks each installed validator *"is this
yours?"* and hands the file to whoever says yes. These tests hold the routing honest — that
ownership decides, that a document nobody claims is refused rather than guessed at, that two
claimants is an error rather than a coin flip, and that the umbrella still parses no documents and
carries no schema knowledge.
"""

from __future__ import annotations

import argparse
import sys

import pytest
from _verbs import make_entry_point

from astro_mine.cli._validate import validate
from astro_mine.cli._validators import (
    ClaimCollisionError,
    InvalidValidatorError,
    claim,
    discover_validators,
)


class _Stub:
    """A validator that owns files whose name contains its marker."""

    def __init__(self, name: str, marker: str, status: int = 0) -> None:
        self.name = name
        self.marker = marker
        self.status = status
        self.seen: list[list[str]] = []

    def claims(self, path: str) -> str | None:
        return f"{self.name}_doc" if self.marker in path else None

    def validate(self, paths, *, as_json):  # type: ignore[no-untyped-def]
        self.seen.append(list(paths))
        return self.status


def _run(paths: list[str], monkeypatch: pytest.MonkeyPatch, validators, as_json: bool = False):  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        "astro_mine.cli._validate.discover_validators", lambda: tuple(validators), raising=True
    )
    return validate.run(argparse.Namespace(file=paths, json=as_json))


def test_each_file_goes_to_its_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    core, guard = _Stub("core", "objective"), _Stub("guard", "safety")
    assert _run(["a.objective.yaml", "b.safety.yaml"], monkeypatch, [core, guard]) == 0
    assert core.seen == [["a.objective.yaml"]]
    assert guard.seen == [["b.safety.yaml"]]


def test_one_owner_is_called_once_for_all_its_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """Grouped, not per-file: a checker that can report on several documents at once should
    produce one report rather than N unrelated ones."""
    core = _Stub("core", "objective")
    _run(["a.objective.yaml", "b.objective.yaml"], monkeypatch, [core])
    assert core.seen == [["a.objective.yaml", "b.objective.yaml"]]


def test_the_worst_status_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    """One bad document must fail the command even when another owner's files are fine."""
    ok, bad = _Stub("core", "objective"), _Stub("guard", "safety", status=1)
    assert _run(["a.objective.yaml", "b.safety.yaml"], monkeypatch, [ok, bad]) == 1


def test_an_unclaimed_document_is_refused_not_guessed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The platform never validates a document against a guessed schema. The umbrella inherits
    that rule: unrecognized means "say who is installed and stop", not "try the first checker"."""
    core = _Stub("core", "objective")
    assert _run(["mystery.yaml", "a.objective.yaml"], monkeypatch, [core]) == 1
    err = capsys.readouterr().err
    assert "no installed validator recognizes mystery.yaml" in err
    assert "Installed: core" in err
    # The claimed file is still checked — one unknown document does not abandon the rest.
    assert core.seen == [["a.objective.yaml"]]


def test_two_claimants_is_a_hard_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Never a precedence rule: which checker judged a document is provenance, and a silent winner
    would make the same file validate differently depending on what else is installed."""
    first, second = _Stub("core", "spec"), _Stub("guard", "spec")
    assert _run(["ambiguous.spec.yaml"], monkeypatch, [first, second]) == 2
    err = capsys.readouterr().err
    assert "claimed by more than one validator" in err
    assert "core" in err and "guard" in err
    assert "Traceback" not in err


def test_no_validators_installed_names_what_to_install(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(["a.yaml"], monkeypatch, []) == 2
    err = capsys.readouterr().err
    assert "astro-mine-core" in err and "astro-mine-guard" in err and "astro-mine-mind" in err
    assert "Traceback" not in err


def test_a_malformed_validator_is_reported_not_raised(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def _boom() -> None:
        raise InvalidValidatorError("the validator 'broken' does not satisfy the contract")

    monkeypatch.setattr("astro_mine.cli._validate.discover_validators", _boom, raising=True)
    assert validate.run(argparse.Namespace(file=["a.yaml"], json=False)) == 2
    assert "does not satisfy" in capsys.readouterr().err


def test_claim_helper_reports_collisions_and_misses() -> None:
    core = _Stub("core", "objective")
    assert claim([core], "a.objective.yaml")[0] is core
    with pytest.raises(LookupError):
        claim([core], "mystery.yaml")
    with pytest.raises(ClaimCollisionError):
        claim([core, _Stub("guard", "objective")], "a.objective.yaml")


def test_discovery_rejects_a_validator_that_does_not_conform() -> None:
    """The same structural check the verb contract gets, with the same kind of message: name the
    entry point and the missing member, so the reader files against the right repo."""
    with pytest.raises(InvalidValidatorError) as excinfo:
        discover_validators([make_entry_point("bogus", "MALFORMED")])
    message = str(excinfo.value)
    assert "'name'" in message
    assert "_verbs:MALFORMED" in message


def test_the_umbrella_parses_no_documents() -> None:
    """The zero-dependency rule, checked where it would break first: routing must not need a YAML
    parser. If `validate` ever grows one, this package has taken a dependency it is not allowed."""
    import astro_mine.cli._validate as module

    assert "yaml" not in sys.modules or "yaml" not in dir(module)
    source = module.__file__ or ""
    assert source
    with open(source, encoding="utf-8") as handle:
        text = handle.read()
    assert "import yaml" not in text
    assert "json.load" not in text
