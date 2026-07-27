"""`astro-mine new` — the federated document scaffolder (RFC-0011 §7).

The umbrella owns the verb and writes no documents: it asks who owns the kind the user typed and
hands over the rest of the command line. These tests hold that division honest — that routing is by
the name the user typed rather than by anything loaded, that the two arguments the umbrella owns are
always there, that a kind nobody offers is refused rather than guessed at, and that the three ways
an environment can be wrong (absent component, silent component, two claimants) each produce a
message a reader can act on instead of a traceback.

The `plugin new` half of the same engine is exercised in `test_plugin_new.py`; what lives here is
the routing shared by both.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest
from _verbs import make_entry_point

from astro_mine.cli import DOCUMENT_SCAFFOLD_GROUP
from astro_mine.cli._manifest import FIRST_PARTY_KINDS, FirstPartyKind
from astro_mine.cli._new import new
from astro_mine.cli._protocol import REQUIRED_MEMBERS
from astro_mine.cli._scaffolds import ScaffoldCollisionError, discover_scaffolds


class _Stub:
    """A scaffold that records the arguments the umbrella built for it."""

    def __init__(self, name: str, status: int | None = 0) -> None:
        self.name = name
        self.help = f"scaffold a {name}"
        self.status = status
        self.seen: list[argparse.Namespace] = []

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--flavour", default="plain")

    def run(self, args: argparse.Namespace) -> int | None:
        self.seen.append(args)
        return self.status


def _route(monkeypatch: pytest.MonkeyPatch, scaffolds: dict[str, object], argv: list[str]) -> int:
    """Run `astro-mine new <argv>` against an injected environment."""
    entries = {name: make_entry_point(name, "ECHO", DOCUMENT_SCAFFOLD_GROUP) for name in scaffolds}
    monkeypatch.setattr("astro_mine.cli._new.discover_scaffolds", lambda group: entries)
    monkeypatch.setattr(
        "astro_mine.cli._new.load_scaffold", lambda entry, *, group: scaffolds[entry.name]
    )
    kind = argv[0] if argv else None
    return new.run(argparse.Namespace(kind=kind, rest=argv[1:]))


def test_the_kind_the_user_typed_decides_the_owner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Routing is by name, not by inspection: unlike `validate`, nothing has to be read to know
    who owns an `asset` — which is what keeps listing free of imports."""
    asset, stack = _Stub("asset"), _Stub("stack")
    assert _route(monkeypatch, {"asset": asset, "stack": stack}, ["asset", "out.yaml"]) == 0
    assert [args.output for args in asset.seen] == ["out.yaml"]
    assert stack.seen == []


def test_the_umbrella_declares_output_and_force_for_every_kind(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The consistency promise: a user who has scaffolded one kind can scaffold the next without
    re-reading the help. A scaffold adds only what is specific to it."""
    asset = _Stub("asset")
    assert _route(monkeypatch, {"asset": asset}, ["asset", "a.yaml", "--force"]) == 0
    (args,) = asset.seen
    assert (args.output, args.force, args.flavour) == ("a.yaml", True, "plain")


def test_a_scaffolds_exit_status_is_the_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    assert _route(monkeypatch, {"asset": _Stub("asset", status=3)}, ["asset", "a.yaml"]) == 3


def test_a_scaffold_returning_none_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    """`sys.exit(None)` means success everywhere else in Python; a scaffold that followed the
    convention should not be punished for it. Same rule as a verb."""
    assert _route(monkeypatch, {"asset": _Stub("asset", status=None)}, ["asset", "a.yaml"]) == 0


def test_bare_new_lists_the_kinds_and_succeeds(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Asking a dispatcher what it can write is a legitimate question with a complete answer —
    the same product decision bare `astro-mine` makes."""
    monkeypatch.setattr("astro_mine.cli._new.discover_scaffolds", lambda group: {})
    assert new.run(argparse.Namespace(kind=None, rest=[])) == 0
    out = capsys.readouterr().out
    assert "usage: astro-mine new <kind> <output>" in out
    # Nothing installed, so every first-party kind is listed as absent, with what to install.
    assert "asset" in out and "astro-mine-fleet" in out
    assert "not installed here" in out


def test_a_third_party_kind_is_listed_from_its_distribution(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A kind the manifest has never heard of is still listed — described from its distribution
    metadata, which is free, rather than from its scaffold, which would cost an import. This is the
    no-PR-to-extend rule showing up in the help text."""
    entries = {
        name: make_entry_point(name, "ECHO", DOCUMENT_SCAFFOLD_GROUP)
        for name in ("asset", "demo-doc")
    }
    monkeypatch.setattr("astro_mine.cli._new.discover_scaffolds", lambda group: entries)
    assert new.run(argparse.Namespace(kind=None, rest=[])) == 0
    out = capsys.readouterr().out
    assert "demo-doc" in out
    assert "provided by" in out
    # A first-party kind that *is* installed is described from the static table — the same reason
    # the top-level verb listing is: reading a scaffold's `help` would mean loading it.
    assert FIRST_PARTY_KINDS["asset"].help in out


def test_the_verb_parses_only_the_kind_and_leaves_the_tail_alone() -> None:
    """The two-phase parse, one level down (RFC-0011 §1a). If this parser ever grew a subparser
    tree, filling it in would mean importing every installed component to render `new --help` —
    the exact cost the top level refuses to pay."""
    parser = argparse.ArgumentParser(prog="astro-mine new")
    new.add_arguments(parser)
    args = parser.parse_args(["asset", "out.yaml", "--force", "--flavour", "rich"])
    assert args.kind == "asset"
    assert args.rest == ["out.yaml", "--force", "--flavour", "rich"]


def test_an_unknown_kind_is_refused_and_lists_what_exists(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _route(monkeypatch, {"asset": _Stub("asset")}, ["nonsense", "out.yaml"]) == 2
    err = capsys.readouterr().err
    assert "unknown kind 'nonsense'" in err
    assert "available: asset" in err
    assert "Traceback" not in err


def test_a_kind_whose_component_is_absent_names_the_install(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The degradation contract (RFC-0011 §4) applied to kinds: a first-party kind whose owner is
    not installed has an exact remedy, and printing "unknown kind" would send the user looking for
    a typo they did not make."""
    assert _route(monkeypatch, {}, ["asset", "out.yaml"]) == 2
    err = capsys.readouterr().err
    assert "needs astro-mine-fleet" in err
    assert "pip install astro-mine-fleet" in err
    assert "Traceback" not in err


def test_an_installed_component_that_offers_no_scaffold_is_not_told_to_install_itself(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The third case, and the reason the missing-kind path probes at all.

    Telling a user who *has* a component installed to `pip install` it would be the umbrella lying
    about an environment it can see. The case that motivated this was `new world` while Worlds
    still registered no scaffold (G2.11, since closed); the rule outlives it, because any component
    can be present while offering no scaffold for a kind the table promises.

    The table entry is repointed at this package because it is the one distribution guaranteed to
    be installed while these tests run — the assertion is about the *behaviour*, not about which
    component happens to be lagging today.
    """
    assert FIRST_PARTY_KINDS["world"].distribution == "astro-mine-worlds"
    monkeypatch.setattr(
        new._scaffolder, "table", {"world": FirstPartyKind("astro-mine-platform", "a WorldSpec")}
    )
    assert _route(monkeypatch, {}, ["world", "out.yaml"]) == 2
    err = capsys.readouterr().err
    assert "is installed and offers none" in err
    assert "pip install" not in err
    assert "Traceback" not in err


def test_the_listing_makes_the_same_distinction_the_error_does(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A listing that filed an installed-but-silent component under "not installed here" would
    contradict what running the command tells the user thirty seconds later — and send them to
    install what they already have. Found in a real nine-component environment, where `world` was
    listed as absent while Worlds was installed.
    """
    monkeypatch.setattr(
        new._scaffolder,
        "table",
        {
            # Installed (this package always is, while these tests run) but offering no scaffold.
            "world": FirstPartyKind("astro-mine-platform", "a WorldSpec"),
            # Genuinely absent.
            "asset": FirstPartyKind("astro-mine-fleet", "a SADF asset"),
        },
    )
    monkeypatch.setattr("astro_mine.cli._new.discover_scaffolds", lambda group: {})
    assert new.run(argparse.Namespace(kind=None, rest=[])) == 0
    out = capsys.readouterr().out

    not_installed, _, no_scaffold = out.partition("offers no scaffold yet")
    assert "asset" in not_installed
    assert (
        "world" in no_scaffold
        or "world" in out.split("not installed here")[1].split("offers no scaffold yet")[1]
    )


def test_two_distributions_offering_one_kind_is_a_hard_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Never a precedence rule, for the reason a verb collision is not: which package generated a
    user's starting file is provenance, and a silent winner bakes the divergence into whatever
    they build on top of it."""

    def _collide(group: str) -> dict[str, object]:
        raise ScaffoldCollisionError("the scaffold kind 'asset' is offered by both A and B")

    monkeypatch.setattr("astro_mine.cli._new.discover_scaffolds", _collide)
    assert new.run(argparse.Namespace(kind="asset", rest=["out.yaml"])) == 2
    err = capsys.readouterr().err
    assert "offered by both A and B" in err
    assert "Traceback" not in err


def test_a_malformed_scaffold_is_reported_as_a_packaging_bug(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Named by entry point and missing member, and blamed on the right repo — never an
    AttributeError out of the umbrella's own frames."""
    entries = {"asset": make_entry_point("asset", "MALFORMED", DOCUMENT_SCAFFOLD_GROUP)}
    monkeypatch.setattr("astro_mine.cli._new.discover_scaffolds", lambda group: entries)
    assert new.run(argparse.Namespace(kind="asset", rest=["out.yaml"])) == 2
    err = capsys.readouterr().err
    assert "does not satisfy the astro_mine.cli.scaffolds contract" in err
    assert "'name'" in err
    assert "A scaffold must provide" in err
    assert "Traceback" not in err


def test_discovery_reads_metadata_without_loading_anything() -> None:
    """The laziness guarantee, one level down. This entry point could never load — the attribute
    does not exist — and discovery is untroubled by that, because it only reads names."""
    entries = [make_entry_point("ghost", "NOT_A_REAL_ATTRIBUTE", DOCUMENT_SCAFFOLD_GROUP)]
    assert set(discover_scaffolds(DOCUMENT_SCAFFOLD_GROUP, entries)) == {"ghost"}


def test_discovery_rejects_a_second_claimant() -> None:
    entries = [
        make_entry_point("asset", "ECHO", DOCUMENT_SCAFFOLD_GROUP),
        make_entry_point("asset", "RETURNS_NONE", DOCUMENT_SCAFFOLD_GROUP),
    ]
    with pytest.raises(ScaffoldCollisionError) as excinfo:
        discover_scaffolds(DOCUMENT_SCAFFOLD_GROUP, entries)
    assert "astro_mine.cli.scaffolds" in str(excinfo.value)


def test_the_scaffold_contract_is_the_subcommand_contract() -> None:
    """Not an accident worth preserving by luck: RFC-0011 §7 gets a third entry-point group, and
    giving it a fourth-member protocol of its own would have meant two shapes for component authors
    to learn and two checkers here to keep in step. Pinned so a future edit has to mean it."""
    assert REQUIRED_MEMBERS == ("name", "help", "add_arguments", "run")
    for member in REQUIRED_MEMBERS:
        assert hasattr(_Stub("asset"), member)


def test_the_umbrella_writes_no_documents_itself() -> None:
    """The rule this whole design exists to hold: the owner owns the artifact. If `new` ever grows
    a template of its own, a component's schema has been duplicated here — and this package has
    started down the road to the templating dependency it is not allowed to have."""
    source = Path(__file__).resolve().parents[2] / "src" / "astro_mine" / "cli" / "_new.py"
    text = source.read_text(encoding="utf-8")
    assert "write_text" not in text
    assert "import yaml" not in text
