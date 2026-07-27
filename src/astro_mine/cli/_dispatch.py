"""The top-level parser and the dispatch loop.

**Why parsing happens in two phases.** The umbrella cannot build a complete argparse tree up
front: filling in one verb's arguments means calling that verb's ``add_arguments``, which means
importing its component — so a single-phase parser would import every installed component just to
show ``--help``. Instead, phase one parses only *which verb* (everything after it is
:data:`argparse.REMAINDER`), and phase two loads that one verb and lets it parse its own tail.
The user pays for the import of the command they actually ran, and for nothing else
(RFC-0011 §1a).

The cost of this design is that the top-level ``--help`` cannot show a verb's own ``help`` string.
That is what :mod:`astro_mine.cli._manifest` is for: first-party descriptions are static strings,
so the listing stays free, and a verb's complete help comes from the provider on
``astro-mine <verb> --help``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from importlib.metadata import EntryPoint

from astro_mine.cli._discovery import (
    VerbCollisionError,
    describe_provider,
    discover_verbs,
    load_verb,
)
from astro_mine.cli._manifest import FIRST_PARTY_VERBS, install_hint
from astro_mine.cli._new import new as _new_verb
from astro_mine.cli._new import plugin as _plugin_verb
from astro_mine.cli._protocol import InvalidSubcommandError, Subcommand
from astro_mine.cli._validate import validate as _validate_verb

__all__ = ["build_parser", "main"]

_DESCRIPTION = "The Astro-Mine umbrella CLI — one front door to the platform's component CLIs."

#: Exit status for a usage error, matching argparse's own convention. A verb whose component is
#: not installed uses it too: the command was well-formed but cannot run in this environment, and
#: conflating that with the verb's *own* failure codes (which start at 1) would make a script
#: unable to tell "I am misconfigured" from "the run failed".
_USAGE_ERROR = 2

#: Verbs the umbrella owns itself. Each is here for a reason rather than for convenience: all three
#: *route* rather than do, and routing is the one job no component can hold without importing its
#: siblings (`conventions.md §1.1`). `validate` sends a document to whoever owns its format
#: (RFC-0011 §6; see _validate.py); `new` and `plugin new` send a scaffold request to whoever owns
#: the kind (RFC-0011 §7; see _new.py). Nothing else belongs here — a verb that *does* something
#: belongs to the component that does it.
#:
#: Built-ins are seeded like Allocate's solver registry seeds CP-SAT — and collide the same way: a
#: distribution advertising a verb that shadows one is a hard error naming both, never a silent
#: winner. Importing this module costs nothing external; only running a verb loads a provider.
_BUILTIN_VERBS: dict[str, Subcommand] = {
    verb.name: verb for verb in (_validate_verb, _new_verb, _plugin_verb)
}


def build_parser(verbs: Mapping[str, EntryPoint] | None = None) -> argparse.ArgumentParser:
    """Build the phase-one parser: the verb, and everything after it, untouched.

    ``verbs`` is injectable for tests; ``None`` reads the installed environment. Built per call,
    never cached, so the verb set always reflects what is installed now.
    """
    discovered = discover_verbs() if verbs is None else verbs
    parser = argparse.ArgumentParser(
        prog="astro-mine",
        description=_DESCRIPTION,
        epilog=_format_verbs(discovered),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=True,
    )
    from astro_mine.cli import __version__

    parser.add_argument("--version", action="version", version=f"astro-mine {__version__}")
    parser.add_argument("verb", nargs="?", help="the action to run; see the list below")
    # REMAINDER, not a subparser tree: the tail belongs to the verb's own parser, which does not
    # exist until the verb is loaded. It also means the umbrella never has to mirror, and drift
    # from, a component's flags.
    parser.add_argument(
        "rest",
        nargs=argparse.REMAINDER,
        help="arguments for the verb (`astro-mine <verb> --help` for its own help)",
    )
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    verbs: Mapping[str, EntryPoint] | None = None,
) -> int:
    """Run the umbrella. Returns the process exit status.

    A bare ``astro-mine`` prints help and exits **0** — the user asked a dispatcher what it can
    do, and the listing is a complete answer whether or not anything is installed.

    A broken *environment* — two packages claiming one verb, or a provider that does not satisfy
    the contract — is reported as a message and a non-zero status, never as a traceback
    (RFC-0011 §4). Both are somebody else's packaging bug, and a stack trace through this
    package's internals would point every reader at the wrong repo. A failure *inside* a verb is
    not caught: that belongs to the component, and swallowing it would hide real errors.
    """
    try:
        discovered = discover_verbs() if verbs is None else verbs
    except VerbCollisionError as exc:
        print(f"astro-mine: {exc}", file=sys.stderr)
        return _USAGE_ERROR

    shadowed = sorted(set(discovered) & set(_BUILTIN_VERBS))
    if shadowed:
        names = ", ".join(f"{v!r} ({describe_provider(discovered[v])})" for v in shadowed)
        print(
            f"astro-mine: {names} shadows a built-in verb the umbrella owns; uninstall the "
            f"package or ask it to rename its `astro_mine.cli` entry point",
            file=sys.stderr,
        )
        return _USAGE_ERROR

    parser = build_parser(discovered)
    args = parser.parse_args(argv)

    if args.verb is None:
        parser.print_help()
        return 0

    builtin = _BUILTIN_VERBS.get(args.verb)
    if builtin is not None:
        subcommand: Subcommand = builtin
    else:
        entry = discovered.get(args.verb)
        if entry is None:
            return _report_missing(parser, args.verb, discovered)
        try:
            subcommand = load_verb(entry)
        except InvalidSubcommandError as exc:
            print(f"astro-mine: {exc}", file=sys.stderr)
            return _USAGE_ERROR
    sub = argparse.ArgumentParser(
        prog=f"astro-mine {args.verb}",
        description=subcommand.help,
    )
    subcommand.add_arguments(sub)
    status = subcommand.run(sub.parse_args(args.rest))
    # `None` is the near-universal Python convention for "finished, no error" (it is what
    # sys.exit(None) means), and a component that ran successfully should not be punished with a
    # crash for following it.
    return 0 if status is None else int(status)


def _report_missing(
    parser: argparse.ArgumentParser, verb: str, discovered: Mapping[str, EntryPoint]
) -> int:
    """A verb that did not resolve: name the fix if we know it, else fail like argparse.

    The split matters. A *known* platform verb whose component is absent is a missing install and
    has an exact remedy, so printing "unknown command" would send the user looking for a typo they
    did not make (RFC-0011 §4). A verb nobody advertises really is unknown.
    """
    hint = install_hint(verb)
    if hint is None:
        available = ", ".join(sorted(discovered)) or "none in this environment"
        parser.error(f"unknown verb {verb!r} (available: {available})")  # exits _USAGE_ERROR
    print(hint, file=sys.stderr)
    return _USAGE_ERROR


def _format_verbs(discovered: Mapping[str, EntryPoint]) -> str:
    """The verb listing shown under ``--help``, built without importing anything.

    Uninstalled first-party verbs are listed too, and marked. On a bare install that turns
    ``astro-mine --help`` into a map of the platform rather than an empty shell — which is the
    discovery problem (**UC-A3**) this package exists to solve — as long as it never implies they
    are runnable here.
    """
    installed = sorted({*discovered, *_BUILTIN_VERBS})
    absent = sorted(v for v in FIRST_PARTY_VERBS if v not in discovered and v not in _BUILTIN_VERBS)
    width = max((len(v) for v in (*installed, *absent)), default=0) + 2
    lines: list[str] = []

    if installed:
        lines.append("Verbs:")
        lines += [
            f"  {verb:<{width}}{_summarize(verb, discovered.get(verb))}" for verb in installed
        ]
    else:
        lines.append("No verbs are registered in this environment yet.")

    if absent:
        lines += [
            "",
            "Available from components that are not installed here:",
            *(
                f"  {verb:<{width}}{FIRST_PARTY_VERBS[verb].help} "
                f"[{FIRST_PARTY_VERBS[verb].distribution}]"
                for verb in absent
            ),
        ]

    lines += [
        "",
        "Every component CLI also works directly (`astro-mine-bench score`, `fleet validate`).",
        "`astro-mine <verb> --help` shows a verb's own options.",
    ]
    return "\n".join(lines)


def _summarize(verb: str, entry: EntryPoint | None) -> str:
    """One line about an installed verb — from the manifest, or from its metadata.

    Never from the provider: reading ``Subcommand.help`` here would import every installed
    component to render a help screen, which is the exact cost this design refuses to pay.
    """
    builtin = _BUILTIN_VERBS.get(verb)
    if builtin is not None:
        return builtin.help
    known = FIRST_PARTY_VERBS.get(verb)
    if known is not None:
        return known.help
    assert entry is not None  # a non-built-in verb always came from an entry point
    return f"provided by {describe_provider(entry)}"
