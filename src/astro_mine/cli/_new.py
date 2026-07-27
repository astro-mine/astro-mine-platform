"""The built-in scaffolding verbs — `astro-mine new` and `astro-mine plugin new` (RFC-0011 §7).

Both are **routers**, like `validate` and for the same reason: the thing being written belongs to a
component, but deciding *which* component belongs to nobody but the umbrella. `astro-mine new
asset` is Fleet's document; `astro-mine plugin new solver` is Allocate's extension group; neither
package can host the verb without knowing about the other seven.

**A second turn of the same crank.** The top level parses only *which verb* and leaves the tail
alone so it can import one component instead of all of them
(:mod:`astro_mine.cli._dispatch`). These verbs repeat that exactly one level down: parse only
*which kind*, load that one scaffold, and let it parse its own tail. `astro-mine new` with no kind
lists what is available without importing anything at all.

**The umbrella owns two arguments, and only two.** It declares ``output`` and ``--force`` before
handing the parser to a scaffold, so `astro-mine new <anything>` has the same skeleton and a user
who has scaffolded one kind can scaffold the next without re-reading the help. Everything else is
the owner's to declare; a scaffold must not re-declare those two.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from importlib.metadata import EntryPoint, PackageNotFoundError, version
from types import MappingProxyType

from astro_mine.cli._discovery import describe_provider
from astro_mine.cli._manifest import (
    FIRST_PARTY_KINDS,
    FIRST_PARTY_PLUGIN_KINDS,
    FirstPartyKind,
)
from astro_mine.cli._protocol import InvalidSubcommandError, Subcommand
from astro_mine.cli._scaffolds import (
    DOCUMENT_SCAFFOLD_GROUP,
    PLUGIN_SCAFFOLD_GROUP,
    ScaffoldCollisionError,
    discover_scaffolds,
    load_scaffold,
)
from astro_mine.cli._templates import CLI_PLUGIN_SCAFFOLD

__all__ = ["new", "plugin"]

_USAGE_ERROR = 2

#: What a user types to ask for help. Spelled out because `plugin new`'s tail is an
#: :data:`argparse.REMAINDER`, which hands these through verbatim instead of acting on them.
_HELP_FLAGS = frozenset({"-h", "--help"})

#: `new` has no built-in kinds — every document belongs to a component. Only `plugin new` does,
#: and only one (see :mod:`astro_mine.cli._templates`).
_NO_BUILTINS: Mapping[str, Subcommand] = MappingProxyType({})


def _is_installed(distribution: str) -> bool:
    """Is this distribution present? A metadata read — no import, so the listing stays free."""
    try:
        version(distribution)
    except PackageNotFoundError:
        return False
    return True


class _Scaffolder:
    """The shared body of both verbs: list kinds, or route one kind to its owner.

    Parameterized by group rather than duplicated, because the two verbs differ in exactly three
    things — the group they read, the table they degrade from, and the words in their messages.
    """

    def __init__(
        self,
        *,
        command: str,
        group: str,
        table: Mapping[str, FirstPartyKind],
        builtins: Mapping[str, Subcommand] = _NO_BUILTINS,
        noun: str = "kind",
    ) -> None:
        self.command = command
        self.group = group
        self.table = table
        self.builtins = builtins
        self.noun = noun

    def dispatch(self, kind: str, rest: list[str]) -> int:
        """Load the one scaffold that owns ``kind`` and hand it the rest of the command line."""
        try:
            discovered = discover_scaffolds(self.group)
        except ScaffoldCollisionError as exc:
            print(f"astro-mine {self.command}: {exc}", file=sys.stderr)
            return _USAGE_ERROR

        shadowed = sorted(set(discovered) & set(self.builtins))
        if shadowed:
            names = ", ".join(f"{k!r} ({describe_provider(discovered[k])})" for k in shadowed)
            print(
                f"astro-mine {self.command}: {names} shadows a {self.noun} the umbrella owns; "
                f"uninstall the package or ask it to rename its `{self.group}` entry point",
                file=sys.stderr,
            )
            return _USAGE_ERROR

        scaffold = self.builtins.get(kind)
        if scaffold is None:
            entry = discovered.get(kind)
            if entry is None:
                return self._report_missing(kind, discovered)
            try:
                scaffold = load_scaffold(entry, group=self.group)
            except InvalidSubcommandError as exc:
                print(f"astro-mine {self.command}: {exc}", file=sys.stderr)
                return _USAGE_ERROR

        parser = argparse.ArgumentParser(
            prog=f"astro-mine {self.command} {kind}", description=scaffold.help
        )
        parser.add_argument("output", help="path to write to")
        parser.add_argument("--force", action="store_true", help="overwrite what is already there")
        scaffold.add_arguments(parser)
        status = scaffold.run(parser.parse_args(rest))
        return 0 if status is None else int(status)

    def listing(self) -> str:
        """What `astro-mine <command>` prints with no kind — built without importing anything."""
        discovered = discover_scaffolds(self.group)
        available = sorted({*discovered, *self.builtins})
        absent = sorted(k for k in self.table if k not in discovered and k not in self.builtins)
        width = max((len(k) for k in (*available, *absent)), default=0) + 2
        lines = [f"usage: astro-mine {self.command} <{self.noun}> <output> [options]", ""]

        if available:
            lines.append(f"{self.noun.title()}s:")
            lines += [f"  {kind:<{width}}{self._summarize(kind, discovered)}" for kind in available]
        else:
            lines.append(f"No {self.noun}s are available in this environment yet.")

        # The same split the missing-kind path makes, and for the same reason: a kind whose owner is
        # installed but silent is not "from a component that is not installed here". Saying so would
        # be the listing contradicting what running the command tells you thirty seconds later, and
        # sending the user to install what they already have.
        uninstalled = [kind for kind in absent if not _is_installed(self.table[kind].distribution)]
        unoffered = [kind for kind in absent if kind not in uninstalled]

        for heading, kinds in (
            ("Available from components that are not installed here:", uninstalled),
            ("Known, but the component that owns them offers no scaffold yet:", unoffered),
        ):
            if kinds:
                lines += [
                    "",
                    heading,
                    *(
                        f"  {kind:<{width}}{self.table[kind].help} "
                        f"[{self.table[kind].distribution}]"
                        for kind in kinds
                    ),
                ]
        lines += ["", f"`astro-mine {self.command} <{self.noun}> --help` shows its own options."]
        return "\n".join(lines)

    def _summarize(self, kind: str, discovered: Mapping[str, EntryPoint]) -> str:
        """One line about an available kind — never from the scaffold: that would cost an import."""
        builtin = self.builtins.get(kind)
        if builtin is not None:
            return builtin.help
        known = self.table.get(kind)
        if known is not None:
            return known.help
        return f"provided by {describe_provider(discovered[kind])}"

    def _report_missing(self, kind: str, discovered: Mapping[str, EntryPoint]) -> int:
        """Say which of the two things went wrong — they have different fixes.

        A kind the platform does not have is a typo. A kind whose owner is absent is an install.
        And a kind whose owner is *present but offers no scaffold* is neither: telling that user to
        `pip install astro-mine-worlds` when they already have it would be the umbrella lying about
        an environment it can see. The probe is a metadata read, so distinguishing the two costs
        nothing and imports nothing.
        """
        known = self.table.get(kind)
        if known is None:
            available = (
                ", ".join(sorted({*discovered, *self.builtins})) or "none in this environment"
            )
            print(
                f"astro-mine {self.command}: unknown {self.noun} {kind!r} (available: {available})",
                file=sys.stderr,
            )
            return _USAGE_ERROR
        try:
            installed = version(known.distribution)
        except PackageNotFoundError:
            print(
                f"`astro-mine {self.command} {kind}` needs {known.distribution} — install it with "
                f"`pip install {known.distribution}` (or `uv add {known.distribution}`), "
                f"then re-run.",
                file=sys.stderr,
            )
        else:
            print(
                f"`astro-mine {self.command} {kind}` needs a {kind!r} scaffold from "
                f"{known.distribution}, but {known.distribution} {installed} is installed and "
                f"offers none — that component does not support scaffolding this yet. Its own CLI "
                f"(`astro-mine-{known.distribution.removeprefix('astro-mine-')} --help`) may still "
                f"be able to author one.",
                file=sys.stderr,
            )
        return _USAGE_ERROR


class _New:
    name = "new"
    help = "scaffold an authored document (routed to the format's owner)"

    _scaffolder = _Scaffolder(command="new", group=DOCUMENT_SCAFFOLD_GROUP, table=FIRST_PARTY_KINDS)

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.description = (
            "Scaffold an authored document. Each kind is written by the component that owns its "
            "format; this command owns no schema of its own (RFC-0011 §7). What it writes is "
            "valid on arrival — `astro-mine validate` accepts it with no hand-editing."
        )
        parser.add_argument("kind", nargs="?", help="what to scaffold; omit to list the kinds")
        parser.add_argument(
            "rest",
            nargs=argparse.REMAINDER,
            help="arguments for the kind (`astro-mine new <kind> --help` for its own help)",
        )

    def run(self, args: argparse.Namespace) -> int:
        if args.kind is None:
            print(self._scaffolder.listing())
            return 0
        return self._scaffolder.dispatch(args.kind, args.rest)


class _Plugin:
    """`astro-mine plugin new <kind>` — one verb with one action, per RFC-0011 §2's surface.

    ``new`` is spelled out rather than folded into `astro-mine new`, because a plugin and a
    document are different things: one is an installable distribution that extends the platform,
    the other is a file the platform reads. Collapsing them would make `astro-mine new solver`
    write a Python package while `astro-mine new asset` writes YAML, from the same word.
    """

    name = "plugin"
    help = "scaffold a plugin package (`plugin new <kind>`)"

    _scaffolder = _Scaffolder(
        command="plugin new",
        group=PLUGIN_SCAFFOLD_GROUP,
        table=FIRST_PARTY_PLUGIN_KINDS,
        builtins={CLI_PLUGIN_SCAFFOLD.name: CLI_PLUGIN_SCAFFOLD},
    )

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.description = (
            "Author a plugin against one of the platform's live extension groups. The recipes "
            "these scaffolds emit are the ones in the plugin-authoring guide "
            "(guide/how-to/write-a-plugin.md)."
        )
        parser.add_argument("action", nargs="?", help="`new` — scaffold a plugin package")
        parser.add_argument("rest", nargs=argparse.REMAINDER, help="`<kind> <output> [options]`")

    def run(self, args: argparse.Namespace) -> int:
        if args.action is None:
            print(self._scaffolder.listing())
            return 0
        if args.action != "new":
            print(
                f"astro-mine plugin: unknown action {args.action!r} (available: new)",
                file=sys.stderr,
            )
            return _USAGE_ERROR
        # `--help` is claimed here rather than by argparse, because REMAINDER stops option
        # processing: without this the flag arrives as the kind positional and comes back as
        # `unknown kind '--help'`, which reads as the tool being confused about the most standard
        # flag there is. `astro-mine new --help` prints help and exits 0; so does this.
        if not args.rest or args.rest[0] in _HELP_FLAGS:
            print(self._scaffolder.listing())
            return 0
        return self._scaffolder.dispatch(args.rest[0], args.rest[1:])


new = _New()
plugin = _Plugin()
