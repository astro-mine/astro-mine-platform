"""Fleet's contributions to the umbrella CLI — the `astro-mine fleet` verb and the `asset` scaffold.

A **component-scoped** surface: RFC-0011 §2 reads it as `astro-mine <component> <verb>`, because
these actions only mean something in Fleet's own vocabulary — unlike `score` or `train`, which a
user names directly (astro-mine/docs#57).

So this is a **passthrough** adapter: it takes the rest of the command line untouched and hands it
to :func:`astro_mine.fleet.cli.main`, the same entry point `fleet` runs. Nothing is
re-declared, which means the umbrella's surface cannot drift from Fleet's real flags — add a
subcommand to Fleet and `astro-mine fleet` has it the same day, with no change here and none
in astro-mine-cli.

Fleet is the platform's exemplar CLI — 14 subcommands covering a whole authoring
lifecycle — and exactly the surface that would be tedious to mirror verb by verb.

**Nothing here imports the umbrella.** The contract is structural — ``name``, ``help``,
``add_arguments(parser)``, ``run(args) -> int`` — so a component is reachable from `astro-mine`
without depending on it (``conventions.md §1.1``). ``astro-mine-cli`` is not a dependency of this
package and must not become one.

**The `asset` scaffold** (:data:`asset_scaffold`) is the second contribution, added for RFC-0011 §7
— which names `fleet new` as the exemplar the umbrella generalizes. It is registered into
``astro_mine.cli.scaffolds`` and satisfies the *same* four-member contract as a verb, so there is
one shape to learn rather than two.
"""

from __future__ import annotations

import argparse

from astro_mine.fleet.cli import _cmd_new, main

__all__ = ["asset_scaffold", "fleet"]


class _Fleet:
    name = "fleet"
    help = "author and publish SADF assets"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "tail",
            nargs=argparse.REMAINDER,
            help="the fleet command line (`fleet --help` lists it)",
        )

    def run(self, args: argparse.Namespace) -> int:
        # SystemExit is caught and converted rather than left to propagate: argparse raises it for
        # a usage error inside Fleet's own parser, and the umbrella's contract is that `run`
        # *returns* the exit status. Converting keeps one rule for every verb; the status itself is
        # unchanged, so `astro-mine fleet --nonsense` still exits exactly as `fleet` would.
        try:
            main(args.tail)
        except SystemExit as exit_:
            code = exit_.code
            if code is None:
                return 0
            return code if isinstance(code, int) else 1
        return 0  # `main` returns None; it signals failure by raising SystemExit


class _AssetScaffold:
    """`astro-mine new asset <path>` — the same command as `fleet new`, reached by another road.

    RFC-0011 §7 puts the *verb* in the umbrella because scaffolding spans components, and leaves the
    *template* with whoever owns the format. SADF is Fleet's, so the bytes are written here — and
    written by :func:`astro_mine.fleet.cli._cmd_new`, the identical handler `fleet new` runs.

    **Delegating to the handler rather than re-rendering the template is the whole design.** A
    second copy of the scaffold here would be a second SADF document to keep valid as the schema
    moves, and the two would diverge silently — the umbrella's output drifting from the component's
    while both still "worked". Because this calls the same function, `astro-mine new asset` and
    `fleet new` cannot produce different bytes, and the validation `_cmd_new` already performs on
    its own output covers both paths at once.

    The argument *names* are chosen to match what that handler reads (``kind``, ``id``, ``name``,
    ``asset_version``, plus the ``output`` and ``--force`` the umbrella itself declares), so the
    parsed namespace is handed over untouched. Only the surface differs: `fleet new` takes the kind
    as a positional, and under the umbrella the first positional is already spoken for, so the kind
    becomes an option with a default. That default is what lets `astro-mine new asset rover.yaml`
    produce a valid document with nothing else typed, which is the acceptance criterion.
    """

    name = "asset"
    help = "a SADF asset (Fleet owns the format)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.description = (
            "Scaffold a minimal, valid SADF asset. The document is validated against Core's SADF "
            "gate before it is written, so `astro-mine validate` accepts it with no hand-editing."
        )
        parser.add_argument(
            "--kind",
            default="rover",
            help="asset kind label, e.g. rover, orbiter, excavator (default: rover)",
        )
        parser.add_argument("--id", help="asset identity id (default: example.<kind>)")
        parser.add_argument("--name", help="asset display name")
        parser.add_argument(
            "--asset-version", default="0.1.0", help="asset version (default: 0.1.0)"
        )

    def run(self, args: argparse.Namespace) -> int:
        return _cmd_new(args)


fleet = _Fleet()
asset_scaffold = _AssetScaffold()
