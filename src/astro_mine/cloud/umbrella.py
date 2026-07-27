"""Cloud's verb on the umbrella CLI — `astro-mine cloud`.

A **component-scoped** surface: RFC-0011 §2 reads it as `astro-mine <component> <verb>`, because
these actions only mean something in Cloud's own vocabulary — unlike `score` or `train`, which a
user names directly (astro-mine/docs#57).

So this is a **passthrough** adapter: it takes the rest of the command line untouched and hands it
to the same ``main`` `astro-mine-cloud` runs. Nothing is
re-declared, which means the umbrella's surface cannot drift from Cloud's real flags — add a
subcommand to Cloud and `astro-mine cloud` has it the same day, with no change here and none
in astro-mine-cli.

**Nothing here imports the umbrella.** The contract is structural — ``name``, ``help``,
``add_arguments(parser)``, ``run(args) -> int`` — so a component is reachable from `astro-mine`
without depending on it (``conventions.md §1.1``). ``astro-mine-cli`` is not a dependency of this
package and must not become one.
"""

from __future__ import annotations

import argparse

from astro_mine.cloud.submission.cli import main

__all__ = ["cloud"]


class _Cloud:
    name = "cloud"
    help = "submit and manage cluster jobs"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "tail",
            nargs=argparse.REMAINDER,
            help="the astro-mine-cloud command line (`astro-mine-cloud --help` lists it)",
        )

    def run(self, args: argparse.Namespace) -> int:
        # SystemExit is caught and converted rather than left to propagate: argparse raises it for
        # a usage error inside Cloud's own parser, and the umbrella's contract is that `run`
        # *returns* the exit status. Converting keeps one rule for every verb; the status itself is
        # unchanged, so `astro-mine cloud --nonsense` exits exactly as `astro-mine-cloud` would.
        try:
            result = main(args.tail)
        except SystemExit as exit_:
            code = exit_.code
            if code is None:
                return 0
            return code if isinstance(code, int) else 1
        return int(result)


cloud = _Cloud()
