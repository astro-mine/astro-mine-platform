"""Mind's verb on the umbrella CLI — `astro-mine mind`.

A **component-scoped** surface: RFC-0011 §2 reads it as `astro-mine <component> <verb>`, because
these actions only mean something in Mind's own vocabulary — unlike `score` or `train`, which a
user names directly (astro-mine/docs#57).

So this is a **passthrough** adapter: it takes the rest of the command line untouched and hands it
to :func:`astro_mine.mind.cli.main`, the same entry point `astro-mine-mind` runs. Nothing is
re-declared, which means the umbrella's surface cannot drift from Mind's real flags — add a
subcommand to Mind and `astro-mine mind` has it the same day, with no change here and none
in astro-mine-cli.

There is no `astro-mine mind run`: stepping a stack needs a Core Environment, which is
Sim's job (astro-mine-mind#25). The passthrough inherits that boundary rather than
blurring it.

**Nothing here imports the umbrella.** The contract is structural — ``name``, ``help``,
``add_arguments(parser)``, ``run(args) -> int`` — so a component is reachable from `astro-mine`
without depending on it (``conventions.md §1.1``). ``astro-mine-cli`` is not a dependency of this
package and must not become one.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from astro_mine.mind.cli import _cmd_validate, main

__all__ = ["mind", "validator"]


class _Mind:
    name = "mind"
    help = "validate and compose planner stacks"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "tail",
            nargs=argparse.REMAINDER,
            help="the astro-mine-mind command line (`astro-mine-mind --help` lists it)",
        )

    def run(self, args: argparse.Namespace) -> int:
        # SystemExit is caught and converted rather than left to propagate: argparse raises it for
        # a usage error inside Mind's own parser, and the umbrella's contract is that `run`
        # *returns* the exit status. Converting keeps one rule for every verb; the status itself is
        # unchanged, so `astro-mine mind --nonsense` still exits exactly as `astro-mine-mind` would.
        try:
            result = main(args.tail)
        except SystemExit as exit_:
            code = exit_.code
            if code is None:
                return 0
            return code if isinstance(code, int) else 1
        return int(result)


mind = _Mind()


class _MindValidator:
    """Mind's half of the federated `astro-mine validate` (RFC-0011 §6).

    Mind owns the stack spec format, so it owns the checker. The umbrella routes; it holds no
    schema knowledge of its own and reimplements nothing here — ``validate`` calls the same
    ``_cmd_validate`` that ``astro-mine-mind validate`` dispatches to, so the two surfaces cannot
    disagree about what is valid.
    """

    name = "mind"

    def claims(self, path: str) -> str | None:
        """Recognize a stack spec by its ``stack_spec_version`` key, or decline.

        Cheap and total: the umbrella asks every installed validator about every file, so a
        document that is Core's or Mind's must come back ``None`` rather than raise — at claim time
        nobody owns the file yet, and raising would turn another component's document into a
        Mind traceback. A file that *is* Mind's but malformed is claimed here and then
        reported properly by the real checker.
        """
        import yaml

        try:
            with open(path, encoding="utf-8") as handle:
                document = yaml.safe_load(handle)
        except (OSError, yaml.YAMLError):
            return None
        if isinstance(document, dict) and "stack_spec_version" in document:
            return "stack_spec"
        return None

    def validate(self, paths: Sequence[str], *, as_json: bool) -> int:
        """Run the same checker `astro-mine-mind validate` runs — not a second implementation."""
        del as_json  # Mind's checker has no JSON mode; its text report is the output
        return int(_cmd_validate(argparse.Namespace(stack=list(paths))))


validator = _MindValidator()
