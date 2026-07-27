"""Guard's verb on the umbrella CLI — `astro-mine guard`.

A **component-scoped** surface: RFC-0011 §2 reads it as `astro-mine <component> <verb>`, because
these actions only mean something in Guard's own vocabulary — unlike `score` or `train`, which a
user names directly (astro-mine/docs#57).

So this is a **passthrough** adapter: it takes the rest of the command line untouched and hands it
to the same ``main`` `astro-mine-guard` runs. Nothing is
re-declared, which means the umbrella's surface cannot drift from Guard's real flags — add a
subcommand to Guard and `astro-mine guard` has it the same day, with no change here and none
in astro-mine-cli.

**Nothing here imports the umbrella.** The contract is structural — ``name``, ``help``,
``add_arguments(parser)``, ``run(args) -> int`` — so a component is reachable from `astro-mine`
without depending on it (``conventions.md §1.1``). ``astro-mine-cli`` is not a dependency of this
package and must not become one.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from astro_mine.guard.cli import _cmd_validate, main

__all__ = ["guard", "validator"]


class _Guard:
    name = "guard"
    help = "author, compile and falsify SafetySpecs"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "tail",
            nargs=argparse.REMAINDER,
            help="the astro-mine-guard command line (`astro-mine-guard --help` lists it)",
        )

    def run(self, args: argparse.Namespace) -> int:
        # SystemExit is caught and converted rather than left to propagate: argparse raises it for
        # a usage error inside Guard's own parser, and the umbrella's contract is that `run`
        # *returns* the exit status. Converting keeps one rule for every verb; the status itself is
        # unchanged, so `astro-mine guard --nonsense` exits exactly as `astro-mine-guard` would.
        try:
            result = main(args.tail)
        except SystemExit as exit_:
            code = exit_.code
            if code is None:
                return 0
            return code if isinstance(code, int) else 1
        return int(result)


guard = _Guard()


class _GuardValidator:
    """Guard's half of the federated `astro-mine validate` (RFC-0011 §6).

    Guard owns the SafetySpec format, so it owns the checker. The umbrella routes; it holds no
    schema knowledge of its own and reimplements nothing here — ``validate`` calls the same
    ``_cmd_validate`` that ``astro-mine-guard validate`` dispatches to, so the two surfaces cannot
    disagree about what is valid.
    """

    name = "guard"

    def claims(self, path: str) -> str | None:
        """Recognize a SafetySpec by its ``safety_version`` key, or decline.

        Cheap and total: the umbrella asks every installed validator about every file, so a
        document that is Core's or Mind's must come back ``None`` rather than raise — at claim time
        nobody owns the file yet, and raising would turn another component's document into a
        Guard traceback. A file that *is* Guard's but malformed is claimed here and then
        reported properly by the real checker.
        """
        import yaml

        try:
            with open(path, encoding="utf-8") as handle:
                document = yaml.safe_load(handle)
        except (OSError, yaml.YAMLError):
            return None
        if isinstance(document, dict) and "safety_version" in document:
            return "safety_spec"
        return None

    def validate(self, paths: Sequence[str], *, as_json: bool) -> int:
        """Run the same checker `astro-mine-guard validate` runs — not a second implementation."""
        del as_json  # Guard's checker has no JSON mode; its text report is the output
        return int(_cmd_validate(argparse.Namespace(spec=list(paths))))


validator = _GuardValidator()
