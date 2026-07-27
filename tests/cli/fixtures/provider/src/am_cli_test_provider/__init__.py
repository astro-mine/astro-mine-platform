"""A third-party distribution that contributes `astro-mine` verbs.

Installed into a throwaway virtualenv by ``tests/test_installed_provider.py``. It exists to prove
two things no in-process test can: that a package with **no relationship to this org** gains an
``astro-mine <verb>`` purely by declaring an entry point, and that listing verbs does not import
it (a module-level marker makes the negative observable in a clean interpreter).

It also carries a worked example of each adapter style, so a component author has something to
copy rather than a paragraph to interpret.

Since RFC-0011 §7 it proves the same thing about **scaffolds**: this distribution contributes an
`astro-mine new <kind>` and an `astro-mine plugin new <kind>` with no change to the umbrella, which
is the acceptance criterion that no first-party scaffold can honestly test — every component in the
platform is one this repo could have special-cased.
"""

from __future__ import annotations

import argparse
from pathlib import Path

__all__ = [
    "MALFORMED",
    "component_main",
    "demo",
    "doc_scaffold",
    "passthrough",
    "plugin_scaffold",
]


class _PerVerb:
    """**Style 1 — per-verb.** The verb owns its own arguments.

    The natural fit when a component wants an action promoted to the top level
    (``astro-mine score``). It costs the component one small adapter per verb, factored out of
    whatever its own parser already does.
    """

    name = "demo"
    help = "a per-verb subcommand, for testing the contract"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("message", nargs="?", default="hello")
        parser.add_argument("--shout", action="store_true")
        parser.add_argument("--exit-code", type=int, default=0)

    def run(self, args: argparse.Namespace) -> int:
        print(args.message.upper() if args.shout else args.message)
        return int(args.exit_code)


class _Passthrough:
    """**Style 2 — component passthrough.** The tail goes to the component's own ``main``.

    RFC-0011 §2's ``astro-mine <component> <verb>`` form, and the cheap on-ramp: every
    Astro-Mine CLI already exposes ``main(argv) -> int``, so the adapter is this class with the
    component's own entry point substituted. Nothing is re-declared, so the umbrella's help can
    never drift from the component's real flags.
    """

    name = "passthrough"
    help = "forwards its tail to a component CLI's own main()"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("tail", nargs=argparse.REMAINDER)

    def run(self, args: argparse.Namespace) -> int:
        return component_main(args.tail)


def component_main(argv: list[str]) -> int:
    """Stands in for a real component's ``main(argv) -> int``."""
    parser = argparse.ArgumentParser(prog="component")
    parser.add_argument("subcommand")
    parser.add_argument("--flag", default="unset")
    args = parser.parse_args(argv)
    print(f"component ran {args.subcommand} with flag={args.flag}")
    return 0


class _DocScaffold:
    """**A document kind** — what a component registers into ``astro_mine.cli.scaffolds``.

    Note what is *not* here: ``output`` and ``--force`` are declared by the umbrella before this
    object is handed the parser, so every kind has the same skeleton and a scaffold only declares
    what is specific to it. A real one writes a document its own validator accepts; this one writes
    something recognizable instead, because the point being proved is the routing.
    """

    name = "demo-doc"
    help = "a third-party document kind, for testing the scaffold contract"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--marker", default="third-party")

    def run(self, args: argparse.Namespace) -> int:
        path = Path(args.output)
        if path.exists() and not args.force:
            print(f"{path}: file exists (use --force to overwrite)")
            return 1
        path.write_text(f"kind: demo-doc\nmarker: {args.marker}\n", encoding="utf-8")
        print(f"wrote {path}")
        return 0


class _PluginScaffold:
    """**A plugin kind** — the second group, ``astro_mine.cli.plugin_scaffolds``.

    Two groups rather than one flag on one group, so that listing the kinds for either verb stays
    a metadata read. This fixture exists to prove a third party can reach both.
    """

    name = "demo-plugin"
    help = "a third-party plugin kind, for testing the scaffold contract"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        del parser

    def run(self, args: argparse.Namespace) -> int:
        target = Path(args.output)
        target.mkdir(parents=True, exist_ok=True)
        (target / "pyproject.toml").write_text('name = "demo-plugin"\n', encoding="utf-8")
        print(f"wrote {target}")
        return 0


demo = _PerVerb()
passthrough = _Passthrough()
doc_scaffold = _DocScaffold()
plugin_scaffold = _PluginScaffold()

#: Resolves fine, satisfies nothing — the packaging bug the umbrella has to report kindly.
MALFORMED = object()
