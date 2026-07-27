"""Subcommands the fast-lane tests point entry points at.

Kept apart from the installed fixture under ``fixtures/provider``: this module is imported
directly by in-process tests, while that one exists to be *installed* into a throwaway venv. The
two must not be merged — an in-process import of the fixture would defeat the very test that
asserts listing verbs imports nothing.
"""

from __future__ import annotations

import argparse

__all__ = ["ECHO", "EXPLODING", "MALFORMED", "NOT_CALLABLE", "RETURNS_NONE", "make_entry_point"]


class _Echo:
    name = "echo"
    help = "echo a message back"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("message", nargs="?", default="hi")
        parser.add_argument("--exit-code", type=int, default=0)

    def run(self, args: argparse.Namespace) -> int:
        print(args.message)
        return int(args.exit_code)


class _ReturnsNone:
    """A verb that follows the ``sys.exit(None)`` convention instead of returning 0."""

    name = "quiet"
    help = "returns None rather than an int"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        del parser

    def run(self, args: argparse.Namespace) -> None:
        del args
        return None


class _Exploding:
    """A verb whose own execution fails — the umbrella must not swallow this."""

    name = "boom"
    help = "raises from inside run()"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        del parser

    def run(self, args: argparse.Namespace) -> int:
        del args
        raise RuntimeError("the component itself failed")


class _NotCallable:
    """Has every required name, but ``run`` is data rather than behaviour."""

    name = "inert"
    help = "run is not callable"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        del parser

    run = "not a function"


ECHO = _Echo()
RETURNS_NONE = _ReturnsNone()
EXPLODING = _Exploding()
NOT_CALLABLE = _NotCallable()
MALFORMED = object()


def make_entry_point(name: str, attribute: str, group: str | None = None):  # type: ignore[no-untyped-def]
    """An EntryPoint pointing at this module — the fast lane's stand-in for an installed package.

    ``group`` defaults to the verb group; the scaffold groups pass their own, since the contract
    they bind to is the same one and only the group name differs.

    Imported lazily so this module stays importable without importlib.metadata in scope.
    """
    from importlib.metadata import EntryPoint

    from astro_mine.cli import VERB_ENTRY_POINT_GROUP

    return EntryPoint(name=name, value=f"_verbs:{attribute}", group=group or VERB_ENTRY_POINT_GROUP)
