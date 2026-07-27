"""Bench's verbs on the umbrella CLI — `astro-mine score`, `fetch`, `submit`, `list`.

Bench owns four of the platform's **headline** actions: the things a user reaches for by naming
the *action* rather than the package that implements it. RFC-0011 §2 puts exactly those at the top
level of `astro-mine`, so each gets a per-verb adapter here rather than hiding behind an
`astro-mine bench …` passthrough (astro-mine/docs#57).

**Nothing here imports the umbrella.** The contract is structural — an object with ``name``,
``help``, ``add_arguments(parser)`` and ``run(args) -> int`` — precisely so a component can be
reachable from `astro-mine` without depending on it (``conventions.md §1.1``). ``astro-mine-cli``
is not a dependency of this package, and must not become one.

**Nothing here re-declares a flag either.** Each adapter attaches the same
``add_*_arguments`` function :mod:`astro_mine.bench.cli` uses for its own parser, and calls the
same handler. That is what keeps `astro-mine score` and `astro-mine-bench score` from drifting: a
flag added in one place appears on both surfaces, and no test has to police a duplicate list.

The umbrella imports this module only when one of these verbs actually runs, so declaring the
entry points costs a user who never types them nothing.
"""

from __future__ import annotations

import argparse
import sys

from astro_mine.bench.cli import (
    FETCH_DESCRIPTION,
    SUBMIT_DESCRIPTION,
    _fetch,
    _list,
    _score,
    _submit,
    add_fetch_arguments,
    add_score_arguments,
    add_submit_arguments,
)

__all__ = ["fetch", "list_scenarios", "score", "submit"]


def _with_streams(args: argparse.Namespace) -> argparse.Namespace:
    """Attach the output streams the handlers write through.

    Bench's handlers print to ``args.stdout``/``args.stderr`` rather than to the real streams, so
    tests can inject buffers — and :func:`astro_mine.bench.cli.main` sets them after parsing. The
    umbrella owns the parser here, so it does not, and a handler would raise ``AttributeError`` on
    its first print. Defaulting rather than overwriting keeps the same buffer-injection trick
    available to a caller of these adapters.
    """
    if getattr(args, "stdout", None) is None:
        args.stdout = sys.stdout
    if getattr(args, "stderr", None) is None:
        args.stderr = sys.stderr
    return args


class _Score:
    name = "score"
    help = "run a policy on a scenario and score it"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        add_score_arguments(parser)

    def run(self, args: argparse.Namespace) -> int:
        return _score(_with_streams(args))


class _Fetch:
    name = "fetch"
    help = "download a scenario's pinned content into a local store"
    description = FETCH_DESCRIPTION

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # The 461 MB / read:packages warning belongs on this surface too — a user who reaches
        # `fetch` through the umbrella needs it just as much as one who typed the Bench binary.
        parser.description = FETCH_DESCRIPTION
        add_fetch_arguments(parser)

    def run(self, args: argparse.Namespace) -> int:
        return _fetch(_with_streams(args))


class _Submit:
    name = "submit"
    help = "submit a policy to a leaderboard"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.description = SUBMIT_DESCRIPTION
        add_submit_arguments(parser)

    def run(self, args: argparse.Namespace) -> int:
        return _submit(_with_streams(args))


class _List:
    name = "list"
    help = "list the scenarios in the zoo"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        del parser  # `list` takes none, on either surface

    def run(self, args: argparse.Namespace) -> int:
        return _list(_with_streams(args))


score = _Score()
fetch = _Fetch()
submit = _Submit()
#: Bound to the ``list`` verb; named for the module-level shadowing `list` would cause.
list_scenarios = _List()
