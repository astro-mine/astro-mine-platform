"""Sim's verbs on the umbrella CLI — `astro-mine run` and `astro-mine record`.

Two of the platform's **headline** actions: a user who wants to run a scenario reaches for
*"run"*, not for the package that owns the physics. RFC-0011 §2 puts those at the top level of
`astro-mine`, so each gets a per-verb adapter rather than an `astro-mine sim …` passthrough
(astro-mine/docs#57).

**Nothing here imports the umbrella.** The contract is structural — ``name``, ``help``,
``add_arguments(parser)``, ``run(args) -> int`` — precisely so a component is reachable from
`astro-mine` without depending on it (``conventions.md §1.1``). ``astro-mine-cli`` is not a
dependency of this package and must not become one.

**Nothing here re-declares a flag.** Both adapters attach the same ``add_*_arguments`` function
:mod:`astro_mine.sim.__main__` uses for its own parser, and call the same handler — so
`astro-mine run` and `astro-mine-sim run` cannot drift.

The umbrella imports this module only when one of these verbs runs, which matters more here than
in most components: importing Sim is not cheap, and a user who never types `run` never pays for it.
"""

from __future__ import annotations

import argparse

from astro_mine.sim.__main__ import (
    _record,
    _run,
    add_record_arguments,
    add_run_arguments,
)

__all__ = ["record", "run"]


class _Run:
    name = "run"
    help = "run a scenario in the simulator"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        add_run_arguments(parser)

    def run(self, args: argparse.Namespace) -> int:
        return _run(args)


class _Record:
    name = "record"
    help = "record a self-contained Sim scenario file"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        add_record_arguments(parser)

    def run(self, args: argparse.Namespace) -> int:
        return _record(args)


run = _Run()
record = _Record()
