"""Worlds's contributions to the umbrella CLI — the `worlds` verb, the validator, the scaffold.

A **component-scoped** surface: RFC-0011 §2 reads it as `astro-mine <component> <verb>`, because
these actions only mean something in Worlds's own vocabulary — unlike `score` or `train`, which a
user names directly (astro-mine/docs#57).

So this is a **passthrough** adapter: it takes the rest of the command line untouched and hands it
to :func:`astro_mine.worlds.cli.main`, the same entry point `worlds` runs. Nothing is
re-declared, which means the umbrella's surface cannot drift from Worlds's real flags — add a
subcommand to Worlds and `astro-mine worlds` has it the same day, with no change here and none
in astro-mine-cli.

Two federated contributions sit beside it (G2.11), each the half Worlds actually owns:

- :data:`validator` — the ``WorldSpec`` checker, under the umbrella's `astro-mine validate`
  (RFC-0011 §6). The verb belongs to astro-mine-cli, because routing a document to the component
  that owns its format is something no single component can do without importing its siblings.
- :data:`world_scaffold` — the ``WorldSpec`` template, under `astro-mine new world` (§7).

**How a WorldSpec is claimed, and why not by a version key.** Guard and Mind claim their documents
by a schema discriminator (``safety_version``, ``stack_spec_version``), and Core claims by required
root plus a version ``const``. ``WorldSpec`` has no such field, and adding one is not free:
``spec_hash`` is a digest over *every* field of ``model_dump`` and ``world_hash`` is a digest over
that, so a new field — even a defaulted one — would move the hash of every world already built,
including the published anchor bundle and the Bench zoo scenarios pinned to it.

So a WorldSpec is claimed by its **required root**: ``world_id`` + ``crs`` + ``region`` +
``source_dem``, the four fields the model gives no default. That is identification rather than
resemblance — the four together are unique to this format — and it costs nothing. If a sibling ever
claimed the same document the umbrella would raise rather than guess, so the failure mode is loud.

**Nothing here imports the umbrella.** Every contract is structural — a verb and a scaffold are
``name``, ``help``, ``add_arguments(parser)``, ``run(args) -> int``; a validator is ``name``,
``claims``, ``validate`` — so a component is reachable from `astro-mine` without depending on it
(``conventions.md §1.1``). ``astro-mine-cli`` is not a dependency of this package and must not
become one.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from astro_mine.worlds.cli import _cmd_validate, main

__all__ = ["REQUIRED_ROOT", "validator", "world_scaffold", "worlds"]

#: The root properties ``WorldSpec`` declares with no default — the claim key. Kept as data so the
#: rule and the model cannot drift apart silently: a test derives this same set from the model and
#: compares, so adding a required field to ``WorldSpec`` without revisiting the claim fails here.
REQUIRED_ROOT: frozenset[str] = frozenset({"world_id", "crs", "region", "source_dem"})


class _Worlds:
    name = "worlds"
    help = "build and publish world bundles"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "tail",
            nargs=argparse.REMAINDER,
            help="the worlds command line (`worlds --help` lists it)",
        )

    def run(self, args: argparse.Namespace) -> int:
        # SystemExit is caught and converted rather than left to propagate: argparse raises it for
        # a usage error inside Worlds's own parser, and the umbrella's contract is that `run`
        # *returns* the exit status. Converting keeps one rule for every verb; the status itself is
        # unchanged, so `astro-mine worlds --nonsense` still exits exactly as `worlds` would.
        try:
            result = main(args.tail)
        except SystemExit as exit_:
            code = exit_.code
            if code is None:
                return 0
            return code if isinstance(code, int) else 1
        return int(result)


class _WorldsValidator:
    """Worlds's half of the federated `astro-mine validate` (RFC-0011 §6)."""

    name = "worlds"

    def claims(self, path: str) -> str | None:
        """Return ``"world_spec"`` if ``path`` is a WorldSpec, else ``None``.

        Cheap and total: the umbrella asks every installed validator about every file, so a
        document that is Core's or Guard's must come back ``None`` rather than raise — at claim
        time nobody owns the file yet, and raising would turn another component's malformed
        document into a Worlds traceback. A file that *is* Worlds's but malformed is claimed here
        and then reported properly by the real checker.
        """
        import yaml

        try:
            with open(path, encoding="utf-8") as handle:
                document = yaml.safe_load(handle)
        except (OSError, yaml.YAMLError):
            return None
        if isinstance(document, dict) and document.keys() >= REQUIRED_ROOT:
            return "world_spec"
        return None

    def validate(self, paths: Sequence[str], *, as_json: bool) -> int:
        """Run the same checker `astro-mine-worlds validate` runs — not a second implementation."""
        return int(_cmd_validate(argparse.Namespace(path=list(paths), json=as_json)))


class _WorldScaffold:
    """`astro-mine new world <path>` — a WorldSpec that validates as written (RFC-0011 §7)."""

    name = "world"
    help = "a WorldSpec (Worlds owns the format)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # `output` and `--force` are declared by the umbrella before this is called, so every kind
        # has the same skeleton and only what is specific to this one is added here.
        parser.description = (
            "Scaffold a WorldSpec. It is the shipped example with your identity substituted, so "
            "it passes `astro-mine validate` with no hand-editing — then point `source_dem` at a "
            "real product and pin its digest."
        )
        parser.add_argument("--id", default="my-world", help="world id (default: my-world)")
        parser.add_argument(
            "--world-version", default="0.1.0", help="world version (default: 0.1.0)"
        )

    def run(self, args: argparse.Namespace) -> int:
        from astro_mine.worlds.spec import WorldSpec, example_world_spec_text

        text = example_world_spec_text(world_id=args.id, version=args.world_version)
        # The scaffold must always be valid; fail loud if a future edit to the shipped example
        # breaks that, rather than handing the user a document to debug.
        try:
            WorldSpec.from_yaml_text(text)
        except Exception as exc:  # pragma: no cover - defensive guard on shipped package data
            print(f"internal error: scaffold failed validation: {exc}", file=sys.stderr)
            return 1

        out = Path(args.output)
        if out.exists() and not args.force:
            print(f"{out}: file exists (use --force to overwrite)", file=sys.stderr)
            return 1
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {out}")
        return 0


worlds = _Worlds()
validator = _WorldsValidator()
world_scaffold = _WorldScaffold()
