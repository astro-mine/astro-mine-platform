"""The built-in `astro-mine validate` verb — a router, not a validator (RFC-0011 §6).

Every other verb on `astro-mine` belongs to a component. This one belongs to the umbrella, because
it is the only verb whose job is precisely *"work out who owns this and ask them"* — and no single
component can do that without knowing about the others, which the narrow waist forbids
(``conventions.md §1.1``).

It validates nothing itself. It groups the given files by owning validator, hands each group to its
owner, and returns the worst status. Core keeps its ``$id``-keyed dispatch over the nine
Core-authored formats; Guard keeps the SafetySpec compiler's checks; Mind keeps the stack-spec
registry check. Nothing is reimplemented here, and this file contains no schema knowledge at all.

**Why it is a built-in rather than Core's registered verb.** It used to be the latter: Core
advertised ``validate`` into ``astro_mine.cli`` and the umbrella just routed to it. That could
never federate — extending it would have meant Core consulting Guard and Mind, i.e. Core importing
its own consumers. Moving the verb up here is what lets each owner keep its checker while a user
still types one command.
"""

from __future__ import annotations

import argparse
import sys

from astro_mine.cli._validators import (
    ClaimCollisionError,
    InvalidValidatorError,
    Validator,
    claim,
    discover_validators,
)

__all__ = ["validate"]

_USAGE_ERROR = 2

_NO_VALIDATORS = (
    "no validators are installed, so `astro-mine validate` has nothing to route to. Install the "
    "package that owns the format you are checking — `astro-mine-core` for SADF, ObjectiveSpec, "
    "MissionSpec, Plan, plugin manifests, PolicyPackage and RunProvenance; `astro-mine-guard` for "
    "a SafetySpec; `astro-mine-mind` for a stack spec."
)


class _Validate:
    name = "validate"
    help = "validate an authored document (routed to the format's owner)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.description = (
            "Validate authored documents. Each file is routed to the component that owns its "
            "format; this command owns no schema of its own (RFC-0011 §6)."
        )
        parser.add_argument("file", nargs="+", help="path to a JSON/YAML document")
        parser.add_argument("--json", action="store_true", help="machine-readable output")

    def run(self, args: argparse.Namespace) -> int:
        try:
            validators = discover_validators()
        except InvalidValidatorError as exc:
            print(f"astro-mine validate: {exc}", file=sys.stderr)
            return _USAGE_ERROR

        if not validators:
            print(f"astro-mine validate: {_NO_VALIDATORS}", file=sys.stderr)
            return _USAGE_ERROR

        # Group first, dispatch second: a validator that can check several files at once should be
        # called once, so its output reads as one report rather than N unrelated ones.
        grouped: dict[str, tuple[Validator, list[str]]] = {}
        status = 0
        for path in args.file:
            try:
                owner, _kind = claim(validators, path)
            except ClaimCollisionError as exc:
                print(f"astro-mine validate: {exc}", file=sys.stderr)
                return _USAGE_ERROR
            except LookupError:
                print(
                    f"astro-mine validate: no installed validator recognizes {path}. "
                    f"Installed: {', '.join(v.name for v in validators)}. A document is never "
                    f"checked against a guessed schema — name its format to the owning CLI "
                    f"instead (e.g. `astro-mine-core validate --kind …`).",
                    file=sys.stderr,
                )
                status = max(status, 1)
                continue
            grouped.setdefault(owner.name, (owner, []))[1].append(path)

        for owner, paths in grouped.values():
            status = max(status, int(owner.validate(paths, as_json=bool(args.json))))
        return status


validate = _Validate()
