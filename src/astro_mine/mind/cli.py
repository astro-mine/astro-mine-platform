"""``astro-mine-mind`` — validate, compose, and list Mind stack specs (G2.6).

Mind's thesis is *"swapping an engine is a spec edit, not a code change."* This CLI is the shell
over the tooling that makes that true — the loader, the composer, and the entry-point registry Mind
already ships — so a user can check and inspect a stack spec without writing Python.

Three verbs:

- ``validate <stack.yaml>`` — schema + model validation, **plus** a registry check that every
  plugin a stack binds is discoverable. A stack naming an unregistered plugin fails here, with the
  entry-point group and the missing name — the error users hit most, and the one a shape-only
  validator misses.
- ``compose <stack.yaml>`` — resolve the spec to a stack and report which plugin bound to which
  tier, from which entry-point group, **at which version**. The debugging tool the plugin system
  lacked: an unregistered or wrong-kind plugin surfaces here, at its cause, not at run time.
- ``stacks`` — enumerate the reference stacks and manifests Mind ships as package data.

**No ``run``.** Stepping a composed stack needs a Core ``Environment``, which Mind does not provide
— that is Sim's job, and Mind must not import Sim (the narrow waist; `conventions.md §1.1`).
``astro-mine-sim run`` is the episode entry point; a Sim-backed ``mind run`` is tracked separately.
No verb here pretends to run an episode it cannot.

The dispatch is importable so the umbrella ``astro-mine`` ([RFC-0011](RFC-0011)) routes in as a thin
call rather than a rewrite.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from importlib import resources
from pathlib import Path

from astro_mine.mind.compose.composer import ComposeError, compose
from astro_mine.mind.registry.registry import ENTRY_POINT_GROUP, TierRegistry
from astro_mine.mind.spec.loader import StackSpecError, load_stack_spec
from astro_mine.mind.spec.model import StackSpecDocument

__all__ = ["iter_manifest_resources", "iter_stack_resources", "main"]

_REFERENCE = "astro_mine.mind.reference"


# --------------------------------------------------------------------------- package data


def iter_stack_resources() -> Iterator[str]:
    """The shipped reference stack-spec filenames, sorted (from package data, wheel-safe)."""
    yield from sorted(
        entry.name
        for entry in resources.files(_REFERENCE).joinpath("stacks").iterdir()
        if entry.name.endswith(".yaml")
    )


def iter_manifest_resources() -> Iterator[str]:
    """The shipped reference plugin-manifest filenames, sorted (from package data, wheel-safe)."""
    yield from sorted(
        entry.name
        for entry in resources.files(_REFERENCE).joinpath("manifests").iterdir()
        if entry.name.endswith(".yaml")
    )


def _read_stack(path: str) -> str:
    """A stack-spec document's text: a filesystem path, ``-`` for stdin, or a shipped resource name.

    A bare name that matches a shipped reference stack (``lunar_prospecting.yaml``) resolves from
    package data — so the CLI works against the shipped curriculum without a checkout.
    """
    if path == "-":
        return sys.stdin.read()
    if not Path(path).exists() and path in set(iter_stack_resources()):
        return resources.files(_REFERENCE).joinpath("stacks", path).read_text(encoding="utf-8")
    return Path(path).read_text(encoding="utf-8")


def _plugin_bindings(document: StackSpecDocument) -> list[tuple[str, str]]:
    """Every ``(role, plugin)`` a stack binds — tiers, their fallbacks, and the shield."""
    spec = document.stack_spec
    bindings: list[tuple[str, str]] = []
    for tier in spec.tiers:
        bindings.append((tier.role.value, tier.plugin))
        if tier.fallback is not None:
            bindings.append((f"{tier.role.value} fallback", tier.fallback.plugin))
    bindings.append(("shield", spec.shield.plugin))
    return bindings


# --------------------------------------------------------------------------- validate


def _cmd_validate(args: argparse.Namespace) -> int:
    failed = False
    registry = TierRegistry.from_entry_points()
    for path in args.stack:
        label = path
        try:
            document = load_stack_spec(_read_stack(path))
        except OSError as exc:
            print(f"FAIL {label}: cannot read file: {exc.strerror or exc}", file=sys.stderr)
            failed = True
            continue
        except StackSpecError as exc:
            print(f"FAIL {label}: {exc}", file=sys.stderr)
            failed = True
            continue

        # Registry validity — the composer's gate, surfaced early: a bound plugin that no installed
        # package registers is the most common real failure, and shape-only validation misses it.
        missing = [
            (role, name) for role, name in _plugin_bindings(document) if name not in registry
        ]
        if missing:
            print(
                f"FAIL {label}: stack {document.stack_spec.id!r} binds unregistered plugin(s):",
                file=sys.stderr,
            )
            for role, name in missing:
                print(
                    f"  {role}: {name!r} is not registered in entry-point group "
                    f"{ENTRY_POINT_GROUP!r} — is its package installed?",
                    file=sys.stderr,
                )
            failed = True
            continue
        spec = document.stack_spec
        print(f"OK  {label}: valid stack {spec.id!r} ({len(spec.tiers)} tiers)")
    return 1 if failed else 0


# --------------------------------------------------------------------------- compose


def _cmd_compose(args: argparse.Namespace) -> int:
    from astro_mine.mind.spec.enums import ExecutionKind

    registry = TierRegistry.from_entry_points()
    try:
        document = load_stack_spec(_read_stack(args.stack))
    except OSError as exc:
        print(f"cannot read {args.stack}: {exc.strerror or exc}", file=sys.stderr)
        return 1
    except StackSpecError as exc:
        print(f"invalid stack: {exc}", file=sys.stderr)
        return 1

    behavior_tree = None
    if document.stack_spec.execution.kind is ExecutionKind.BEHAVIOR_TREE:
        try:
            from astro_mine.mind.reference import load_reference_bt

            behavior_tree = load_reference_bt()
        except Exception as exc:  # any resolution failure is reported to the user, not raised
            print(
                f"compose: this stack uses behavior-tree execution and its tree could not be "
                f"resolved from package data ({exc}). Pass a composition-execution stack, or run "
                f"compose in an environment where the reference tree resolves.",
                file=sys.stderr,
            )
            return 1

    try:
        graph = compose(document, registry, behavior_tree=behavior_tree)
    except ComposeError as exc:
        print(f"compose failed: {exc}", file=sys.stderr)
        return 1

    print(f"stack: {graph.stack_id}")
    print(f"execution: {graph.execution.kind.value}")
    print(f"entry-point group: {ENTRY_POINT_GROUP}")
    print("tiers (role -> plugin @ version):")
    for tier in graph.tiers:
        line = f"  {tier.role.value:<10} {tier.plugin_name} @ {tier.manifest.version}"
        if tier.fallback_name is not None:
            fb_version = graph.provenance.plugin_versions.get(tier.fallback_name, "?")
            line += f"  (fallback: {tier.fallback_name} @ {fb_version})"
        print(line)
    print(f"  {'shield':<10} {graph.shield.plugin_name} @ {graph.shield.manifest.version}")
    cores = ", ".join(
        f"{k}={v}" for k, v in sorted(graph.provenance.core_interface_versions.items())
    )
    print(f"core interface versions: {cores}")
    return 0


# --------------------------------------------------------------------------- stacks


def _cmd_stacks(args: argparse.Namespace) -> int:
    stacks = list(iter_stack_resources())
    manifests = list(iter_manifest_resources())
    print(
        f"reference stacks ({len(stacks)}) — "
        f"load with astro_mine.mind.reference.load_stack_resource(<name>):"
    )
    for name in stacks:
        print(f"  {name}")
    print(f"\nreference manifests ({len(manifests)}) — tier/shield plugin descriptors:")
    for name in manifests:
        print(f"  {name}")
    return 0


# --------------------------------------------------------------------------- parser


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astro-mine-mind",
        description="Validate, compose, and list Mind stack specs. A bare stack name resolves "
        "against the shipped reference stacks (see 'astro-mine-mind stacks').",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser(
        "validate", help="validate one or more stack specs (schema + registry)"
    )
    validate.add_argument(
        "stack", nargs="+", help="stack path(s), '-' for stdin, or a reference stack name"
    )
    validate.set_defaults(func=_cmd_validate)

    compose_p = sub.add_parser("compose", help="resolve a stack to tier -> plugin @ version")
    compose_p.add_argument("stack", help="stack path, '-', or a reference stack name")
    compose_p.set_defaults(func=_cmd_compose)

    stacks = sub.add_parser("stacks", help="list the reference stacks and manifests Mind ships")
    stacks.set_defaults(func=_cmd_stacks)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
