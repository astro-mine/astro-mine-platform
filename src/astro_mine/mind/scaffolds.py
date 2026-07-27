"""Mind's scaffolds for the umbrella CLI — `astro-mine new stack` and `plugin new tier`.

RFC-0011 §7 puts the scaffolding *verbs* in the umbrella, because scaffolding spans components and
has no single-component home. It leaves the *templates* with whoever owns the thing being written:
the stack spec is Mind's format, and ``astro_mine.mind.tier_plugins`` is Mind's extension group, so
both are written here.

**Separate from :mod:`astro_mine.mind.umbrella` on purpose.** That module is adapters — a
passthrough to Mind's own ``main`` and a validator that calls Mind's own checker, each a handful of
lines with no content of its own. These carry *templates*, which are content: text that has to stay
valid as the spec and the plugin contract move. Mixing the two would bury a hundred lines of
document in a file whose job is routing.

**What the scaffolds emit must survive Mind's own checker**, which is a higher bar than schema
validity: ``astro-mine-mind validate`` also asks the registry whether every plugin a stack binds is
discoverable (`cli.py`). A stack naming a plugin nobody registers fails — so the scaffold binds
Mind's own reference tiers, which a bare `pip install astro-mine-mind` already provides. A scaffold
whose output only validates once the user has installed something else would be a worked example of
the error it is meant to prevent.

**Nothing here imports the umbrella.** The contract is structural — ``name``, ``help``,
``add_arguments(parser)``, ``run(args) -> int`` — the same four members a verb has, so a component
contributes a scaffold by writing the object it already knows how to write (``conventions.md
§1.1``). ``astro-mine-cli`` is not a dependency of this package and must not become one.
"""

from __future__ import annotations

import argparse
import keyword
import re
import sys
from pathlib import Path

__all__ = ["stack_scaffold", "tier_scaffold"]

_USAGE_ERROR = 2

#: The reference tiers a bare `pip install astro-mine-mind` registers under
#: ``astro_mine.mind.tier_plugins``. The scaffold binds these so its output passes the registry
#: check on the first run; swapping one for a real backend is the spec edit Mind exists to make
#: cheap (`mind.md §3`).
_REFERENCE_TIERS = {
    "mission": "mind.reference.mission",
    "tamp": "mind.reference.tamp",
    "control": "mind.reference.control",
    "shield": "mind.reference.shield",
}


def _stack_spec(*, stack_id: str, name: str, scenario_ref: str | None) -> str:
    """A minimal stack spec that Mind's loader, composer and registry check all accept."""
    scenario = f"\n  scenario_ref: {scenario_ref}" if scenario_ref else ""
    return f"""\
# Stack spec v0.1 — scaffolded by `astro-mine new stack`.
#
# Mind's thesis is that swapping an engine is a spec edit, not a code change. Everything below is
# a binding you can repoint: `astro-mine-mind compose <this file>` reports which plugin bound to
# which tier, from which entry-point group, at which version.
#
# Validate: `astro-mine validate <path>` (or `astro-mine-mind validate <path>`), which checks the
# schema *and* that every plugin named here is discoverable.
stack_spec_version: "0.1"
stack_spec:
  id: {stack_id}
  name: {name}{scenario}
  # Three tiers, mission -> per-agent TAMP -> controller. The bindings are Mind's reference
  # plugins, which a bare install already registers, so this file validates as written. List
  # alternatives with `astro-mine-mind stacks`.
  tiers:
    - role: mission
      plugin: {_REFERENCE_TIERS["mission"]}
      # How long a mission plan stays valid, and what forces a replan before it expires.
      validity_horizon_s: 5.0
      replan_triggers:
        - kind: plan_expired
    - role: tamp
      plugin: {_REFERENCE_TIERS["tamp"]}
      replan_triggers:
        - kind: periodic
          every_ticks: 3
    - role: control
      plugin: {_REFERENCE_TIERS["control"]}
  # The shield is mandatory and is the stack's single output path: every action the executive
  # emits crosses it before it reaches the Environment. The reference one passes actions through;
  # co-install astro-mine-guard and repoint this to `guard.shield` for the real safety core.
  shield:
    plugin: {_REFERENCE_TIERS["shield"]}
"""


class _StackScaffold:
    """`astro-mine new stack <path>` — a stack spec that validates and composes as written."""

    name = "stack"
    help = "an autonomy stack spec (Mind owns the format)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # `output` and `--force` are declared by the umbrella before this is called, so every kind
        # has the same skeleton and only what is specific to this one is added here.
        parser.description = (
            "Scaffold an autonomy stack spec. It binds Mind's reference tiers, so it passes "
            "`astro-mine validate` — including the registry check — with no hand-editing."
        )
        parser.add_argument("--id", default="my-stack", help="stack id (default: my-stack)")
        parser.add_argument("--name", help="human-readable name (default: derived from --id)")
        parser.add_argument("--scenario-ref", help="the scenario this stack is written against")

    def run(self, args: argparse.Namespace) -> int:
        name = args.name or args.id.replace("-", " ").replace("_", " ").capitalize()
        text = _stack_spec(stack_id=args.id, name=name, scenario_ref=args.scenario_ref)
        # The scaffold must always be valid; fail loud if a future edit to the template breaks
        # that, rather than handing the user a document to debug.
        from astro_mine.mind.spec.loader import StackSpecError, load_stack_spec

        try:
            load_stack_spec(text)
        except StackSpecError as exc:  # pragma: no cover - defensive guard on a constant template
            print(f"internal error: scaffold failed validation: {exc}", file=sys.stderr)
            return 1
        return _write({Path(args.output): text}, force=args.force)


# --------------------------------------------------------------------------- the tier plugin


def _tier_pyproject(*, distribution: str, module: str, plugin: str, tier: str) -> str:
    return f"""\
# Generated by `astro-mine plugin new tier`.
#
# The entry point below is the whole contract: install this package and Mind's registry finds the
# plugin. Nothing has to be registered anywhere else, and no Astro-Mine repo has to change.
# The entry-point NAME is the plugin name a stack spec binds, and should match manifest.name.
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "{distribution}"
version = "0.1.0"
description = "A {tier} tier contributed to Astro-Mine-Mind."
requires-python = ">=3.12"
# astro-mine-mind for the TierPlugin container, astro-mine-core for the manifest and messages.
# Note what is NOT here: astro-mine-cli. The umbrella loads this package; it is not a dependency
# of it.
dependencies = [
  "astro-mine-mind",
  "astro-mine-core",
]

[project.entry-points."astro_mine.mind.tier_plugins"]
"{plugin}" = "{module}:{_symbol(plugin)}_plugin"

[tool.hatch.build.targets.wheel]
packages = ["src/{module}"]

# The manifest ships as package data beside the module — it is the plugin's public face, and the
# registry reads it at discovery.
[tool.hatch.build.targets.wheel.force-include]
"src/{module}/manifest.yaml" = "{module}/manifest.yaml"
"""


def _tier_module(*, module: str, plugin: str, tier: str) -> str:
    symbol = _symbol(plugin)
    return f'''\
"""A {tier} tier for an Astro-Mine-Mind autonomy stack.

Generated by `astro-mine plugin new tier`. Bind it from a stack spec::

    tiers:
      - role: {tier}
        plugin: {plugin}

**The contract is a zero-argument provider** returning a ``TierPlugin`` — a Core manifest plus a
factory that builds the tier from a params mapping. Mind re-checks the factory's result against the
Core ``Policy`` contract after construction, so the object below must answer ``decide``.

The manifest's ``kind`` must be ``policy`` — true for a controller, an allocator and a shield
alike — and ``attributes.tier`` names the role Mind's composer will bind this to.
"""

from __future__ import annotations

from importlib import resources
from typing import Any

from astro_mine.core.messages.model import ActionBatch
from astro_mine.core.registry import load_manifest
from astro_mine.mind.registry import TierPlugin


class {symbol.title().replace("_", "")}:
    """A Core Policy: observations in, actions out."""

    def decide(self, observations: Any, context: Any) -> ActionBatch:
        """Called once per tick. Return the actions this tier wants taken."""
        del observations, context
        return ActionBatch(actions=[])


def {symbol}_plugin() -> TierPlugin:
    """The entry-point provider: a Core manifest + a factory.

    Kept cheap on purpose — it loads package data and closes over a constructor. Whatever heavy
    import your tier needs belongs *inside* the factory, so that merely enumerating the registry
    does not pay for it.
    """
    text = resources.files("{module}").joinpath("manifest.yaml").read_text(encoding="utf-8")
    return TierPlugin(
        manifest=load_manifest(text).manifest,
        factory=lambda params: {symbol.title().replace("_", "")}(),
    )
'''


def _tier_manifest(*, plugin: str, tier: str) -> str:
    return f"""\
# The plugin's public face. The entry point is only the in-process discovery mechanism; this is
# what the platform indexes, negotiates against, and gates on (Core PluginManifest).
manifest_version: "0.1"
manifest:
  # Should match the entry-point name in pyproject.toml — that is the name a stack spec binds.
  name: {plugin}
  version: "0.1.0"
  # Required: `policy`, even for an allocator or a shield.
  kind: policy
  description: A {tier} tier contributed as a Mind tier plugin.
  core_interfaces:
    policy: "0.1.0"
    messages: "0.1.0"
  # bit_exact if a seeded run reproduces its output exactly; loosen it if you carry an RNG or a
  # clock. Declaring more determinism than you have is how a benchmark result stops meaning
  # anything.
  determinism_class: bit_exact
  inputs: [Observation, ActionBatch]
  outputs: [ActionBatch]
  signature:
    scheme: unsigned          # sign it for real publication — see the plugin-authoring guide
  attributes:
    # The role Mind's composer binds this to, cross-checked against the stack spec.
    tier: {tier}
"""


class _TierScaffold:
    """`astro-mine plugin new tier <dir>` — a package registering into Mind's tier group."""

    name = "tier"
    help = "an autonomy tier (astro_mine.mind.tier_plugins)"

    #: The roles Mind's composer recognizes. `allocator` and `shield` are tiers too — the manifest
    #: kind is `policy` for all of them, which is the part authors get wrong.
    TIERS = ("mission", "tamp", "control", "shield", "allocator")

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        parser.description = (
            "Scaffold a package that contributes a tier to Mind's autonomy stack. The result "
            "installs, registers, and is discovered by `astro-mine-mind compose`."
        )
        parser.add_argument(
            "--tier",
            default="control",
            choices=self.TIERS,
            help="the role this tier fills (default: control)",
        )
        parser.add_argument(
            "--plugin", help="plugin name as a stack binds it (default: demo.<tier>)"
        )
        parser.add_argument("--distribution", help="distribution name (default: directory name)")
        parser.add_argument("--module", help="import package (default: distribution, `-` as `_`)")

    def run(self, args: argparse.Namespace) -> int:
        target = Path(args.output)
        distribution = args.distribution or target.name
        module = args.module or re.sub(r"[-.]", "_", distribution)
        plugin = args.plugin or f"demo.{args.tier}"

        if not module.isidentifier() or keyword.iskeyword(module):
            print(
                f"astro-mine plugin new tier: {module!r} is not a usable Python package name; "
                f"pass --module explicitly",
                file=sys.stderr,
            )
            return _USAGE_ERROR
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", plugin):
            print(
                f"astro-mine plugin new tier: {plugin!r} is not a usable plugin name; "
                f"pass --plugin explicitly",
                file=sys.stderr,
            )
            return _USAGE_ERROR

        files = {
            target / "pyproject.toml": _tier_pyproject(
                distribution=distribution, module=module, plugin=plugin, tier=args.tier
            ),
            target / "src" / module / "__init__.py": _tier_module(
                module=module, plugin=plugin, tier=args.tier
            ),
            target / "src" / module / "manifest.yaml": _tier_manifest(
                plugin=plugin, tier=args.tier
            ),
        }
        status = _write(files, force=args.force)
        if status == 0:
            print(
                f"\nInstall it and Mind finds it:\n"
                f"  pip install -e {target}\n"
                f"  astro-mine-mind compose <a stack spec binding {plugin}>"
            )
        return status


# --------------------------------------------------------------------------- shared


def _symbol(name: str) -> str:
    """A plugin name is dotted (`demo.control`); a Python symbol is not."""
    return re.sub(r"[-.]", "_", name)


def _write(files: dict[Path, str], *, force: bool) -> int:
    """Write every file, or none of them — a half-written package is worse than none.

    ``--force`` is declared by the umbrella for every kind, so honouring it is not optional: a
    scaffold that ignored it would silently destroy whatever the user had already authored there.
    """
    existing = [path for path in files if path.exists()]
    if existing and not force:
        listing = ", ".join(str(path) for path in existing)
        print(f"{listing}: file exists (use --force to overwrite)", file=sys.stderr)
        return 1
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"wrote {path}")
    return 0


stack_scaffold = _StackScaffold()
tier_scaffold = _TierScaffold()
