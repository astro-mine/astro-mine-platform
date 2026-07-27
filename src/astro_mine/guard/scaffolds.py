"""Guard's scaffold for the umbrella CLI — `astro-mine new safety`.

RFC-0011 §7 puts the scaffolding *verb* in the umbrella, because scaffolding spans components and
has no single-component home. It leaves the *template* with whoever owns the format, and a
SafetySpec is Guard's.

**What a starting safety contract should contain is a judgement, not a default.** The anchor spec
(``reference/safety_specs/anchor.safety.yaml``) is 188 lines and exercises every constraint kind
and all three keep-out geometries — the right thing to *read*, and the wrong thing to be handed as
a starting point, because a user editing 188 lines of somebody else's contract is not authoring
theirs. `astro-mine-guard validate anchor` is one command away, and this template points at it.

So the scaffold emits the **smallest contract that is complete**: one hard constraint, the signal
it reads, and the directive allowlist. Three things it will not leave to the reader:

1. **Silence grants nothing.** An absent or empty ``admissible_directives`` certifies *no*
   directive, whatever deployment configuration says — the asymmetry with ``kinematic_limit``,
   where an absent authored limit lets the configured one stand. That is fail-safe and it is not
   guessable, so the block is present and says so. A scaffold that omitted it would produce a valid
   contract that silently blocks every directive the user's stack emits, and they would debug the
   stack.
2. **``on_uncertain`` can never be ``passthrough``** — the fail-safe guarantee lives in the schema,
   not only the runtime. It is written explicitly at its default rather than left implicit, because
   the one place an author will want to "just let it through" is the one place they cannot.
3. **A signal is declared once and referenced by key.** The vocabulary and the constraint are
   separate on purpose; showing both halves for a single signal teaches the shape in four lines.

``keep_out`` is deliberately *not* in the emitted document. It is the constraint most authors want
first and the one with a cross-field rule — its geometry frame must match ``safe_pose.frame``,
enforced in the loader — so a scaffold that included it would hand over a document with an invariant
the user must maintain before they understand it. It is named in a comment with the command that
shows a worked one.

**Nothing here imports the umbrella.** The contract is structural — ``name``, ``help``,
``add_arguments(parser)``, ``run(args) -> int`` — the same four members a verb has (``conventions.md
§1.1``). ``astro-mine-cli`` is not a dependency of this package and must not become one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

__all__ = ["safety_scaffold"]


def _safety_spec(*, spec_id: str, name: str, scenario_ref: str | None) -> str:
    """The smallest SafetySpec that is a complete contract, with the traps written down."""
    scenario = f"\n  scenario_ref: {scenario_ref}" if scenario_ref else ""
    return f"""\
# SafetySpec v0.1 — scaffolded by `astro-mine new safety`.
#
# A hard-constraint safety contract: authored and reviewed once, then reused unchanged across
# design-time training, sim validation, and operations. It is content-addressed, so the hash below
# a `astro-mine-guard validate` run is the contract's immutable identity.
#
#   astro-mine validate <path>          # or `astro-mine-guard validate <path>`
#   astro-mine-guard compile <path>     # lower it to the IR the trusted core enforces
#   astro-mine-guard falsify <path>     # search for a counterexample
#   astro-mine-guard validate anchor    # the shipped 188-line reference contract, worked
safety_version: "0.1"
safety:
  id: {spec_id}
  name: {name}{scenario}

  # Every signal a constraint reads, declared once and referenced below by `key`. `source` says
  # where the value is resolved from — `sadf` a Fleet budget path, `worlds` a terrain/keep-out
  # field, `observation` a Core Environment channel, `derived` computed from others — and `unit`
  # is the explicit SI unit the threshold is stated in. No implicit units anywhere.
  signals:
    - key: power_available_w
      unit: W
      source: sadf
      description: Instantaneous power available to the vehicle.

  # The hard constraints. At least one is required; each names its kind and carries exactly the
  # one typed block that matches.
  #
  # `on_uncertain` is written out at its default deliberately: it decides what happens when the
  # shield cannot tell whether the constraint holds, and it can never be `passthrough`. The
  # fail-safe guarantee is in the schema, not just the runtime — the one place you will want to let
  # something through is the one place you cannot.
  constraints:
    - kind: power_floor
      id: c_power_floor
      description: Power available must never fall below the survival floor.
      on_uncertain: fallback
      power_floor:
        signal: power_available_w
        floor_w: 5.0

  # Other kinds: energy_floor, thermal_ceiling, thermal_floor, torque_ceiling, kinematic_limit,
  # temporal (bounded STL/MTL), and keep_out. `keep_out` is left out of this scaffold on purpose:
  # its geometry frame must match `safe_pose.frame`, a cross-field rule worth meeting deliberately
  # rather than inheriting. `astro-mine-guard validate anchor` shows all of them in place.

  # The MODE/TASK directives this contract certifies, by enumeration.
  #
  # SILENCE GRANTS NOTHING. A directive carries no continuous quantity the shield could correct, so
  # it can only be checked against this allowlist — which makes the grant itself the safety
  # decision, reviewed and content-addressed here rather than set in deployment config. Deleting
  # this block does not mean "whatever the config says"; it means NO directive is certifiable, and
  # every one your stack emits is refused. Deployment config may only ever narrow this set.
  admissible_directives:
    modes:
      # A SADF `loads_by_mode` profile name — the load profile the floors above are stated against.
      - safe_hold
    tasks:
      # Core's closed TaskKind vocabulary. Grant only what this contract actually certifies as
      # safe to enter unconditionally: `standby` and `charge` reduce load and restore energy, while
      # `excavate` or `haul` would raise duty cycle against limits nothing here bounds.
      - standby
      - charge
"""


class _SafetyScaffold:
    """`astro-mine new safety <path>` — a SafetySpec that validates and compiles as written."""

    name = "safety"
    help = "a SafetySpec (Guard owns the format)"

    def add_arguments(self, parser: argparse.ArgumentParser) -> None:
        # `output` and `--force` are declared by the umbrella before this is called, so every kind
        # has the same skeleton and only what is specific to this one is added here.
        parser.description = (
            "Scaffold a hard-constraint safety contract. It is the smallest complete contract "
            "rather than a reduced copy of the anchor spec: one constraint, its signal, and the "
            "directive allowlist. It validates and compiles with no hand-editing."
        )
        parser.add_argument("--id", default="my-safety", help="spec id (default: my-safety)")
        parser.add_argument("--name", help="human-readable name (default: derived from --id)")
        parser.add_argument("--scenario-ref", help="the scenario this contract is stated against")

    def run(self, args: argparse.Namespace) -> int:
        name = args.name or args.id.replace("-", " ").replace("_", " ").capitalize()
        text = _safety_spec(spec_id=args.id, name=name, scenario_ref=args.scenario_ref)
        # The scaffold must always be valid; fail loud if a future edit to the template breaks that,
        # rather than handing a user a safety contract to debug.
        from astro_mine.guard.spec.loader import SafetySpecError, load_safety_spec

        try:
            load_safety_spec(text)
        except SafetySpecError as exc:  # pragma: no cover - defensive guard on a constant template
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


safety_scaffold = _SafetyScaffold()
