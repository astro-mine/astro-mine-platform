"""``astro-mine-guard`` — a shell over Guard's SafetySpec tooling (G2.6).

Guard ships a spec validator, a deterministic compiler, an offline dev signer, and a seeded
falsification harness, and until now exposed **none of them from a shell**. This CLI does, with
four verbs:

- ``validate`` — a ``SafetySpec`` against its schema, with actionable errors;
- ``compile``  — a validated spec to the content-addressed :class:`CompiledSafetyModel`, printing
  the compiled artifact's content hash;
- ``falsify``  — the seeded adversarial search against **any** spec, reporting the seed that
  produced any violation (CX-REPRO: a falsification you cannot replay is an anecdote);
- ``sign``     — an offline cosign signature over a spec's content hash (the dev path).

**Fail-safe, never fail-open.** Guard's semantics are fail-closed by design; the CLI inherits that
— an unparseable or ambiguous spec is a *failure*, never a pass, and no verb reports success on a
document it did not fully check.

All four take a spec path, ``-`` for stdin, or the literal ``anchor`` for the shipped reference
spec.
``falsify`` did not, and so could only ever falsify the anchor: the authoring loop
``validate → compile → falsify → sign`` stopped one step short of the step that justifies trusting
what you wrote (issue #35). It takes one now, deriving the search's start and attack from the spec's
own safe set — see :mod:`astro_mine.guard.falsify.derive` for why no scenario binding is needed.

**The compiled Rust core (`_core`) is kept out of the import path** so the spec tooling loads
without it (the ``__init__`` contract). ``validate``/``compile``/``sign`` need only the spec
tooling. ``falsify`` needs the shield runtime for its *shielded* verification; it imports ``_core``
lazily and, when it is absent, runs the unshielded control search and **degrades with a message
naming the fix** — never an ``ImportError`` traceback.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from astro_mine.guard.reference import ANCHOR_SAFETY_SPEC_RESOURCE, anchor_safety_spec_text
from astro_mine.guard.spec import (
    CompileError,
    SafetySpecError,
    compile_spec,
    compiled_content_hash,
    load_safety_spec,
    sign_digest,
    spec_content_hash,
)

if TYPE_CHECKING:
    from astro_mine.guard.spec.model import SafetyDocument

__all__ = ["main"]

# The offline dev signing keypair — the workspace convention (LUNAR-TR-004; the local/dev path, no
# Fulcio/Rekor/network). Real callers pass --key; if neither this default nor --key resolves to a
# file, `sign` fails closed naming --key rather than signing with a phantom key.
_DEFAULT_KEY = Path(
    os.environ.get(
        "ASTRO_MINE_GUARD_DEV_KEY",
        "/mnt/d/MyProjects/AstroMine/files/hub-registry/keys/anchor-dev.key.pem",
    )
)
_DEFAULT_PUB = Path(
    os.environ.get(
        "ASTRO_MINE_GUARD_DEV_PUB",
        "/mnt/d/MyProjects/AstroMine/files/hub-registry/keys/anchor-dev.pub.pem",
    )
)

# The anchor falsification cadence: a coarse sample period keeps the two ~14-day survival monitors'
# ring buffers tiny, so a per-agent SafetyCore builds instantly and the seeded rollouts run fast.
_ANCHOR_FALSIFY_PERIOD_S = 120_960.0


def _read_spec(path: str) -> str:
    """The document text at ``path``, or the packaged anchor spec when ``path`` is ``anchor``."""
    if path == "anchor":
        return anchor_safety_spec_text()
    if path == "-":
        return sys.stdin.read()
    return Path(path).read_text(encoding="utf-8")


# --------------------------------------------------------------------------- validate


def _cmd_validate(args: argparse.Namespace) -> int:
    failed = False
    for path in args.spec:
        label = ANCHOR_SAFETY_SPEC_RESOURCE if path == "anchor" else path
        try:
            source = _read_spec(path)
        except OSError as exc:
            print(f"FAIL {label}: cannot read file: {exc.strerror or exc}", file=sys.stderr)
            failed = True
            continue
        try:
            document = load_safety_spec(source)  # parse + structural + fail-safe semantic
        except SafetySpecError as exc:
            print(f"FAIL {label}: {exc}", file=sys.stderr)
            failed = True
            continue
        print(f"OK  {label}: valid SafetySpec {document.safety.id} ({spec_content_hash(document)})")
    return 1 if failed else 0


# --------------------------------------------------------------------------- compile


def _cmd_compile(args: argparse.Namespace) -> int:
    from astro_mine.core.hashing import canonical_json

    try:
        source = _read_spec(args.spec)
        document = load_safety_spec(source)
        compiled = compile_spec(document, sample_period_s=args.sample_period)
    except OSError as exc:
        print(f"cannot read {args.spec}: {exc.strerror or exc}", file=sys.stderr)
        return 1
    except SafetySpecError as exc:
        print(f"invalid spec: {exc}", file=sys.stderr)
        return 1
    except CompileError as exc:
        print(f"compile failed: {exc}", file=sys.stderr)
        return 1

    content_hash = compiled_content_hash(compiled)
    rendered = canonical_json(compiled.model_dump(mode="json"))
    if args.out:
        Path(args.out).write_bytes(rendered + b"\n")
        print(f"wrote {args.out}")
    else:
        sys.stdout.buffer.write(rendered + b"\n")
    print(f"spec_id:       {document.safety.id}", file=sys.stderr)
    print(f"spec_hash:     {spec_content_hash(document)}", file=sys.stderr)
    print(f"compiled_hash: {content_hash}", file=sys.stderr)
    return 0


# --------------------------------------------------------------------------- falsify


def _cmd_falsify(args: argparse.Namespace) -> int:
    from astro_mine.guard.falsify import (
        DEFAULT_DT,
        DEFAULT_U_MAX,
        AdversaryPolicy,
        SeededAdversary,
        WorstCaseAdversary,
        control_rollout,
        control_violations,
        shielded_rollout,
        shielded_violations,
    )
    from astro_mine.guard.falsify.derive import FalsifyDeriveError, initial_state

    # Spec tooling only — no _core needed to get this far, so a spec that cannot be falsified
    # says so whether or not the Rust core is built.
    try:
        document = (
            load_anchor_document()
            if args.spec == "anchor"
            else load_safety_spec(_read_spec(args.spec))
        )
        compiled = compile_spec(document, sample_period_s=args.sample_period)
        # The start is read out of the spec's *own* safe set: a position clear of its keep-out
        # geometry, and each signal inside the envelope its own bounds carve out. No scenario is
        # involved — the plant is a synthetic double integrator (see falsify/derive.py).
        initial = initial_state(compiled)
    except OSError as exc:
        print(f"cannot read {args.spec}: {exc.strerror or exc}", file=sys.stderr)
        return 1
    except SafetySpecError as exc:
        print(f"invalid spec: {exc}", file=sys.stderr)
        return 1
    except CompileError as exc:
        print(f"compile failed: {exc}", file=sys.stderr)
        return 1
    except FalsifyDeriveError as exc:
        print(f"cannot falsify {args.spec}: {exc}", file=sys.stderr)
        return 1

    print(f"spec:    {document.safety.id} ({spec_content_hash(document)})", file=sys.stderr)
    print(
        f"start:   position {tuple(round(x, 3) for x in initial.position)}, "
        f"{len(initial.signals)} signal(s) inside their own bounds",
        file=sys.stderr,
    )

    # 1. The unshielded control: the raw worst-case attack MUST breach, or the search is vacuous.
    worst = WorstCaseAdversary(compiled)
    control_steps = control_rollout(worst, spatial_dim=3, initial=initial, horizon=args.horizon)
    control = control_violations(control_steps, compiled, u_max=DEFAULT_U_MAX, dt=DEFAULT_DT)
    if not control:
        print(
            f"falsify: the unshielded control search found no violations against "
            f"{document.safety.id!r} — the harness is vacuous and proves nothing, so a "
            "'shield held' "
            "result here would mean nothing either. Either the spec declares no reachable hard "
            "constraint, or --horizon is too short for its envelopes to be crossed. Aborting.",
            file=sys.stderr,
        )
        return 1
    print(f"control (unshielded): {len(control)} violation(s) — the search is real")

    # 2. Shielded verification needs the Rust safety core. Degrade honestly if it is absent.
    try:
        from astro_mine.guard.audit.sink import CollectingSink
        from astro_mine.guard.wrap import CoreConfig, PolicyShield
    except ImportError:
        print(
            "\nshield verification skipped: the Rust safety core (astro_mine.guard._core) is not "
            "built.\n  Build it with:  maturin develop --release   (or `uv sync`)\n"
            "The unshielded search above confirms the harness bites; run again with the core built "
            "to verify the shield holds.",
            file=sys.stderr,
        )
        return 0

    seeds = list(args.seed) if args.seed else list(range(args.trials))
    breached: list[tuple[int, int]] = []
    for seed in seeds:
        # `compiled=` is what makes the random walk attack THIS spec's signals rather than the
        # anchor's six keys — the whole of what stopped falsify from taking a spec (issue #35).
        adversary = SeededAdversary(seed, compiled=compiled)
        sink = CollectingSink()
        shield = PolicyShield(
            AdversaryPolicy(adversary, spatial_dim=3),
            compiled,
            sink=sink,
            core_config=CoreConfig(),
        )
        steps = shielded_rollout(
            shield, adversary, initial=initial, horizon=args.horizon, sink=sink
        )
        violations = shielded_violations(steps, compiled, u_max=DEFAULT_U_MAX, dt=DEFAULT_DT)
        status = "held" if not violations else f"BREACHED ({len(violations)})"
        print(f"  seed {seed:>4}: shield {status}")
        if violations:
            breached.append((seed, len(violations)))

    if breached:
        print(
            f"\nFALSIFIED: the shield admitted a violation for seed(s) "
            f"{', '.join(str(s) for s, _ in breached)}. Reproduce with --seed <n>.",
            file=sys.stderr,
        )
        return 1
    print(f"\nshield held across {len(seeds)} seed(s) — zero hard-constraint violations")
    return 0


# --------------------------------------------------------------------------- sign


def _cmd_sign(args: argparse.Namespace) -> int:
    from astro_mine.guard.spec import load_signed_safety_spec

    label = ANCHOR_SAFETY_SPEC_RESOURCE if args.spec == "anchor" else args.spec
    if not args.key.is_file():
        print(
            f"sign: no signing key at {args.key} — pass --key <private-key.pem> "
            "(or set ASTRO_MINE_GUARD_DEV_KEY)",
            file=sys.stderr,
        )
        return 1
    try:
        source = _read_spec(args.spec)
    except OSError as exc:
        print(f"cannot read {label}: {exc.strerror or exc}", file=sys.stderr)
        return 1

    # Structural + fail-safe validation BEFORE we sign — never sign an unvalidated spec.
    try:
        document = load_safety_spec(source)
    except SafetySpecError as exc:
        print(f"refusing to sign an invalid spec: {exc}", file=sys.stderr)
        return 1

    digest = document.content_hash()
    signature = sign_digest(digest, args.key.read_bytes())
    print(f"spec_id:      {document.safety.id}")
    print(f"content_hash: {digest}")
    print(f"signature:    {signature.model_dump_json()}")

    if args.verify:
        if not args.pub.is_file():
            print(f"--verify: no public key at {args.pub} — pass --pub", file=sys.stderr)
            return 1
        _doc, provenance = load_signed_safety_spec(
            source, signature, trusted_public_key_pem=args.pub.read_bytes()
        )
        print(f"verified:     {provenance.verified} (signer {provenance.signer_id})")
    return 0


# --------------------------------------------------------------------------- helpers/parser


def load_anchor_document() -> SafetyDocument:
    """The reviewed anchor SafetyDocument, from package data."""
    from astro_mine.guard.reference import load_anchor_safety_spec

    return load_anchor_safety_spec()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="astro-mine-guard",
        description="Validate, compile, falsify and sign Guard SafetySpecs. "
        "Pass 'anchor' as the spec to use the shipped reference anchor spec.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="validate one or more SafetySpecs")
    validate.add_argument(
        "spec", nargs="*", default=["anchor"], help="spec path(s), '-' for stdin, or 'anchor'"
    )
    validate.set_defaults(func=_cmd_validate)

    compile_p = sub.add_parser("compile", help="compile a SafetySpec to its content-addressed IR")
    compile_p.add_argument("spec", nargs="?", default="anchor", help="spec path, '-', or 'anchor'")
    compile_p.add_argument("--out", help="write the compiled model here (default: stdout)")
    compile_p.add_argument(
        "--sample-period", type=float, default=1.0, help="compiled sample period in seconds"
    )
    compile_p.set_defaults(func=_cmd_compile)

    falsify = sub.add_parser(
        "falsify",
        help="run the seeded adversarial search against a SafetySpec",
        description="Search for a counterexample to a SafetySpec: an unshielded control attack "
        "that "
        "must breach (or the result proves nothing), then the same attack behind the shield. The "
        "start and the attack are derived from the spec's own safe set — there is no scenario to "
        "bind, because the plant is synthetic.",
    )
    falsify.add_argument("spec", nargs="?", default="anchor", help="spec path, '-', or 'anchor'")
    falsify.add_argument(
        "--seed", type=int, action="append", help="seed to test (repeatable; overrides --trials)"
    )
    falsify.add_argument("--trials", type=int, default=8, help="number of seeds 0..N-1 to test")
    falsify.add_argument("--horizon", type=int, default=120, help="rollout horizon in ticks")
    falsify.add_argument(
        "--sample-period",
        type=float,
        default=_ANCHOR_FALSIFY_PERIOD_S,
        help="compiled sample period (coarse keeps the survival buffers small)",
    )
    falsify.set_defaults(func=_cmd_falsify)

    sign = sub.add_parser("sign", help="sign a SafetySpec's content hash (offline dev signer)")
    sign.add_argument("spec", nargs="?", default="anchor", help="spec path, '-', or 'anchor'")
    sign.add_argument(
        "--key", type=Path, default=_DEFAULT_KEY, help="private key PEM (default: anchor dev key)"
    )
    sign.add_argument(
        "--pub", type=Path, default=_DEFAULT_PUB, help="public key PEM for --verify (pinned trust)"
    )
    sign.add_argument(
        "--verify", action="store_true", help="re-load through the fail-closed signed-load gate"
    )
    sign.set_defaults(func=_cmd_sign)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
