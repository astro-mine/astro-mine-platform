# astro-mine-guard

**Runtime safety assurance for [Astro-Mine](https://github.com/astro-mine).**
The verifiable shield that wraps any policy so declared hard constraints cannot be
violated: a declarative `SafetySpec`, a minimal Rust safety core (CBF-QP shields +
STL/MTL runtime monitors + a simplex backup + arbiter), and the `PolicyShield` wrapper
over the Core Policy/Planner API. A compromised or pathological policy can degrade
mission performance — never safety. Fail safe, never fail open. **Safety-critical.**

> **Status:** Phase 1. The `SafetySpec` schema/compiler (RM-P1-GUARD-01), the **trusted
> Rust safety core** (RM-P1-GUARD-02 — arbiter + CBF-QP shield + STL/MTL monitors + simplex
> backup + watchdog, with a PyO3 binding), the `PolicyShield` wrapper (RM-P1-GUARD-03/-06), the
> anchor safety content (RM-P1-GUARD-04), and **signed loading + adversarial falsification**
> (RM-P1-GUARD-05) are in. See the
> [architecture](https://github.com/astro-mine/docs/blob/main/architecture/guard.md)
> and [Phase-1 roadmap](https://github.com/astro-mine/docs/blob/main/roadmap/phase-1-autonomy-studio.md).

## Layout

```
src/astro_mine/guard/       # import path: astro_mine.guard (Python orchestration + spec)
rust/                       # the trusted safety core (TCB) — Rust crate, PyO3 extension
tests/                      # Python tests; rust/tests/ holds the core's Rust gates
```

The trusted core (`arbiter`, `shields`, `monitors`, `backup`, and the compiled `spec`
evaluator) is a small, deterministic, allocation-free **Rust** crate (`rust/`). It compiles and
tests standalone (no Python) for the edge, and the wheel is **maturin**-built so it bundles the
core as the `astro_mine.guard._core` PyO3 extension (guard.md §4). Remaining `RM-P1-GUARD-*`
modules (`models/ coord/ wrap/ audit/`, the `PolicyShield` wrapper) follow. See
[`architecture/guard.md`](https://github.com/astro-mine/docs/blob/main/architecture/guard.md).

## Development

Targets **Python 3.12** with a per-repo **conda** env and **uv**; the Rust core needs a
**stable Rust toolchain** (the maturin build compiles it during `uv sync`).

```bash
conda create -n astro-mine-guard python=3.12
conda activate astro-mine-guard
uv sync && uv run pytest          # builds + tests the Python side (incl. the PyO3 core)
cd rust && cargo test             # the standalone Rust TCB gates
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Command-line interface — `astro-mine-guard`

The spec tooling from a shell. Four verbs; pass `anchor` as the spec to use Guard's **shipped
reference anchor SafetySpec** (lunar polar water-ice prospecting), which resolves from package data
(`astro_mine.guard.reference.load_anchor_safety_spec()`) — no checkout required.

```bash
astro-mine-guard validate my.safety.yaml         # schema + fail-safe checks, actionable errors
astro-mine-guard compile  anchor --out m.json    # → content-addressed CompiledSafetyModel + hash
astro-mine-guard falsify  --trials 16            # seeded adversarial search on the anchor scenario
astro-mine-guard sign     anchor --verify        # offline cosign signature over the content hash
```

`validate`/`compile`/`sign` load **without** the compiled Rust core. `falsify` needs it for the
*shielded* verification; without it, the verb still runs the unshielded control search (proving the
search bites) and prints how to build the core — never a traceback. A falsification reports the
**seed** that produced any violation, so it is replayable (`--seed N`). `falsify` fails the run if
the search is vacuous or the shield admits a violation.

## Signed loading & falsification (RM-P1-GUARD-05)

The shield's correctness depends on the integrity of its inputs, so the two halves of the
safety case's integrity story are:

- **Signed loading (fail-closed).** `astro_mine.guard.spec.signed` refuses to load an unsigned
  or tampered `SafetySpec` / `CompiledSafetyModel`: it recomputes the artifact's content hash
  from the loaded bytes and verifies a keyed **cosign ECDSA P-256** signature (offline — no
  Fulcio/Rekor) covers *that* hash, before the trusted core sees the bytes. Verification is a
  **Python load gate — not a change to the Rust TCB**; the enforced guarantee stays in the core,
  which independently re-derives and reports the `spec_content_hash` it enforced into every
  verdict (verify-twice). The signing primitive **mirrors** `astro-mine-hub`'s ECDSA-P256 signer
  on the Core `Signature` type (the accepted `fleet → hub → guard` signer-duplication pattern —
  never a sibling import). `require_signature=False` stays the local/dev default (RFC-0004,
  opt-in signing); `astro-mine-guard sign` is the offline dev signer (the packaged home of the
  former `scripts/sign_spec.py`). Production trusted-key distribution beyond the dev key is decided
  with Hub (RFC-0004).
- **Adversarial falsification (the central validation gate).** `astro_mine.guard.falsify` is
  untrusted, stdlib-only tooling that *attacks the shield from outside*: a seeded, reproducible
  search over policy actions and disturbances drives a minimal double-integrator plant on the
  anchor and asserts **zero hard-constraint violations** — while the deliberately unshielded
  control run finds violations (the search is real). It runs in the ordinary CI pytest job; the
  real-Sim / real-ONNX and optimizer-based variants are `sim`/`slow`-marked and deselected.

## License

Apache-2.0 — see [LICENSE](LICENSE). Copyright Astro-Mine project contributors.
