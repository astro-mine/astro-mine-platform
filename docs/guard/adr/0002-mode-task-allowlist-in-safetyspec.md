# ADR-0002 — The MODE/TASK allowlist lives in the `SafetySpec`; `CoreConfig` may only narrow it

- **Status:** Accepted
- **Date:** 2026-07-13
- **Component:** `astro-mine-guard` — the safety contract (`spec/`) and the Rust action gate
  (`rust/src/arbiter.rs`)
- **Traceability:** `RM-P1-GUARD-03`; [RFC-0004 Amendment 2][rfc-0004] (the ratifying artifact);
  `guard.md` §3, §5, §9.1, §9.3; `conventions.md` §9; astro-mine-guard#25 (raised by
  astro-mine-guard#24)

## Context

RM-P1-GUARD-03 extended `PolicyShield` to classify **every** action kind — there is no pass-through
branch left in the marshal layer. A `MODE` or `TASK` proposal carries no continuous quantity to
project, so the shield cannot *correct* it; the trusted core can only certify it by **enumeration**
against an allowlist. That allowlist shipped in `CoreConfig.action_policy`
(`certified_modes` / `certified_tasks`), and the code said so, with a note deferring the question:

> *Deferred (RFC-gated):* promoting the allowlist into the `SafetySpec` itself — so it is
> content-addressed and reviewed alongside the keep-out geometry — is an additive, RFC-gated change
> to a safety contract (guard.md §3, §5) and lands with that RFC.

This ADR resolves that deferral. Four facts about the shipped code decide it.

**1. An allowlisted directive is a `passthrough` — the one thing RFC-0004 designed out of the
schema.** RFC-0004 is explicit that `OnUncertain` has **no `passthrough` member** and that "let the
policy's action through unchecked" is *not expressible*. But an admitted directive resolves in the
arbiter to `Intervention::None` / `Reason::Certified` with an **empty** `certified_action`, and
`wrap/shield.py` then re-emits the wrapped policy's proposal **byte-for-byte, uncertified**. That is
a passthrough by any operational definition. The schema kept the door shut; configuration had cut a
new one beside it, and the only thing standing in it was a dict of plugin `params`.

**2. A MODE transition is an actuation path with direct safety semantics.** `ModeCommand.mode` names
a SADF `loads_by_mode` mode, and Sim's engines (`mobility`, `granular`, `manipulation`) switch
behavior on it. `loads_by_mode` **is** the power/thermal load profile the anchor spec's survival
floors are stated against — so a MODE switch is the most direct available way to invalidate the very
constraints Guard is enforcing (leaving a survival-heater profile during lunar night, say). It is
also the only **prospective** control on that path: the monitors are `ScalarBound`s over *measured*
signals and observe the consequence one or more ticks *after* the load profile has already changed.

**3. The hole was already open, with a wrong answer in it.** The Mind reference stack
(`astro-mine-mind`, `reference/stacks/lunar_prospecting_anchor.yaml`) shipped
`certified_modes: ["velocity"]`, justified in-file with *"Sim's engine actuates VELOCITY setpoints,
so VELOCITY is the mode this stack certifies."* That rationale is **factually wrong**:
`certified_modes` gates `ModeCommand.mode` *names*, not `ControlMode` values. `ActionCodec.classify`
routes an `ACTUATOR`/`VELOCITY` action to the shield as a **numeric** command, and the arbiter never
consults the allowlist for it. So the entry was (a) **inert** for the actions the stack actually
emits (`ActionKind.ACTUATOR`), and (b) a **standing grant** that any `MODE` directive happening to be
named `"velocity"` would cross the TCB untouched. A safety-relevant permission, granted in a YAML, in
a different repo, on a mistaken premise, caught by nothing. This is not a hypothetical failure mode
being designed against — it is the one that already happened.

**4. `SafetySpec` is the only artifact on this path with an integrity story.** It is
content-addressed (`content_hash_json`), signed and fail-closed-verified before the TCB sees it
(`spec/signed.py`), **re-derived inside** the trusted core, and stamped into every `SafetyVerdict` as
`spec_content_hash`. `CoreConfig` has none of that: it is an unsigned `params` dict from a stack YAML
and appears nowhere in the verdict record. Before this change, two runs could report the **same**
`spec_content_hash` and enforce **different** action gates — which breaks the guarantee `guard.md` §5
and §9.3 rest on.

## Decision

**The grant moves into the reviewed contract. The configuration knob survives, narrowing-only.**

A new *optional* `SafetySpec.admissible_directives{ modes: [str], tasks: [TaskKind] }` carries the
reviewed grant, lowered by the compiler to `CompiledSafetyModel.admissible_directives` and decoded
fail-closed in the TCB. `CoreConfig.action_policy` is retained, but the gate's effective allowlist is
now the **intersection**:

```
effective = spec.admissible_directives  ∩  config.action_policy

spec silent (field absent)      ⇒ effective = ∅     (NOT "whatever config says")
spec authors ∅                  ⇒ effective = ∅
config silent (empty allowlist) ⇒ effective = ∅
```

A directive is certifiable **iff the reviewed spec admits it *and* the configuration admits it**.
Configuration can *tighten* the reviewed grant — the legitimate "run this deployment stricter than the
contract allows" case — and can no longer **create** a permission. The intersection is computed
**once**, in `SafetyCore::from_model`, so the hot path stays allocation-free (`guard.md` §2
principle 6).

Ratified as [RFC-0004 Amendment 2][rfc-0004]. Additive: `safety_version` stays `"0.1"`, and there is
**no `astro-mine-core` change** — Guard's schema `$ref`s Core's existing `TaskKind` by its published
`$id` (RFC-0009 §1), which is a *read* of Core's public vocabulary, not a change to it.

## Rationale

**Why the tighten-only precedent applies — and where it deliberately does not.** `kinematic_limit`
already establishes the rule: a spec-authored ceiling `tighten()`s the configured one
(`min(config, authored)`, `rust/src/shield.rs`) and can never loosen it. `admissible_directives` is
the same rule on a different lattice — and it merges the **other way round on silence**:

| merged thing | greatest lower bound | identity ("no opinion") | spec silent ⇒ |
|---|---|---|---|
| a scalar **ceiling** (`kinematic_limit` → `ActionLimits`) | `min(config, authored)` | `+∞` | **config stands** |
| a **permission set** (`admissible_directives`) | `config ∩ authored` | `∅` | **nothing is admitted** |

Both are the greatest-lower-bound of the two inputs, so "configuration may only tighten the reviewed
contract" is *one* rule, not two. Only the **identity element of the meet** differs. An unstated
ceiling is `+∞` (no constraint), so `min(config, absent) = config` is safe. An unstated permission set
is `∅` (no authority), so `config ∩ absent = ∅` — **silence must grant nothing**. Reading a silent
spec as "config stands" would make the permission set fail *open* by silence, preserve exactly the
unreviewed grant this change exists to revoke, and make "add a permission" achievable by **deleting**
a line from the contract. This asymmetry is the load-bearing rule; it is stated explicitly here, and
in the RFC, so it is not re-derived by analogy to `tighten()` at some later review.

**It is machine-checked, not asserted.** `rust/src/verify.rs` carries a Kani harness over a *symbolic*
directive name: **no configuration, however permissive, admits a directive the model does not author**,
and a spec-silent model admits nothing. That is the whole point of the change, discharged as a theorem
over every directive name rather than a handful of sampled literals — the same discipline the rest of
the gate is held to (`gate_admits_nothing_by_default`, `gate_never_admits_an_unmodelled_action`).

**Why an optional top-level field and not a new `ConstraintKind`.** A `Constraint` in this vocabulary
is a **predicate over a declared signal**, carrying an `on_uncertain` selector naming the verified
backup to run when it fires. A directive allowlist has no signal, no predicate, and no meaningful
`on_uncertain` (a rejected directive is not a *fired* constraint — it is an **absent certificate**).
Forcing it into the tagged union would pollute the constraint vocabulary and every compiler and
monitor path that consumes it with a member that is not a constraint. `safe_pose` (RFC-0004
Amendment 1) faced the identical question and answered it the same way: a spec-level fact that is
neither a predicate nor a geometry is a **top-level optional field**.

**The counter-argument, recorded.** The MODE vocabulary is **open** — mode names are free strings
drawn from a SADF asset's `loads_by_mode` — so an allowlist over them is asset- and
deployment-specific, and pinning it in a content-addressed spec means a **new spec hash and a re-sign
for every fleet variant**. That is a real ergonomic cost and it is the strongest case for leaving the
allowlist in configuration. It does not carry. `SafetySpec` **already** holds fleet-specific content
(the torque ceiling, the keep-out geometry, the `safe_pose`); `scenario_ref` exists precisely to bind
a spec to the scenario it is stated against; and the narrowing-only knob preserves the "run stricter
than the contract" use-case with **no** re-sign. The friction lands only on the operation that *should*
be expensive: granting a robot new authority.

## Consequences

- **A spec that authors no `admissible_directives` certifies no directive**, whatever `CoreConfig`
  says. Every spec written before this change is silent, so this is a **behavioural** change for any
  deployment that was relying on a config-only grant. The only such deployment in the workspace was
  the Mind reference stack, whose grant was inert and mistaken (Context #3); it is removed rather than
  re-authored. Any deployment that genuinely needs a directive grant now authors it in the spec, where
  it is reviewed, hashed, signed, and bound into every `SafetyVerdict`.
- **The spec's content hash now bounds the gate.** Two runs reporting the same `spec_content_hash` can
  no longer enforce different action gates: configuration can only have narrowed the grant the hash
  commits to. This is what closes the `guard.md` §5 / §9.3 gap.
- **Additive, and the CI guards prove it.** `tests/test_schema_compat.py` (enums append-only, no new
  *required* field), `scripts/check_model_drift.py`, and `buf breaking` all stay green — an optional
  field with a default satisfies all three, which is the mechanical proof the schema change is
  additive.
- **`fallback_control_mode` / `fallback_target` deliberately stay in `CoreConfig`.** They are not a
  *permission*: they name the actuation **channel** a rejected proposal is answered in (a
  velocity-tracking plant must be answered with a velocity command, not an `EFFORT` brake its actuator
  would ignore). That is a property of the plant and its wiring, not of the safety contract, and
  `ActionPolicy.__post_init__` already constrains it to the control modes the TCB has a plant model
  for. No setting of it can widen what is *certifiable*.
- **Deferred.** Validating authored MODE names against a SADF asset's `loads_by_mode` (needs a
  spec↔asset binding the loader does not have); per-regime directive profiles (RFC-0001, P3).

[rfc-0004]: https://github.com/astro-mine/docs/blob/main/rfc/0004-safetyspec-safety-contract.md
