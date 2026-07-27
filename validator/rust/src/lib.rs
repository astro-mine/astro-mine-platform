//! Rust fast-path validator for the Astro-Mine-Core JSON Schemas (RM-P0-CORE-07, RM-P1-CORE-08).
//!
//! The Python reference loader (`astro_mine.core.<comp>.loader`) is authoritative; this is
//! the optional Rust fast path (core.md §8, §11) and MUST produce identical *structural*
//! verdicts. It mirrors the Python `_check_structural` step: compile each Core JSON Schema
//! with the `jsonschema` crate (Draft 2020-12) and validate a document against it. Every
//! Core JSON Schema family is covered — SADF, ObjectiveSpec, the control-plane message
//! catalog, the plugin manifest, MissionSpec, PolicyPackage, RunProvenance and Plan — and a
//! Python-side drift guard (`tests/test_validator_rust_parity.py`) fails if a new Core schema
//! lands without one here.
//!
//! The semantic checks layered on top in Python (the dual-use `operational_targeting` gate,
//! unique-id and tagged-union payload coupling, the Plan contingency-label rules) stay in the
//! reference loader and are deliberately out of scope here — this is a structural pre-filter,
//! not a replacement.
//!
//! The RFC-0007 units/frames/time **guards** (`require_frame` / `require_crs` / …) are a
//! different contract — semantic, not schema-shaped — and live in [`units`], which runs the
//! shared `units/schema/conformance.json` vectors in this crate's own test suite.

pub mod units;

use jsonschema::{Resource, Validator};
use serde_json::{json, Value};
use std::collections::BTreeMap;

const SADF: &str = include_str!("../../../src/astro_mine/core/sadf/schema/sadf.schema.json");
const OBJECTIVE: &str =
    include_str!("../../../src/astro_mine/core/objective/schema/objective.schema.json");
const MESSAGES: &str =
    include_str!("../../../src/astro_mine/core/messages/schema/messages.schema.json");
const MANIFEST: &str =
    include_str!("../../../src/astro_mine/core/registry/schema/manifest.schema.json");
const MISSION: &str =
    include_str!("../../../src/astro_mine/core/mission/schema/mission.schema.json");
const POLICY_PACKAGE: &str =
    include_str!("../../../src/astro_mine/core/policy/schema/policy_package.schema.json");
const RUN_PROVENANCE: &str =
    include_str!("../../../src/astro_mine/core/provenance/schema/run_provenance.schema.json");
const PLAN: &str = include_str!("../../../src/astro_mine/core/plan/schema/plan.schema.json");
const UNITS: &str = include_str!("../../../src/astro_mine/core/units/schema/units.schema.json");

/// `messages.schema.json` and `mission.schema.json` `$ref` the shared `units.schema.json`
/// across files (RFC-0007). They name it *absolutely*, by the units schema's own `$id`, so
/// registering the vocabulary under that URI resolves the cross-file ref — mirroring the
/// Python `units_registry` (astro_mine.core.units.schema_ref).
///
/// This is the units `$id` verbatim, not a URI synthesized from a consumer's base;
/// `tests/test_validator_rust_parity.py` asserts the two never drift apart.
const UNITS_REF_URI: &str = "https://schemas.astro-mine.org/core/units/v0.1/units.schema.json";

/// Control-plane message types validated by name. Mirrors the Python `validate_action_batch`
/// / `validate_contact_plan` entry points; the per-tick hot-path families are Cap'n Proto
/// (not JSON Schema) and are intentionally absent.
const MESSAGE_ROOTS: [&str; 2] = ["ActionBatch", "ContactPlan"];

/// Compiled Core schema validators, keyed by logical name: `"sadf"`, `"objective"`,
/// `"manifest"`, `"mission"`, `"policy_package"`, `"run_provenance"`, `"plan"`, and
/// `"messages.<Type>"` for each control-plane root.
pub struct CoreValidators {
    by_name: BTreeMap<String, Validator>,
}

impl CoreValidators {
    /// Compile every bundled Core schema. Panics if a bundled schema is not valid JSON
    /// Schema — that is a build-time defect in Core, never a runtime input error.
    pub fn new() -> Self {
        let mut by_name = BTreeMap::new();
        by_name.insert("sadf".to_string(), compile(&parse(SADF)));
        by_name.insert("objective".to_string(), compile(&parse(OBJECTIVE)));
        // `manifest` is a rooted document schema (required: manifest_version + manifest),
        // so validate against the whole schema, exactly as the Python registry loader does
        // (registry/loader.py `_check_structural`).
        by_name.insert("manifest".to_string(), compile(&parse(MANIFEST)));
        // The other rooted document schemas, each mirroring its Python loader's structural
        // step: policy/loader.py, provenance/loader.py, plan/loader.py.
        by_name.insert(
            "policy_package".to_string(),
            compile(&parse(POLICY_PACKAGE)),
        );
        by_name.insert(
            "run_provenance".to_string(),
            compile(&parse(RUN_PROVENANCE)),
        );
        by_name.insert("plan".to_string(), compile(&parse(PLAN)));
        // `mission` is rooted too, but — like `messages` — it $refs the shared units
        // vocabulary across files (RFC-0007), so it compiles with the units resource
        // registered (mission/loader.py passes `units_registry(schema)` for the same reason).
        // Its own `$id` is the base the relative ref resolves against, so no wrapper is needed.
        by_name.insert("mission".to_string(), compile_with_units(&parse(MISSION)));
        // `messages` is a $defs-only catalog with no root document; validate each
        // control-plane type against `{$ref: #/$defs/<root>, $defs: ...}`, exactly as the
        // Python `_root_validator` does (messages/loader.py). The wrapper carries the
        // messages `$id` so the cross-file units `$ref` (RFC-0007) resolves deterministically.
        let messages = parse(MESSAGES);
        let messages_id = messages
            .get("$id")
            .and_then(Value::as_str)
            .expect("messages schema has an $id");
        let defs = messages.get("$defs").cloned().unwrap_or(Value::Null);
        for root in MESSAGE_ROOTS {
            let wrapped = json!({
                "$schema": "https://json-schema.org/draft/2020-12/schema",
                "$id": messages_id,
                "$ref": format!("#/$defs/{root}"),
                "$defs": defs,
            });
            by_name.insert(format!("messages.{root}"), compile_with_units(&wrapped));
        }
        Self { by_name }
    }

    /// The schema names this validator knows, sorted.
    pub fn names(&self) -> Vec<&str> {
        self.by_name.keys().map(String::as_str).collect()
    }

    /// True iff `document` satisfies the named schema. Panics if `name` is unknown.
    pub fn is_valid(&self, name: &str, document: &Value) -> bool {
        self.validator(name).is_valid(document)
    }

    /// Validate `document`, returning the schema-validation error messages (empty == valid).
    /// Panics if `name` is unknown.
    pub fn validate(&self, name: &str, document: &Value) -> Vec<String> {
        self.validator(name)
            .iter_errors(document)
            .map(|e| e.to_string())
            .collect()
    }

    fn validator(&self, name: &str) -> &Validator {
        self.by_name
            .get(name)
            .unwrap_or_else(|| panic!("unknown Core schema {name:?}; known: {:?}", self.names()))
    }
}

impl Default for CoreValidators {
    fn default() -> Self {
        Self::new()
    }
}

fn parse(src: &str) -> Value {
    serde_json::from_str(src).expect("bundled Core schema is valid JSON")
}

fn compile(schema: &Value) -> Validator {
    jsonschema::validator_for(schema).expect("bundled Core schema is a valid JSON Schema")
}

/// Compile a schema that `$ref`s the shared units vocabulary across files (RFC-0007),
/// registering `units.schema.json` under the URI the relative ref resolves to so
/// `jsonschema` deep-validates the nested units types (the Rust analogue of the Python
/// `units_registry`).
fn compile_with_units(schema: &Value) -> Validator {
    let units = Resource::from_contents(parse(UNITS)).expect("units schema is a valid resource");
    jsonschema::options()
        .with_resource(UNITS_REF_URI, units)
        .build(schema)
        .expect("bundled Core schema is a valid JSON Schema")
}

#[cfg(test)]
mod tests {
    use super::*;

    // The same checked-in corpus the Python consistency test loads. Parity claim: the Rust
    // fast path must agree with the Python reference's structural verdict on each.
    const SADF_EXAMPLES: [&str; 2] = [
        include_str!("../../../examples/assets/lunar-scout-rover.sadf.yaml"),
        include_str!("../../../examples/assets/neo-sep-carrier.sadf.yaml"),
    ];
    const OBJECTIVE_EXAMPLES: [&str; 1] = [include_str!(
        "../../../examples/objectives/lunar-polar-ice-prospecting.objective.yaml"
    )];
    const MANIFEST_EXAMPLES: [&str; 2] = [
        include_str!("../../../examples/plugins/greedy-prospecting-baseline.manifest.yaml"),
        include_str!("../../../examples/plugins/lunar-terramechanics-engine.manifest.yaml"),
    ];
    const MISSION_EXAMPLES: [&str; 2] = [
        include_str!("../../../examples/mission/lunar-surface-single-phase.mission.yaml"),
        include_str!("../../../examples/mission/neo-sample-return-multiphase.mission.yaml"),
    ];
    const POLICY_PACKAGE_EXAMPLES: [&str; 2] = [
        include_str!("../../../examples/policy/greedy-prospecting-baseline.policy-package.yaml"),
        include_str!("../../../examples/policy/minimal.policy-package.yaml"),
    ];
    const RUN_PROVENANCE_EXAMPLES: [&str; 2] = [
        include_str!("../../../examples/run-provenance/full.run-provenance.yaml"),
        include_str!("../../../examples/run-provenance/minimal.run-provenance.yaml"),
    ];
    const PLAN_EXAMPLES: [&str; 2] = [
        include_str!("../../../examples/plan/lunar-prospecting-contingent.plan.yaml"),
        include_str!("../../../examples/plan/standing-control.plan.yaml"),
    ];

    fn yaml(src: &str) -> Value {
        serde_yaml::from_str(src).expect("example corpus is valid YAML")
    }

    /// Every example in a family validates against its schema — the parity claim, asserted
    /// against the same checked-in files the Python reference loads.
    fn assert_corpus_is_valid(name: &str, corpus: &[&str]) {
        let v = CoreValidators::new();
        for (i, src) in corpus.iter().enumerate() {
            let errs = v.validate(name, &yaml(src));
            assert!(
                errs.is_empty(),
                "{name} example #{i} should be valid: {errs:?}"
            );
        }
    }

    #[test]
    fn known_schema_names_are_registered() {
        let v = CoreValidators::new();
        for name in [
            "sadf",
            "objective",
            "manifest",
            "mission",
            "policy_package",
            "run_provenance",
            "plan",
            "messages.ActionBatch",
            "messages.ContactPlan",
        ] {
            assert!(v.names().contains(&name), "missing schema {name}");
        }
    }

    #[test]
    fn sadf_corpus_is_structurally_valid() {
        // Parity with the Python reference: every checked-in SADF example validates.
        assert_corpus_is_valid("sadf", &SADF_EXAMPLES);
    }

    #[test]
    fn objective_corpus_is_structurally_valid() {
        assert_corpus_is_valid("objective", &OBJECTIVE_EXAMPLES);
    }

    #[test]
    fn manifest_corpus_is_structurally_valid() {
        // Parity with the Python reference: every checked-in plugin-manifest example
        // validates (registry/loader.py validates the same corpus the consistency test loads).
        assert_corpus_is_valid("manifest", &MANIFEST_EXAMPLES);
    }

    #[test]
    fn mission_corpus_is_structurally_valid() {
        // Parity with tests/test_mission_consistency.py, which loads the same two examples.
        // Also proves the cross-file units `$ref` (RFC-0007) resolves: the multi-phase example
        // carries typed `epoch` / `frame_ref` siblings that only validate through units.schema.json.
        assert_corpus_is_valid("mission", &MISSION_EXAMPLES);
    }

    #[test]
    fn policy_package_corpus_is_structurally_valid() {
        // Parity with tests/test_policy.py::test_examples_corpus_loads.
        assert_corpus_is_valid("policy_package", &POLICY_PACKAGE_EXAMPLES);
    }

    #[test]
    fn run_provenance_corpus_is_structurally_valid() {
        // Parity with tests/test_provenance_consistency.py, which loads the same two examples.
        assert_corpus_is_valid("run_provenance", &RUN_PROVENANCE_EXAMPLES);
    }

    #[test]
    fn plan_corpus_is_structurally_valid() {
        // Parity with tests/test_plan.py::test_examples_corpus_loads.
        assert_corpus_is_valid("plan", &PLAN_EXAMPLES);
    }

    #[test]
    fn empty_documents_are_rejected() {
        // Parity: every rooted document schema declares required members, so `{}` fails
        // validation (mirrors the Python loader tests that expect `load_*("{}")` to raise).
        let v = CoreValidators::new();
        for name in [
            "sadf",
            "objective",
            "manifest",
            "mission",
            "policy_package",
            "run_provenance",
            "plan",
        ] {
            assert!(!v.is_valid(name, &json!({})), "{name} should reject {{}}");
        }
    }

    #[test]
    fn document_versions_are_pinned() {
        // Parity: every rooted document pins its schema minor with a `const`, so a document
        // claiming another version is rejected (mirrors the `bad-version-const` corpus cases
        // in the Python consistency tests and test_plan.py::test_wrong_version_is_rejected).
        let v = CoreValidators::new();
        for (name, doc) in [
            (
                "mission",
                json!({"mission_version": "0.2", "mission": {"id": "m", "name": "M"}}),
            ),
            (
                "policy_package",
                json!({
                    "policy_package_version": "0.2",
                    "policy_package": {"name": "p", "version": "0.1.0", "onnx_model": {"digest": "sha256:aa"}},
                }),
            ),
            (
                "run_provenance",
                json!({"run_provenance_version": "0.2", "run_provenance": {}}),
            ),
            (
                "plan",
                json!({
                    "plan_version": "0.2",
                    "plan": {"base": {"plan_id": "p", "tier": "control", "validity": {"issued_at_s": 0.0}}},
                }),
            ),
        ] {
            assert!(
                !v.is_valid(name, &doc),
                "{name} should reject a bad version const"
            );
        }
    }

    #[test]
    fn unknown_fields_are_rejected() {
        // Parity: every Core document schema is `additionalProperties: false`, matching the
        // Pydantic models' `extra="forbid"` (the `unknown-*` cases in the Python corpora).
        let v = CoreValidators::new();
        for (name, doc) in [
            (
                "mission",
                json!({"mission_version": "0.1", "mission": {"id": "m", "name": "M", "bogus": 1}}),
            ),
            (
                "policy_package",
                json!({
                    "policy_package_version": "0.1",
                    "policy_package": {
                        "name": "p", "version": "0.1.0",
                        "onnx_model": {"digest": "sha256:aa"}, "bogus": 1,
                    },
                }),
            ),
            (
                "run_provenance",
                json!({"run_provenance_version": "0.1", "run_provenance": {"bogus": 1}}),
            ),
            (
                "plan",
                json!({
                    "plan_version": "0.1",
                    "plan": {
                        "base": {
                            "plan_id": "p", "tier": "control",
                            "validity": {"issued_at_s": 0.0}, "bogus": 1,
                        },
                    },
                }),
            ),
        ] {
            assert!(
                !v.is_valid(name, &doc),
                "{name} should reject an unknown field"
            );
        }
    }

    #[test]
    fn mission_rejects_a_bad_regime_and_maneuver_type() {
        // The same invalid documents as the Python `bad-regime` / `bad-maneuver-type` corpus
        // cases, which expect MissionValidationError.
        let v = CoreValidators::new();
        let bad_regime = json!({
            "mission_version": "0.1",
            "mission": {"id": "m", "name": "M", "phases": [{"id": "p", "regime": "hyperspace"}]},
        });
        assert!(!v.is_valid("mission", &bad_regime));

        let bad_maneuver = json!({
            "mission_version": "0.1",
            "mission": {
                "id": "m", "name": "M",
                "phases": [{
                    "id": "p", "regime": "interplanetary_transit",
                    "legs": [{"id": "l", "trajectory_ref": {"id": "t", "frame": "J2000", "maneuvers": [{
                        "epoch_tdb_s": 0.0, "delta_v_mps": 1.0,
                        "direction": {"x": 0.0, "y": 0.0, "z": 1.0}, "maneuver_type": "warp",
                    }]}}],
                }],
            },
        });
        assert!(!v.is_valid("mission", &bad_maneuver));
    }

    #[test]
    fn mission_enforces_the_shared_units_vocabulary() {
        // RFC-0007: the typed `epoch` sibling resolves through the cross-file $ref, so a bad
        // time scale is rejected *here*, not only in the Python reference (the
        // `bad-maneuver-epoch-scale` corpus case). A validator compiled without the units
        // resource would silently accept it — this is the test that pins that.
        let v = CoreValidators::new();
        let leg = |scale: &str| {
            json!({
                "mission_version": "0.1",
                "mission": {
                    "id": "m", "name": "M",
                    "phases": [{
                        "id": "p", "regime": "interplanetary_transit",
                        "legs": [{"id": "l", "trajectory_ref": {"id": "t", "frame": "J2000", "maneuvers": [{
                            "epoch_tdb_s": 0.0, "delta_v_mps": 1.0,
                            "direction": {"x": 0.0, "y": 0.0, "z": 1.0},
                            "maneuver_type": "impulsive",
                            "epoch": {"tdb_seconds": 0.0, "scale": scale},
                        }]}}],
                    }],
                },
            })
        };
        assert!(v.is_valid("mission", &leg("et")), "et is a valid TimeScale");
        assert!(
            !v.is_valid("mission", &leg("utc")),
            "utc is not a TimeScale"
        );
    }

    #[test]
    fn policy_package_rejects_a_bad_tensor_dtype_and_missing_model() {
        // The same invalid documents as tests/test_policy.py's rejection tests.
        let v = CoreValidators::new();
        let bad_dtype = json!({
            "policy_package_version": "0.1",
            "policy_package": {
                "name": "p", "version": "0.1.0",
                "onnx_model": {"digest": "sha256:aa"},
                "io_signature": {"inputs": [{"name": "obs", "dtype": "float16", "shape": [1]}]},
            },
        });
        assert!(!v.is_valid("policy_package", &bad_dtype));

        let no_model = json!({
            "policy_package_version": "0.1",
            "policy_package": {"name": "p", "version": "0.1.0"},
        });
        assert!(!v.is_valid("policy_package", &no_model));
    }

    #[test]
    fn run_provenance_rejects_bad_member_types() {
        // The `bad-seed-type` / `bad-seeds-value` / `outcome-missing-verdict` corpus cases.
        let v = CoreValidators::new();
        for payload in [
            json!({"seed": "not-an-int"}),
            json!({"seeds": {"episode": "x"}}),
            json!({"engine_versions": {"orbital": 3}}),
            json!({"error_budget_outcomes": [{"name": "x"}]}),
        ] {
            let doc = json!({"run_provenance_version": "0.1", "run_provenance": payload});
            assert!(!v.is_valid("run_provenance", &doc), "should reject {doc}");
        }
    }

    #[test]
    fn plan_requires_a_validity_and_a_base() {
        let v = CoreValidators::new();
        let no_validity = json!({
            "plan_version": "0.1",
            "plan": {"base": {"plan_id": "p", "tier": "control"}},
        });
        assert!(!v.is_valid("plan", &no_validity));
        assert!(!v.is_valid("plan", &json!({"plan_version": "0.1", "plan": {}})));
        // A standing plan (`horizon_s: null`) is valid; a non-positive horizon is not.
        let horizon = |h: Value| {
            json!({
                "plan_version": "0.1",
                "plan": {"base": {
                    "plan_id": "p", "tier": "control",
                    "validity": {"issued_at_s": 0.0, "horizon_s": h},
                }},
            })
        };
        assert!(v.is_valid("plan", &horizon(Value::Null)));
        assert!(v.is_valid("plan", &horizon(json!(1.0))));
        assert!(!v.is_valid("plan", &horizon(json!(0.0))));
    }

    #[test]
    fn plan_semantic_rules_stay_in_the_reference_loader() {
        // The documented boundary (see the module header): duplicate contingency triggers are a
        // *semantic* rule the Python loader enforces (plan/loader.py `_check_semantics`) and JSON
        // Schema cannot express. The fast path is a structural pre-filter, so it accepts this —
        // pinning the divergence so it stays deliberate rather than becoming a silent gap.
        let v = CoreValidators::new();
        let dupes = json!({
            "plan_version": "0.1",
            "plan": {
                "base": {"plan_id": "p", "tier": "control", "validity": {"issued_at_s": 0.0}},
                "branches": [
                    {"trigger": "comms_lost", "action": "hold_cached"},
                    {"trigger": "comms_lost", "action": "safe_idle"},
                ],
            },
        });
        assert!(v.is_valid("plan", &dupes));
    }

    #[test]
    fn empty_message_batches_are_valid() {
        // Parity: the control-plane roots declare no required members, so an empty batch/plan
        // is structurally valid — the Python `validate_action_batch({})` / `validate_contact_plan({})`
        // both accept it. Asserting this pins the agreement, not an accident.
        let v = CoreValidators::new();
        for name in ["messages.ActionBatch", "messages.ContactPlan"] {
            assert!(
                v.is_valid(name, &json!({})),
                "{name} should accept an empty batch"
            );
        }
    }

    #[test]
    fn action_batch_rejects_unknown_kind() {
        // The same invalid document as the Python test_messages_validators_are_implemented,
        // which expects MessagesValidationError.
        let v = CoreValidators::new();
        let bad = json!({"actions": [{"kind": "warp"}]});
        assert!(!v.is_valid("messages.ActionBatch", &bad));
    }
}
