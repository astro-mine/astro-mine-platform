//! Guard's Rust binding of Core's units guards, pinned to Core's shared conformance vectors.
//!
//! RFC-0007 (Design §3) ships the frame/CRS/time guard rules as a data file
//! (`units/schema/conformance.json`, RM-P1-CORE-08) and requires each non-Python binding to run
//! it in its own CI. Guard's Rust safety core is one of the two designated non-Python targets, so
//! this test loads Core's vectors (vendored at `tests/vectors/conformance.json`, pinned to the
//! Core rev this repo depends on) and asserts every case's verdict — return vs raise — matches
//! `astro_mine_guard_core::units`, proving parity with the Python reference across all six guard
//! rules, including `ET ≡ TDB` (rule 3).

use astro_mine_guard_core::units::{
    require_crs, require_epoch, require_epoch_window, require_frame, require_si_unit,
    scales_equivalent,
};
use serde_json::Value;

fn vectors() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/vectors/conformance.json"
    );
    let text = std::fs::read_to_string(path).expect("read vendored conformance.json");
    serde_json::from_str(&text).expect("parse conformance.json")
}

/// A `center`-style optional-string field: absent or JSON null → `None`; a non-string present value
/// is malformed (returns `Some(Err)` so the guard rejects it).
fn opt_str<'a>(obj: &'a serde_json::Map<String, Value>, key: &str) -> Result<Option<&'a str>, ()> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s.as_str())),
        Some(_) => Err(()),
    }
}

fn frame_ok(v: &Value) -> bool {
    let Some(obj) = v.as_object() else {
        return false;
    };
    let (Some(name), Some(fc)) = (
        obj.get("name").and_then(Value::as_str),
        obj.get("frame_class").and_then(Value::as_str),
    ) else {
        return false;
    };
    let center = match opt_str(obj, "center") {
        Ok(c) => c,
        Err(()) => return false,
    };
    require_frame(name, fc, center).is_ok()
}

fn epoch_ok(v: &Value) -> bool {
    let Some(obj) = v.as_object() else {
        return false;
    };
    if !obj
        .get("tdb_seconds")
        .map(Value::is_number)
        .unwrap_or(false)
    {
        return false;
    }
    match obj.get("scale").and_then(Value::as_str) {
        Some(scale) => require_epoch(scale).is_ok(),
        None => false,
    }
}

fn epoch_fields(v: &Value) -> Option<(String, f64)> {
    let e = v.as_object()?;
    let scale = e.get("scale")?.as_str()?.to_string();
    let tdb = e.get("tdb_seconds")?.as_f64()?;
    Some((scale, tdb))
}

fn window_ok(v: &Value) -> bool {
    let Some(obj) = v.as_object() else {
        return false;
    };
    let (Some((ss, st)), Some((es, et))) = (
        obj.get("start").and_then(epoch_fields),
        obj.get("end").and_then(epoch_fields),
    ) else {
        return false;
    };
    require_epoch_window(&ss, st, &es, et).is_ok()
}

fn crs_ok(v: &Value) -> bool {
    let Some(obj) = v.as_object() else {
        return false;
    };
    let (Some(body), Some(bff), Some(radius)) = (
        obj.get("body").and_then(Value::as_str),
        obj.get("body_fixed_frame").and_then(Value::as_str),
        obj.get("reference_radius_m").and_then(Value::as_f64),
    ) else {
        return false;
    };
    let (proj, datum) = match (opt_str(obj, "projection"), opt_str(obj, "datum")) {
        (Ok(p), Ok(d)) => (p, d),
        _ => return false,
    };
    require_crs(body, bff, radius, proj, datum).is_ok()
}

fn si_unit_ok(v: &Value) -> bool {
    match v.as_str() {
        Some(u) => require_si_unit(u).is_ok(),
        None => false,
    }
}

fn run_category(vectors: &Value, key: &str, guard: fn(&Value) -> bool) {
    let cases = vectors[key]
        .as_array()
        .unwrap_or_else(|| panic!("conformance.json missing category {key:?}"));
    assert!(!cases.is_empty(), "category {key:?} has no vectors");
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let expected = case["valid"].as_bool().unwrap();
        let got = guard(&case["value"]);
        assert_eq!(
            got, expected,
            "category {key:?} case {name:?}: guard returned {got}, expected {expected}"
        );
    }
}

#[test]
fn reference_frame_vectors_pass() {
    run_category(&vectors(), "reference_frame", frame_ok);
}

#[test]
fn epoch_vectors_pass_including_et_tdb_equivalence() {
    // Rule 3: an `et`-scaled epoch is accepted everywhere a `tdb` one is (the `et-alias` vector).
    run_category(&vectors(), "epoch", epoch_ok);
    assert!(scales_equivalent("et", "tdb"));
    assert!(scales_equivalent("tdb", "et"));
}

#[test]
fn epoch_window_vectors_pass() {
    run_category(&vectors(), "epoch_window", window_ok);
}

#[test]
fn planetary_crs_vectors_pass() {
    run_category(&vectors(), "planetary_crs", crs_ok);
    // Rule 4's non-finite radius is exercised in-language (JSON has no infinity literal), per the
    // conformance.json $comment.
    assert!(require_crs("MOON", "MOON_ME", f64::INFINITY, None, None).is_err());
    assert!(require_crs("MOON", "MOON_ME", 1_737_400.0, None, None).is_ok());
}

#[test]
fn si_unit_vectors_pass() {
    run_category(&vectors(), "si_unit", si_unit_ok);
}
