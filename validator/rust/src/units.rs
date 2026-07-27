//! The RFC-0007 units / frames / time guards — Rust binding (RM-P1-CORE-08).
//!
//! `astro_mine.core.units.validate` is the **reference** implementation (core.md §8: "the Rust
//! validator is the recommended fast path; Python is the reference"); this module implements the
//! same six normative rules (RFC-0007 Design §3) in Rust and discharges the obligation the way
//! every binding must — by running the shared vectors, `units/schema/conformance.json`, in its own
//! CI (see the test module below). The rules, in brief:
//!
//! 1. A frame is present; `name` / `center` are non-empty, whitespace-free tokens.
//! 2. `frame_class` ∈ [`FrameClass`]; `scale` ∈ [`TimeScale`].
//! 3. [`TimeScale::Et`] and [`TimeScale::Tdb`] denote the **same** scale (SPICE ET ≡ TDB) —
//!    compare with [`scales_equivalent`], never `==`.
//! 4. A CRS is present; `body` / `body_fixed_frame` are tokens; `reference_radius_m` is finite
//!    and `> 0`.
//! 5. [`EpochWindow`] `start`/`end` are present and `end` is strictly after `start`.
//! 6. An Earth datum/projection marker (WGS84, EPSG:4326, urn:ogc:def:crs:OGC) is rejected when
//!    `body` is not `EARTH` and accepted when it is — an Earth CRS is not forbidden, an *implicit*
//!    one is. Refusing Earth CRSs outright is a component-local policy (e.g. View), not this rule.
//!
//! The guards mirror the reference's two entry paths: a JSON path (`require_*`, the analogue of the
//! Python mapping branch) and a typed path (`<Type>::new`, the analogue of the Pydantic model), so
//! a rule JSON cannot express — a non-finite reference radius — is still reachable in-language.
//! Unknown members are rejected, mirroring the Pydantic models' `extra="forbid"`.

use serde_json::{Map, Value};
use std::fmt;

/// Raised when a unit, frame, CRS, or epoch fails validation at the waist. The Rust analogue of
/// the reference's `UnitsValidationError`; the *message* is diagnostic, only the verdict is
/// contractual (the conformance vectors assert the verdict, not the wording).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitsValidationError(String);

impl UnitsValidationError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }

    /// The diagnostic message.
    pub fn message(&self) -> &str {
        &self.0
    }
}

impl fmt::Display for UnitsValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for UnitsValidationError {}

/// The result of a units guard.
pub type UnitsResult<T> = Result<T, UnitsValidationError>;

/// Wire spellings of [`TimeScale`] — mirrors the Python `TimeScale` StrEnum values. Closed;
/// append-only by RFC (a Python-vs-Rust drift guard asserts the two lists agree).
pub const TIME_SCALES: [&str; 2] = ["tdb", "et"];

/// Wire spellings of [`FrameClass`] — mirrors the Python `FrameClass` StrEnum values.
pub const FRAME_CLASSES: [&str; 3] = ["body_fixed", "inertial", "topocentric"];

/// The strict SI unit symbols Core recognizes — mirrors the Python `SiUnit` StrEnum values.
pub const SI_UNITS: [&str; 16] = [
    "m", "kg", "s", "A", "K", "mol", "cd", "rad", "sr", "Hz", "N", "Pa", "J", "W", "V", "C",
];

/// Recognized dimensionless ratio markers — mirrors the Python `DIMENSIONLESS_UNITS` set.
pub const DIMENSIONLESS_UNITS: [&str; 3] = ["dimensionless", "mass_fraction", "volume_fraction"];

/// The NAIF body name an Earth datum/projection marker is legitimate on (rule 6).
pub const EARTH: &str = "EARTH";

/// Case-insensitive substrings that mark an Earth datum/projection (rule 6). Not a closed list of
/// every Earth CRS — the common defaulting-bug markers a lunar/other-body product must never carry.
const EARTH_CRS_MARKERS: [&str; 3] = ["wgs84", "epsg:4326", "urn:ogc:def:crs:ogc"];

/// Admissible epoch time scale at the waist (rule 2). Only the SI-second, SPICE ephemeris scales
/// are representable, so a civil/atomic scale (UTC/TAI/…) cannot be smuggled in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TimeScale {
    /// Barycentric Dynamical Time (`tdb`).
    Tdb,
    /// SPICE Ephemeris Time (`et`) — the same scale as [`TimeScale::Tdb`] (rule 3).
    Et,
}

impl TimeScale {
    /// Parse a wire spelling; `None` for anything outside the closed vocabulary.
    pub fn from_wire(value: &str) -> Option<Self> {
        match value {
            "tdb" => Some(Self::Tdb),
            "et" => Some(Self::Et),
            _ => None,
        }
    }

    /// The wire spelling.
    pub const fn as_wire(self) -> &'static str {
        match self {
            Self::Tdb => "tdb",
            Self::Et => "et",
        }
    }
}

/// Reference-frame class (rule 2).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrameClass {
    /// Rotates with a body (e.g. `MOON_ME`).
    BodyFixed,
    /// Does not rotate with a body (e.g. `J2000` / `ICRF`).
    Inertial,
    /// A local surface/site frame.
    Topocentric,
}

impl FrameClass {
    /// Parse a wire spelling; `None` for anything outside the closed vocabulary.
    pub fn from_wire(value: &str) -> Option<Self> {
        match value {
            "body_fixed" => Some(Self::BodyFixed),
            "inertial" => Some(Self::Inertial),
            "topocentric" => Some(Self::Topocentric),
            _ => None,
        }
    }

    /// The wire spelling.
    pub const fn as_wire(self) -> &'static str {
        match self {
            Self::BodyFixed => "body_fixed",
            Self::Inertial => "inertial",
            Self::Topocentric => "topocentric",
        }
    }
}

/// A named reference frame — `name` is a SPICE frame name, `center` the SPICE body it is centered
/// on (or `None` for a centre-agnostic sky frame).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceFrame {
    /// SPICE frame name (e.g. `MOON_ME`, `J2000`).
    pub name: String,
    /// The frame's class.
    pub frame_class: FrameClass,
    /// SPICE body the frame is centered on, if any.
    pub center: Option<String>,
}

impl ReferenceFrame {
    /// The typed path (the Pydantic-model analogue): validates the tokens of rule 1.
    pub fn new(
        name: impl Into<String>,
        frame_class: FrameClass,
        center: Option<String>,
    ) -> UnitsResult<Self> {
        let name = check_token(&name.into(), "ReferenceFrame.name")?;
        let center = center
            .map(|c| check_token(&c, "ReferenceFrame.center"))
            .transpose()?;
        Ok(Self {
            name,
            frame_class,
            center,
        })
    }
}

/// An instant in TDB/ET — SI seconds past the J2000 TDB epoch (SPICE ephemeris time directly).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Epoch {
    /// Seconds past J2000 TDB.
    pub tdb_seconds: f64,
    /// The (required) time scale — there is no implicit one.
    pub scale: TimeScale,
}

impl Epoch {
    /// The typed path. `scale` is a [`TimeScale`], so rule 2 holds by construction.
    pub const fn new(tdb_seconds: f64, scale: TimeScale) -> Self {
        Self { tdb_seconds, scale }
    }
}

/// A half-open epoch interval `[start, end)` over which a product is defined.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EpochWindow {
    /// Window start.
    pub start: Epoch,
    /// Window end — strictly after `start` (rule 5).
    pub end: Epoch,
}

impl EpochWindow {
    /// The typed path: enforces the strict ordering of rule 5. Written as `<=`, exactly as the
    /// reference's `_check_order` is, so the two agree edge-for-edge.
    pub fn new(start: Epoch, end: Epoch) -> UnitsResult<Self> {
        if end.tdb_seconds <= start.tdb_seconds {
            return Err(UnitsValidationError::new(format!(
                "EpochWindow end must be strictly after start (start={}, end={})",
                start.tdb_seconds, end.tdb_seconds
            )));
        }
        Ok(Self { start, end })
    }
}

/// An explicit planetary coordinate reference system — the minimum needed to reproject spatial
/// data without guessing. No field defaults to an Earth/WGS84 value (conventions.md §5).
#[derive(Debug, Clone, PartialEq)]
pub struct PlanetaryCRS {
    /// NAIF body name (e.g. `MOON`).
    pub body: String,
    /// The body's body-fixed frame (e.g. `MOON_ME`).
    pub body_fixed_frame: String,
    /// PROJ planetary reference radius (`+R`), finite and `> 0` (rule 4).
    pub reference_radius_m: f64,
    /// Explicit PROJ/WKT/EPSG string for a *projected* CRS; `None` means body-fixed geographic.
    pub projection: Option<String>,
    /// Explicit datum, if any.
    pub datum: Option<String>,
}

impl PlanetaryCRS {
    /// The typed path: enforces rules 4 and 6. This is where the finite-radius rule is reachable —
    /// JSON has no infinity literal, so the conformance vectors cannot express it.
    pub fn new(
        body: impl Into<String>,
        body_fixed_frame: impl Into<String>,
        reference_radius_m: f64,
        projection: Option<String>,
        datum: Option<String>,
    ) -> UnitsResult<Self> {
        let body = check_token(&body.into(), "PlanetaryCRS.body")?;
        let body_fixed_frame =
            check_token(&body_fixed_frame.into(), "PlanetaryCRS.body_fixed_frame")?;
        // Rule 4: finite (rejects ±inf and NaN) and `> 0` (rejects 0 and negatives).
        if !reference_radius_m.is_finite() || reference_radius_m <= 0.0 {
            return Err(UnitsValidationError::new(format!(
                "reference_radius_m must be finite and > 0, got {reference_radius_m}"
            )));
        }
        let crs = Self {
            body,
            body_fixed_frame,
            reference_radius_m,
            projection,
            datum,
        };
        // Rule 6: an Earth datum/projection marker is only valid on body EARTH.
        if !crs.body.eq_ignore_ascii_case(EARTH) && crs.has_earth_marker() {
            return Err(UnitsValidationError::new(format!(
                "Earth CRS marker (WGS84 / EPSG:4326 / urn:ogc:def:crs:OGC) on non-Earth body {:?}: \
                 an implicit Earth/WGS84 CRS is a defaulting bug (conventions.md §5). \
                 Set body=EARTH for a legitimate Earth-analog CRS.",
                crs.body
            )));
        }
        Ok(crs)
    }

    /// Whether the CRS's projection/datum carries an Earth datum/projection marker.
    fn has_earth_marker(&self) -> bool {
        [self.projection.as_deref(), self.datum.as_deref()]
            .into_iter()
            .flatten()
            .any(|field| {
                let lowered = field.to_ascii_lowercase();
                EARTH_CRS_MARKERS.iter().any(|m| lowered.contains(m))
            })
    }
}

/// Whether two time scales denote the same physical scale (rule 3).
///
/// [`TimeScale::Et`] and [`TimeScale::Tdb`] are the **same** scale (SPICE ET ≡ TDB), so this
/// returns true for any `(et, tdb)` pairing. Consumers MUST use this instead of `a == b` when
/// gating on TDB, so an epoch spelled `et` is not spuriously rejected against a `tdb` one — a
/// naive `scale == Tdb` is a latent bug in every binding. The equality arm is not redundant: the
/// scale vocabulary is append-only by RFC, and the reference is written the same way.
pub fn scales_equivalent(a: TimeScale, b: TimeScale) -> bool {
    a == b
        || matches!(
            (a, b),
            (TimeScale::Et, TimeScale::Tdb) | (TimeScale::Tdb, TimeScale::Et)
        )
}

/// Whether `unit` is a recognized SI-consistent unit token (an SI symbol or a dimensionless
/// ratio marker).
pub fn is_si_unit(unit: &str) -> bool {
    SI_UNITS.contains(&unit) || DIMENSIONLESS_UNITS.contains(&unit)
}

/// Return `unit` if it is SI-consistent, else fail loudly.
///
/// Core validates the simple unit tokens used in Core-schema fields; it is not a dimensional
/// analysis engine — composite units (e.g. `kg/m3`) in component data products are SI-consistent
/// strings validated by those components, not enumerated here.
pub fn require_si_unit(unit: &str) -> UnitsResult<&str> {
    if !is_si_unit(unit) {
        return Err(UnitsValidationError::new(format!(
            "unit {unit:?} is not a recognized SI-consistent unit token"
        )));
    }
    Ok(unit)
}

/// Coerce/validate an explicit reference frame from JSON, failing loudly on a missing one.
///
/// `null`/missing/invalid is rejected — there is no implicit Earth/WGS84 frame (rules 1-2).
pub fn require_frame(value: &Value) -> UnitsResult<ReferenceFrame> {
    let obj = require_object(
        value,
        "a reference frame is required; none was given (no implicit Earth/WGS84 frame)",
        "reference frame",
    )?;
    deny_unknown(obj, &["name", "frame_class", "center"], "ReferenceFrame")?;
    let name = required_str(obj, "name", "ReferenceFrame")?;
    let frame_class = required_str(obj, "frame_class", "ReferenceFrame")?;
    let frame_class = FrameClass::from_wire(frame_class).ok_or_else(|| {
        UnitsValidationError::new(format!(
            "ReferenceFrame.frame_class {frame_class:?} is not a FrameClass member \
             (known: {})",
            FRAME_CLASSES.join(", ")
        ))
    })?;
    let center = optional_str(obj, "center", "ReferenceFrame")?.map(str::to_owned);
    ReferenceFrame::new(name, frame_class, center)
}

/// Coerce/validate an explicit epoch from JSON, failing loudly on a missing one (rules 2-3).
///
/// An `et`-scaled epoch is accepted everywhere a `tdb`-scaled one is (SPICE ET ≡ TDB); compare
/// scales with [`scales_equivalent`], never `==`.
pub fn require_epoch(value: &Value) -> UnitsResult<Epoch> {
    let obj = require_object(
        value,
        "an epoch is required; none was given (no implicit time scale)",
        "epoch",
    )?;
    deny_unknown(obj, &["tdb_seconds", "scale"], "Epoch")?;
    let tdb_seconds = required_number(obj, "tdb_seconds", "Epoch")?;
    let scale = required_str(obj, "scale", "Epoch")?;
    let scale = TimeScale::from_wire(scale).ok_or_else(|| {
        UnitsValidationError::new(format!(
            "Epoch.scale {scale:?} is not a TimeScale member (known: {})",
            TIME_SCALES.join(", ")
        ))
    })?;
    Ok(Epoch::new(tdb_seconds, scale))
}

/// Coerce/validate an epoch window from JSON, failing loudly on a missing/mis-ordered one (rule 5).
pub fn require_epoch_window(value: &Value) -> UnitsResult<EpochWindow> {
    let obj = require_object(
        value,
        "an epoch window is required; none was given",
        "epoch window",
    )?;
    deny_unknown(obj, &["start", "end"], "EpochWindow")?;
    let start = require_epoch(obj.get("start").unwrap_or(&Value::Null))?;
    let end = require_epoch(obj.get("end").unwrap_or(&Value::Null))?;
    EpochWindow::new(start, end)
}

/// Coerce/validate an explicit planetary CRS from JSON, failing loudly on a missing/implicit one.
///
/// Rejects `null`/missing/invalid so spatial data without an explicit CRS is refused at ingest
/// (rule 4), and applies the body/datum consistency rule (rule 6).
pub fn require_crs(value: &Value) -> UnitsResult<PlanetaryCRS> {
    let obj = require_object(
        value,
        "spatial data requires an explicit planetary CRS; none was given (no implicit Earth/WGS84)",
        "planetary CRS",
    )?;
    deny_unknown(
        obj,
        &[
            "body",
            "body_fixed_frame",
            "reference_radius_m",
            "projection",
            "datum",
        ],
        "PlanetaryCRS",
    )?;
    let body = required_str(obj, "body", "PlanetaryCRS")?;
    let body_fixed_frame = required_str(obj, "body_fixed_frame", "PlanetaryCRS")?;
    let reference_radius_m = required_number(obj, "reference_radius_m", "PlanetaryCRS")?;
    let projection = optional_str(obj, "projection", "PlanetaryCRS")?.map(str::to_owned);
    let datum = optional_str(obj, "datum", "PlanetaryCRS")?.map(str::to_owned);
    PlanetaryCRS::new(
        body,
        body_fixed_frame,
        reference_radius_m,
        projection,
        datum,
    )
}

// --- shared field helpers (the Pydantic validators' Rust analogues) ------------------

/// Require a non-empty, whitespace-free identifier (a SPICE frame/body name; rule 1). A blank,
/// padded, or whitespace-bearing name is almost always a defaulting bug.
fn check_token(value: &str, field: &str) -> UnitsResult<String> {
    if value.is_empty() || value.chars().any(char::is_whitespace) {
        return Err(UnitsValidationError::new(format!(
            "{field} must be a non-empty, whitespace-free token, got {value:?}"
        )));
    }
    Ok(value.to_owned())
}

fn require_object<'a>(
    value: &'a Value,
    missing: &str,
    what: &str,
) -> UnitsResult<&'a Map<String, Value>> {
    match value {
        Value::Null => Err(UnitsValidationError::new(missing)),
        Value::Object(obj) => Ok(obj),
        other => Err(UnitsValidationError::new(format!(
            "cannot interpret {} as a {what}",
            json_kind(other)
        ))),
    }
}

/// Reject unknown/typo'd members — the analogue of the models' `extra="forbid"`.
fn deny_unknown(obj: &Map<String, Value>, allowed: &[&str], what: &str) -> UnitsResult<()> {
    for key in obj.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(UnitsValidationError::new(format!(
                "invalid {what}: unknown field {key:?}"
            )));
        }
    }
    Ok(())
}

fn required_str<'a>(obj: &'a Map<String, Value>, key: &str, what: &str) -> UnitsResult<&'a str> {
    match obj.get(key) {
        Some(Value::String(s)) => Ok(s),
        Some(other) => Err(UnitsValidationError::new(format!(
            "invalid {what}: {key} must be a string, got {}",
            json_kind(other)
        ))),
        None => Err(UnitsValidationError::new(format!(
            "invalid {what}: {key} is required"
        ))),
    }
}

/// An optional string member: absent or explicit `null` are both `None` (mirrors `str | None`).
fn optional_str<'a>(
    obj: &'a Map<String, Value>,
    key: &str,
    what: &str,
) -> UnitsResult<Option<&'a str>> {
    match obj.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(other) => Err(UnitsValidationError::new(format!(
            "invalid {what}: {key} must be a string, got {}",
            json_kind(other)
        ))),
    }
}

fn required_number(obj: &Map<String, Value>, key: &str, what: &str) -> UnitsResult<f64> {
    match obj.get(key) {
        Some(Value::Number(n)) => n.as_f64().ok_or_else(|| {
            UnitsValidationError::new(format!("invalid {what}: {key} is not representable as f64"))
        }),
        Some(other) => Err(UnitsValidationError::new(format!(
            "invalid {what}: {key} must be a number, got {}",
            json_kind(other)
        ))),
        None => Err(UnitsValidationError::new(format!(
            "invalid {what}: {key} is required"
        ))),
    }
}

fn json_kind(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// The shared, language-neutral vectors (RFC-0007 Design §3; RM-P1-CORE-08) — the *same* file
    /// the Python reference runs in `tests/test_units_conformance.py`. Running them here is how the
    /// Rust binding discharges the "every Core binding … passes the shared units conformance
    /// vectors" exit criterion of RM-P1-CORE-08.
    const CONFORMANCE: &str =
        include_str!("../../../src/astro_mine/core/units/schema/conformance.json");

    /// Dispatch a vector to its guard, exactly as the Python `_GUARDS` table does. An unknown kind
    /// panics: a guard family added to the vectors without a Rust guard is a failure, not a skip.
    fn verdict(kind: &str, value: &Value) -> bool {
        match kind {
            "reference_frame" => require_frame(value).is_ok(),
            "epoch" => require_epoch(value).is_ok(),
            "epoch_window" => require_epoch_window(value).is_ok(),
            "planetary_crs" => require_crs(value).is_ok(),
            // The reference's require_si_unit takes a token; a non-string value is invalid, as it
            // is in Python (`unit in KNOWN_UNITS` is False for a non-token).
            "si_unit" => value
                .as_str()
                .is_some_and(|unit| require_si_unit(unit).is_ok()),
            other => panic!("conformance kind {other:?} has no Rust guard — the binding is behind"),
        }
    }

    #[test]
    fn shared_conformance_vectors_pass() {
        let doc: Value = serde_json::from_str(CONFORMANCE).expect("conformance vectors are JSON");
        let families = doc.as_object().expect("conformance vectors are an object");
        let mut checked = 0usize;
        let mut kinds = 0usize;
        for (kind, cases) in families {
            if kind.starts_with('$') {
                continue; // $comment
            }
            kinds += 1;
            for case in cases.as_array().expect("a family is an array of cases") {
                let name = case["name"].as_str().expect("case has a name");
                let expected = case["valid"].as_bool().expect("case has a verdict");
                assert_eq!(
                    verdict(kind, &case["value"]),
                    expected,
                    "conformance vector {kind}/{name}: expected valid={expected}"
                );
                checked += 1;
            }
        }
        // Guard against a silently-empty run (a renamed file or an empty object would pass above).
        assert_eq!(kinds, 5, "every guard family must have vectors");
        assert!(checked >= 40, "expected the full vector set, got {checked}");
    }

    // --- rules JSON cannot express (mirrors tests/test_units_conformance.py) --------------

    #[test]
    fn reference_radius_must_be_finite() {
        // Rule 4: +inf satisfies `> 0` but is rejected. JSON has no infinity literal, so this is
        // reachable only on the typed path — the same split the Python reference test makes.
        assert!(PlanetaryCRS::new("MOON", "MOON_ME", f64::INFINITY, None, None).is_err());
        assert!(PlanetaryCRS::new("MOON", "MOON_ME", f64::NAN, None, None).is_err());
        assert!(PlanetaryCRS::new("MOON", "MOON_ME", 1_737_400.0, None, None).is_ok());
    }

    #[test]
    fn earth_crs_body_datum_consistency() {
        // Rule 6: reject an Earth marker off EARTH, accept it on EARTH.
        assert!(PlanetaryCRS::new(
            "MOON",
            "MOON_ME",
            1_737_400.0,
            Some("+proj=longlat +datum=WGS84".to_string()),
            None,
        )
        .is_err());
        let crs = PlanetaryCRS::new(
            "EARTH",
            "ITRF93",
            6_378_137.0,
            Some("+proj=longlat +datum=WGS84".to_string()),
            None,
        )
        .expect("an Earth CRS on body EARTH is valid at the waist");
        assert_eq!(crs.body, "EARTH");
    }

    #[test]
    fn et_scale_is_equivalent_to_tdb() {
        // Rule 3: an ET epoch passes every consumer path that accepts TDB.
        let et = require_epoch(&json!({"tdb_seconds": 42.0, "scale": "et"})).expect("valid epoch");
        assert_eq!(et.scale, TimeScale::Et);
        assert!(scales_equivalent(TimeScale::Et, TimeScale::Tdb));
        assert!(scales_equivalent(TimeScale::Tdb, TimeScale::Et));

        // A naive `scale == Tdb` gate would reject an ET epoch; scales_equivalent must not.
        fn accepts_tdb(e: Epoch) -> bool {
            scales_equivalent(e.scale, TimeScale::Tdb)
        }
        assert!(accepts_tdb(Epoch::new(0.0, TimeScale::Et)));
        assert!(accepts_tdb(Epoch::new(0.0, TimeScale::Tdb)));
    }

    #[test]
    fn unknown_members_are_rejected() {
        // The models are `extra="forbid"`; a typo'd member is not silently ignored.
        assert!(
            require_frame(&json!({"name": "J2000", "frame_class": "inertial", "oops": 1})).is_err()
        );
        assert!(require_epoch(&json!({"tdb_seconds": 0.0, "scale": "tdb", "oops": 1})).is_err());
        assert!(require_crs(
            &json!({"body": "MOON", "body_fixed_frame": "MOON_ME", "reference_radius_m": 1.0, "oops": 1})
        )
        .is_err());
    }

    #[test]
    fn non_object_inputs_are_rejected() {
        for value in [json!(12345), json!("MOON_ME"), json!([])] {
            assert!(require_frame(&value).is_err());
            assert!(require_epoch(&value).is_err());
            assert!(require_epoch_window(&value).is_err());
            assert!(require_crs(&value).is_err());
        }
    }

    #[test]
    fn field_types_are_enforced() {
        assert!(require_epoch(&json!({"tdb_seconds": "0.0", "scale": "tdb"})).is_err());
        assert!(require_epoch(&json!({"tdb_seconds": true, "scale": "tdb"})).is_err());
        assert!(require_frame(&json!({"name": 1, "frame_class": "inertial"})).is_err());
        assert!(
            require_frame(&json!({"name": "J2000", "frame_class": "inertial", "center": 1}))
                .is_err()
        );
        assert!(
            require_epoch_window(&json!({"start": {"tdb_seconds": 0.0, "scale": "tdb"}})).is_err()
        );
    }

    #[test]
    fn vocabularies_round_trip() {
        // The wire-spelling lists and the enums are one vocabulary (the Python-vs-Rust drift guard
        // in tests/test_validator_rust_parity.py reads these lists).
        for wire in TIME_SCALES {
            assert_eq!(
                TimeScale::from_wire(wire).expect("known scale").as_wire(),
                wire
            );
        }
        for wire in FRAME_CLASSES {
            assert_eq!(
                FrameClass::from_wire(wire).expect("known class").as_wire(),
                wire
            );
        }
        assert!(TimeScale::from_wire("utc").is_none());
        assert!(FrameClass::from_wire("galactic").is_none());
        for unit in SI_UNITS.iter().chain(DIMENSIONLESS_UNITS.iter()) {
            assert!(is_si_unit(unit), "{unit} must be a known unit");
        }
        assert!(!is_si_unit("furlong"));
    }

    #[test]
    fn errors_render_their_message() {
        let err = require_si_unit("furlong").expect_err("furlong is not SI");
        assert!(err.message().contains("furlong"));
        assert_eq!(err.to_string(), err.message());
    }
}
