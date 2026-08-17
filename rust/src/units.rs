// SPDX-License-Identifier: Apache-2.0
//! Units / frames / time guards — the non-Python binding of Core's `require_*` contract.
//!
//! RFC-0007 (Design §3) ratifies the frame/CRS/time guard rules as a normative contract Core
//! ships as a **shared conformance vector file** (`units/schema/conformance.json`,
//! RM-P1-CORE-08). Core's Python `validate.py` is the reference implementation; every non-Python
//! binding implements the same rules in its own language and discharges the obligation by running
//! those vectors in its own CI. Guard's Rust safety core is one of the two designated non-Python
//! targets (View's TypeScript is the other), so it implements the guards here and pins them with
//! Core's vectors (`rust/tests/conformance.rs`).
//!
//! This is deliberately ~pure token/enum validation over borrowed primitives: **it links nothing
//! new** (no serde, no Core crate) so the trusted computing base's dependency surface does not
//! grow. The safety core calls [`require_frame`] on a decoded keep-out / safe-pose frame
//! (`model.rs`), fail-closed; typing the frame does not widen what the core reads for control.
//!
//! The six normative rules (RFC-0007 Design §3; conventions.md §5), in brief:
//! 1. a frame is present; `name`/`center` are non-empty, whitespace-free tokens;
//! 2. `frame_class` ∈ `FrameClass`; `scale` ∈ `TimeScale`;
//! 3. `TimeScale.ET` and `TimeScale.TDB` denote the **same** scale (SPICE ET ≡ TDB) — compare
//!    with [`scales_equivalent`], never `==`;
//! 4. a CRS is present; `body`/`body_fixed_frame` are tokens; `reference_radius_m` is finite `> 0`;
//! 5. `EpochWindow.end` is strictly after `.start`;
//! 6. an Earth datum/projection marker is rejected off `EARTH` and accepted on it.

/// A failed units guard. `String` message only — this never crosses the wire; it exists so a
/// rejection is loud in logs/tests, mirroring Core's `UnitsValidationError`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnitsError(pub String);

impl core::fmt::Display for UnitsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "units validation error: {}", self.0)
    }
}

impl std::error::Error for UnitsError {}

/// Members of `FrameClass` (units/enums.py) — closed, append-only by RFC.
pub const FRAME_CLASSES: [&str; 3] = ["body_fixed", "inertial", "topocentric"];

/// Members of `TimeScale` — SPICE ephemeris scales only (`et` ≡ `tdb`).
pub const TIME_SCALES: [&str; 2] = ["tdb", "et"];

/// The unit tokens Core recognizes at the waist: SI symbols + dimensionless markers
/// (`SI_UNITS | DIMENSIONLESS_UNITS`, units/enums.py). Composite units are validated by
/// components, not enumerated here.
pub const KNOWN_UNITS: [&str; 19] = [
    "m",
    "kg",
    "s",
    "A",
    "K",
    "mol",
    "cd",
    "rad",
    "sr",
    "Hz",
    "N",
    "Pa",
    "J",
    "W",
    "V",
    "C",
    "dimensionless",
    "mass_fraction",
    "volume_fraction",
];

/// Case-insensitive Earth datum/projection markers (rule 6). Not exhaustive — the common
/// defaulting-bug markers a non-Earth product must never carry.
const EARTH_CRS_MARKERS: [&str; 3] = ["wgs84", "epsg:4326", "urn:ogc:def:crs:ogc"];

/// A SPICE-ready identifier: non-empty and whitespace-free (matches `units.model._validate_token`).
fn is_token(s: &str) -> bool {
    !s.is_empty() && !s.chars().any(char::is_whitespace)
}

/// Whether two time scales denote the same physical scale (rule 3): `et` ≡ `tdb`. Consumers MUST
/// use this, never `==`, so an `et`-spelled epoch is not spuriously rejected against a `tdb` one.
pub fn scales_equivalent(a: &str, b: &str) -> bool {
    a == b || (TIME_SCALES.contains(&a) && TIME_SCALES.contains(&b))
}

/// Whether `unit` is a recognized SI-consistent unit token (rule for `require_si_unit`).
pub fn is_si_unit(unit: &str) -> bool {
    !unit.is_empty() && KNOWN_UNITS.contains(&unit)
}

/// Rule 1-2: a reference frame's `name` (and `center`, when given) are non-empty whitespace-free
/// tokens and `frame_class` is a `FrameClass` member. A `frame_class` of `""` (proto3's default /
/// a missing field) is not a member, so an unset class fails loudly — no implicit frame.
pub fn require_frame(
    name: &str,
    frame_class: &str,
    center: Option<&str>,
) -> Result<(), UnitsError> {
    if !is_token(name) {
        return Err(UnitsError(format!(
            "frame name must be a non-empty, whitespace-free token, got {name:?}"
        )));
    }
    if !FRAME_CLASSES.contains(&frame_class) {
        return Err(UnitsError(format!(
            "frame_class {frame_class:?} is not a FrameClass member ({FRAME_CLASSES:?})"
        )));
    }
    if let Some(c) = center {
        if !is_token(c) {
            return Err(UnitsError(format!(
                "frame center must be a non-empty, whitespace-free token, got {c:?}"
            )));
        }
    }
    Ok(())
}

/// Rule 2-3: an epoch's `scale` is a `TimeScale` member. `et` is accepted everywhere `tdb` is.
pub fn require_epoch(scale: &str) -> Result<(), UnitsError> {
    if TIME_SCALES.contains(&scale) {
        Ok(())
    } else {
        Err(UnitsError(format!(
            "epoch scale {scale:?} is not a TimeScale member ({TIME_SCALES:?}) — no implicit scale"
        )))
    }
}

/// Rule 5 (+ 2): both endpoints carry valid scales and `end` is strictly after `start`.
pub fn require_epoch_window(
    start_scale: &str,
    start_tdb: f64,
    end_scale: &str,
    end_tdb: f64,
) -> Result<(), UnitsError> {
    require_epoch(start_scale)?;
    require_epoch(end_scale)?;
    if end_tdb > start_tdb {
        Ok(())
    } else {
        Err(UnitsError(format!(
            "EpochWindow end ({end_tdb}) must be strictly after start ({start_tdb})"
        )))
    }
}

/// Rule 4 (+ 6): a planetary CRS's `body`/`body_fixed_frame` are tokens, `reference_radius_m` is
/// finite `> 0`, and an Earth datum/projection marker is only valid on body `EARTH`.
pub fn require_crs(
    body: &str,
    body_fixed_frame: &str,
    reference_radius_m: f64,
    projection: Option<&str>,
    datum: Option<&str>,
) -> Result<(), UnitsError> {
    if !is_token(body) {
        return Err(UnitsError(format!(
            "CRS body must be a token, got {body:?}"
        )));
    }
    if !is_token(body_fixed_frame) {
        return Err(UnitsError(format!(
            "CRS body_fixed_frame must be a token, got {body_fixed_frame:?}"
        )));
    }
    if !(reference_radius_m.is_finite() && reference_radius_m > 0.0) {
        return Err(UnitsError(format!(
            "reference_radius_m must be finite and > 0, got {reference_radius_m}"
        )));
    }
    let has_earth_marker = [projection, datum].into_iter().flatten().any(|s| {
        let lower = s.to_ascii_lowercase();
        EARTH_CRS_MARKERS.iter().any(|m| lower.contains(m))
    });
    if !body.eq_ignore_ascii_case("EARTH") && has_earth_marker {
        return Err(UnitsError(format!(
            "Earth CRS marker on non-Earth body {body:?} — an implicit Earth/WGS84 CRS is a \
             defaulting bug (conventions.md §5)"
        )));
    }
    Ok(())
}

/// Rule for `require_si_unit`: reject a unit token outside the SI-consistent vocabulary.
pub fn require_si_unit(unit: &str) -> Result<(), UnitsError> {
    if is_si_unit(unit) {
        Ok(())
    } else {
        Err(UnitsError(format!(
            "unit {unit:?} is not a recognized SI-consistent unit token"
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_token_and_class_rules() {
        assert!(require_frame("MOON_ME", "body_fixed", Some("MOON")).is_ok());
        assert!(require_frame("J2000", "inertial", None).is_ok());
        assert!(require_frame("", "inertial", None).is_err()); // empty name
        assert!(require_frame("MOON ME", "body_fixed", None).is_err()); // whitespace
        assert!(require_frame("J2000", "galactic", None).is_err()); // bad class
        assert!(require_frame("MOON_ME", "body_fixed", Some("")).is_err()); // empty center
        assert!(require_frame("J2000", "", None).is_err()); // missing class (proto3 default)
    }

    #[test]
    fn et_and_tdb_are_the_same_scale() {
        assert!(scales_equivalent("et", "tdb"));
        assert!(scales_equivalent("tdb", "et"));
        assert!(require_epoch("et").is_ok());
        assert!(require_epoch("utc").is_err());
    }
}
