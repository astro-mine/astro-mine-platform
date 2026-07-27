"""Load + validate + compile the anchor SafetySpec end-to-end (RM-P1-GUARD-01 acceptance).

Acceptance: "A SafetySpec for the anchor scenario (power floor, thermal ceiling, slope/keep-out,
a night-survival temporal clause) validates and compiles to monitor + shield artifacts."
"""

from __future__ import annotations

import pytest

from astro_mine.guard.spec import (
    CompiledSafetyModel,
    ConstraintKind,
    GeometryKind,
    OnUncertain,
    SafetyDocument,
    SafetySpecValidationError,
    TemporalOp,
    load_safety_spec,
    validate_safety_spec,
)


def test_anchor_validates(anchor_document: SafetyDocument) -> None:
    spec = anchor_document.safety
    assert anchor_document.safety_version == "0.1"
    assert spec.id == "anchor-lunar-polar-v0"
    kinds = {c.kind for c in spec.constraints}
    # every constraint family the acceptance names is present
    assert {
        ConstraintKind.POWER_FLOOR,
        ConstraintKind.ENERGY_FLOOR,
        ConstraintKind.THERMAL_CEILING,
        ConstraintKind.THERMAL_FLOOR,
        ConstraintKind.TORQUE_CEILING,
        ConstraintKind.KINEMATIC_LIMIT,
        ConstraintKind.KEEP_OUT,
        ConstraintKind.TEMPORAL,
    } <= kinds


def test_anchor_keep_out_geometry(anchor_document: SafetyDocument) -> None:
    shapes = {
        c.keep_out.volume.shape
        for c in anchor_document.safety.constraints
        if c.kind == ConstraintKind.KEEP_OUT and c.keep_out is not None
    }
    assert shapes == {GeometryKind.BOX, GeometryKind.SPHERE, GeometryKind.HALF_SPACE}


def test_anchor_fail_safe_defaults(anchor_document: SafetyDocument) -> None:
    # Every constraint resolves to a safe action — never passthrough (it does not exist).
    resolutions = {c.on_uncertain for c in anchor_document.safety.constraints}
    assert resolutions <= {OnUncertain.FALLBACK, OnUncertain.HOLD, OnUncertain.SAFE_STATE}
    assert "passthrough" not in {str(r) for r in resolutions}
    # Constraints without an explicit on_uncertain default to fallback.
    by_id = {c.id: c for c in anchor_document.safety.constraints}
    assert by_id["c_power_floor"].on_uncertain == OnUncertain.FALLBACK


def test_anchor_compiles_to_monitors_and_terms(anchor_compiled: CompiledSafetyModel) -> None:
    m = anchor_compiled
    assert m.compiled_version == "0.1"
    assert m.spec_id == "anchor-lunar-polar-v0"
    assert m.spec_content_hash.startswith("sha256:")

    # Barrier / keep-out terms — one per keep-out constraint, all three geometries.
    assert {t.shape for t in m.keep_out_terms} == {
        GeometryKind.BOX,
        GeometryKind.SPHERE,
        GeometryKind.HALF_SPACE,
    }

    # Monitor automata — one per temporal clause, with bounded history windows.
    assert len(m.monitors) == 2
    ops = {mon.root.op for mon in m.monitors}
    assert ops == {TemporalOp.UNTIL, TemporalOp.ALWAYS}
    for mon in m.monitors:
        assert mon.history_window_len > 0  # finite, bounded
        assert mon.node_count >= 2
        assert mon.predicate_indices  # references real predicate slots


def test_anchor_every_constraint_appears_in_ir(
    anchor_document: SafetyDocument, anchor_compiled: CompiledSafetyModel
) -> None:
    m = anchor_compiled
    covered = (
        {b.constraint_id for b in m.scalar_bounds}
        | {t.constraint_id for t in m.keep_out_terms}
        | {mon.constraint_id for mon in m.monitors}
    )
    # every constraint id declared in the spec is represented in the compiled IR
    assert covered == {c.id for c in anchor_document.safety.constraints}


def test_reject_unbounded_temporal_operator() -> None:
    # A temporal operator with no interval_s has no statically-bounded monitor — fail-safe.
    src = """
safety_version: "0.1"
safety:
  id: bad
  name: bad
  signals: [{key: soc, unit: J, source: observation}]
  constraints:
    - kind: temporal
      id: t
      temporal:
        formula:
          op: always
          args:
            - {op: predicate, signal: soc, cmp: ge, threshold: 1.0}
"""
    with pytest.raises(SafetySpecValidationError, match="requires a bounded interval_s"):
        load_safety_spec(src)


def test_reject_undeclared_signal() -> None:
    src = """
safety_version: "0.1"
safety:
  id: bad
  name: bad
  signals: [{key: soc, unit: J, source: observation}]
  constraints:
    - {kind: power_floor, id: p, power_floor: {signal: not_declared, floor_w: 1.0}}
"""
    with pytest.raises(SafetySpecValidationError, match="not declared in signals"):
        load_safety_spec(src)


def test_reject_wrong_union_payload() -> None:
    # kind says power_floor but the energy_floor payload is set instead.
    src = """
safety_version: "0.1"
safety:
  id: bad
  name: bad
  signals: [{key: soc, unit: J, source: observation}]
  constraints:
    - {kind: power_floor, id: p, energy_floor: {signal: soc, floor_j: 1.0}}
"""
    with pytest.raises(SafetySpecValidationError):
        load_safety_spec(src)


def test_reject_duplicate_constraint_ids() -> None:
    src = """
safety_version: "0.1"
safety:
  id: bad
  name: bad
  signals: [{key: soc, unit: J, source: observation}]
  constraints:
    - {kind: energy_floor, id: dup, energy_floor: {signal: soc, floor_j: 1.0}}
    - {kind: energy_floor, id: dup, energy_floor: {signal: soc, floor_j: 2.0}}
"""
    with pytest.raises(SafetySpecValidationError, match="duplicate constraint id"):
        load_safety_spec(src)


def test_reject_kinematic_without_bound() -> None:
    src = """
safety_version: "0.1"
safety:
  id: bad
  name: bad
  signals: [{key: v, unit: "m/s", source: observation}]
  constraints:
    - {kind: kinematic_limit, id: k, kinematic_limit: {signal: v}}
"""
    with pytest.raises(SafetySpecValidationError, match="at least one"):
        load_safety_spec(src)


def test_reject_empty_unit() -> None:
    src = """
safety_version: "0.1"
safety:
  id: bad
  name: bad
  signals: [{key: soc, unit: "  ", source: observation}]
  constraints:
    - {kind: energy_floor, id: e, energy_floor: {signal: soc, floor_j: 1.0}}
"""
    with pytest.raises(SafetySpecValidationError, match="unit must be"):
        load_safety_spec(src)


def test_reject_zero_normal_half_space() -> None:
    src = """
safety_version: "0.1"
safety:
  id: bad
  name: bad
  signals: [{key: soc, unit: J, source: observation}]
  constraints:
    - kind: keep_out
      id: ko
      keep_out:
        margin_m: 1.0
        volume:
          shape: half_space
          half_space: {frame: F, normal: {x: 0, y: 0, z: 0}, offset_m: 1.0}
"""
    with pytest.raises(SafetySpecValidationError, match="non-zero"):
        load_safety_spec(src)


def test_validate_safety_spec_accepts_document(anchor_document: SafetyDocument) -> None:
    # validate on a typed document, a mapping, and text all agree (no exception).
    validate_safety_spec(anchor_document)
    validate_safety_spec(anchor_document.model_dump(mode="json"))
    validate_safety_spec(anchor_document.model_dump_json())


def test_validate_rejects_bad_type() -> None:
    with pytest.raises(SafetySpecValidationError, match="cannot validate"):
        validate_safety_spec(42)  # type: ignore[arg-type]


def test_safe_pose_frame_mismatch_is_rejected() -> None:
    # The retreat target and the keep-out safe set must share a CRS (LUNAR-TR-001).
    src = """
safety_version: "0.1"
safety:
  id: t
  name: t
  safe_pose: {frame: OTHER_FRAME, position_m: {x: 1.0, y: 2.0, z: 3.0}}
  signals: [{key: soc, unit: J, source: observation}]
  constraints:
    - {kind: energy_floor, id: e, on_uncertain: safe_state,
       energy_floor: {signal: soc, floor_j: 1.0}}
    - kind: keep_out
      id: k
      keep_out:
        margin_m: 3.0
        volume:
          shape: sphere
          sphere: {frame: MOON_ME, center_m: {x: 0.0, y: 0.0, z: 0.0}, radius_m: 30.0}
"""
    with pytest.raises(SafetySpecValidationError, match="does not match the keep-out"):
        load_safety_spec(src)


def test_safe_pose_matching_frame_is_accepted() -> None:
    src = """
safety_version: "0.1"
safety:
  id: t
  name: t
  safe_pose: {frame: MOON_ME, position_m: {x: 40.0, y: 0.0, z: 0.0}}
  signals: [{key: soc, unit: J, source: observation}]
  constraints:
    - {kind: energy_floor, id: e, on_uncertain: safe_state,
       energy_floor: {signal: soc, floor_j: 1.0}}
    - kind: keep_out
      id: k
      keep_out:
        margin_m: 3.0
        volume:
          shape: sphere
          sphere: {frame: MOON_ME, center_m: {x: 0.0, y: 0.0, z: 0.0}, radius_m: 30.0}
"""
    doc = load_safety_spec(src)
    assert doc.safety.safe_pose is not None
    assert doc.safety.safe_pose.frame == "MOON_ME"


def test_safe_pose_without_keepouts_needs_only_a_frame() -> None:
    # No keep-out geometry ⇒ nothing to match against, but the frame must still be explicit.
    doc = load_safety_spec(
        """
safety_version: "0.1"
safety:
  id: t
  name: t
  safe_pose: {frame: MOON_ME, position_m: {x: 1.0, y: 0.0, z: 0.0}}
  signals: [{key: soc, unit: J, source: observation}]
  constraints:
    - {kind: energy_floor, id: e, on_uncertain: safe_state,
       energy_floor: {signal: soc, floor_j: 1.0}}
"""
    )
    assert doc.safety.safe_pose is not None


def _typed_frame_src(sphere_ref: str, pose_ref: str) -> str:
    # A spec whose keep-out and safe pose carry the typed ReferenceFrame sibling (RFC-0007).
    return f"""
safety_version: "0.1"
safety:
  id: t
  name: t
  safe_pose:
    frame: MOON_ME
    frame_ref: {pose_ref}
    position_m: {{x: 40.0, y: 0.0, z: 0.0}}
  signals: [{{key: soc, unit: J, source: observation}}]
  constraints:
    - {{kind: energy_floor, id: e, on_uncertain: safe_state,
       energy_floor: {{signal: soc, floor_j: 1.0}}}}
    - kind: keep_out
      id: k
      keep_out:
        margin_m: 3.0
        volume:
          shape: sphere
          sphere:
            frame: MOON_ME
            frame_ref: {sphere_ref}
            center_m: {{x: 0.0, y: 0.0, z: 0.0}}
            radius_m: 30.0
"""


def test_typed_frame_ref_is_accepted_and_carried() -> None:
    ref = "{name: MOON_ME, frame_class: body_fixed, center: MOON}"
    doc = load_safety_spec(_typed_frame_src(ref, ref))
    sphere = doc.safety.constraints[1].keep_out.volume.sphere  # type: ignore[union-attr]
    assert sphere.frame_ref is not None and sphere.frame_ref.name == "MOON_ME"
    assert doc.safety.safe_pose.frame_ref.name == "MOON_ME"  # type: ignore[union-attr]


def test_frame_ref_name_must_match_frame_token() -> None:
    # The typed frame and the string frame token must name one frame (RFC-0007).
    good = "{name: MOON_ME, frame_class: body_fixed, center: MOON}"
    bad = "{name: OTHER, frame_class: body_fixed}"
    with pytest.raises(SafetySpecValidationError, match="does not match frame"):
        load_safety_spec(_typed_frame_src(bad, good))


def test_frame_ref_with_bad_frame_class_is_rejected_before_the_tcb() -> None:
    # An unknown frame_class fails require_frame at compile time (before the compiler / TCB).
    good = "{name: MOON_ME, frame_class: body_fixed, center: MOON}"
    unknown = "{name: MOON_ME, frame_class: galactic}"
    with pytest.raises(SafetySpecValidationError):
        load_safety_spec(_typed_frame_src(unknown, good))


# --- admissible_directives: the reviewed MODE/TASK grant (RFC-0004 Amendment 2) ----------------


def _spec_with(block: str) -> str:
    return f"""
safety_version: "0.1"
safety:
  id: t
  name: t
{block}
  signals: [{{key: soc, unit: J, source: observation}}]
  constraints:
    - kind: energy_floor
      id: c_soc
      energy_floor: {{signal: soc, floor_j: 1.0}}
"""


def test_the_anchor_authors_its_directive_grant(anchor_document: SafetyDocument) -> None:
    # The grant is part of the reviewed contract, so it is part of the contract's content hash —
    # which is what lets a SafetyVerdict's spec_content_hash bound what the gate could admit.
    grant = anchor_document.safety.admissible_directives
    assert grant is not None
    assert grant.modes == ["safe_hold"]
    assert [t.value for t in grant.tasks] == ["standby", "charge"]


def test_a_spec_may_be_silent_about_directives() -> None:
    # Absent is legal (the field is additive — every pre-Amendment-2 spec is silent) and means the
    # contract grants NOTHING. It does not mean "defer to the configuration".
    doc = load_safety_spec(_spec_with(""))
    assert doc.safety.admissible_directives is None


def test_an_unknown_task_kind_is_rejected_at_authoring_time() -> None:
    # `tasks` is Core's closed TaskKind vocabulary, `$ref`d by its published `$id` (RFC-0009 §1).
    # A typo'd or invented task is refused when the contract is *loaded* — not silently carried
    # into the trusted core as an allowlist entry nothing will ever match (or worse, something
    # will).
    with pytest.raises(SafetySpecValidationError):
        load_safety_spec(_spec_with("  admissible_directives: {tasks: [not_a_task]}"))


def test_the_grant_changes_the_contract_content_hash() -> None:
    # The whole point: two specs that differ only in what they authorise are DIFFERENT contracts,
    # and hash differently. Before Amendment 2 the grant lived in an unsigned config dict, so two
    # runs could report the same spec_content_hash and enforce different action gates
    # (guard.md §5, §9.3).
    silent = load_safety_spec(_spec_with(""))
    granting = load_safety_spec(_spec_with("  admissible_directives: {modes: [safe_hold]}"))
    assert silent.content_hash() != granting.content_hash()
