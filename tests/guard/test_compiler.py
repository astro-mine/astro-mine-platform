"""Unit tests for the constraint compiler — the lowering, static-bounds pass, and determinism."""

from __future__ import annotations

import math

import pytest

from astro_mine.guard.spec import (
    CompiledSafetyModel,
    PredicateOp,
    SafetyDocument,
    load_safety_spec,
)
from astro_mine.guard.spec.compiler import CompileError, compile_spec


def _doc(constraints: str, signals: str = "") -> SafetyDocument:
    sig = signals or "[{key: soc, unit: J, source: observation}]"
    return load_safety_spec(
        f"""
safety_version: "0.1"
safety:
  id: t
  name: t
  signals: {sig}
  constraints:
{constraints}
"""
    )


def test_scalar_floor_and_ceiling_ops() -> None:
    doc = _doc(
        """
    - {kind: power_floor, id: p, power_floor: {signal: pw, floor_w: 15.0}}
    - {kind: thermal_ceiling, id: t, thermal_ceiling: {signal: tk, limit_k: 320.0}}
""",
        signals="[{key: pw, unit: W, source: sadf}, {key: tk, unit: K, source: observation}]",
    )
    m = compile_spec(doc)
    ops = {
        (m.predicate_table.signals[a.signal_index], a.op, a.threshold)
        for a in m.predicate_table.atoms
    }
    assert ("pw", PredicateOp.GE, 15.0) in ops  # a floor is >=
    assert ("tk", PredicateOp.LE, 320.0) in ops  # a ceiling is <=


def test_predicate_dedup() -> None:
    # The energy floor and the temporal predicate share the same atom (soc >= 100) -> one slot.
    doc = _doc(
        """
    - {kind: energy_floor, id: e, energy_floor: {signal: soc, floor_j: 100.0}}
    - kind: temporal
      id: t
      temporal:
        formula:
          op: always
          interval_s: {lo: 0.0, hi: 10.0}
          args:
            - {op: predicate, signal: soc, cmp: ge, threshold: 100.0}
"""
    )
    m = compile_spec(doc)
    assert m.resource_bounds.predicate_slot_count == 1  # deduplicated


def test_half_space_is_normalized() -> None:
    doc = _doc(
        """
    - kind: keep_out
      id: ko
      keep_out:
        margin_m: 1.0
        volume:
          shape: half_space
          half_space: {frame: F, normal: {x: 3.0, y: 4.0, z: 0.0}, offset_m: 10.0}
""",
    )
    m = compile_spec(doc)
    term = m.keep_out_terms[0]
    assert term.normal is not None and term.offset is not None
    # (3,4,0) normalizes to (0.6, 0.8, 0.0); offset scales by 1/5.
    assert term.normal == pytest.approx([0.6, 0.8, 0.0])
    assert term.offset == pytest.approx(2.0)
    assert math.isclose(math.sqrt(sum(n * n for n in term.normal)), 1.0)


def test_box_half_extents_and_sphere_radius() -> None:
    doc = _doc(
        """
    - kind: keep_out
      id: box
      keep_out:
        margin_m: 2.0
        volume:
          shape: box
          box: {frame: F, center_m: {x: 1, y: 2, z: 3}, dimensions_m: {x: 10, y: 20, z: 40}}
    - kind: keep_out
      id: sph
      keep_out:
        margin_m: 1.0
        volume:
          shape: sphere
          sphere: {frame: F, center_m: {x: 0, y: 0, z: 0}, radius_m: 7.0}
""",
    )
    m = compile_spec(doc)
    by_id = {t.constraint_id: t for t in m.keep_out_terms}
    assert by_id["box"].center == [1.0, 2.0, 3.0]
    assert by_id["box"].half_extents == [5.0, 10.0, 20.0]  # dimensions / 2
    assert by_id["sph"].radius == 7.0
    assert by_id["sph"].center == [0.0, 0.0, 0.0]


def test_kinematic_two_bounds_two_atoms() -> None:
    doc = _doc(
        """
    - kind: kinematic_limit
      id: k
      kinematic_limit: {signal: v, max_velocity_mps: 0.5, max_accel_mps2: 0.1}
""",
        signals="[{key: v, unit: 'm/s', source: observation}]",
    )
    m = compile_spec(doc)
    bounds = [b for b in m.scalar_bounds if b.constraint_id == "k"]
    assert len(bounds) == 2  # one atom per set bound


def test_history_window_scales_with_sample_period() -> None:
    doc = _doc(
        """
    - kind: temporal
      id: t
      temporal:
        formula:
          op: eventually
          interval_s: {lo: 0.0, hi: 100.0}
          args:
            - {op: predicate, signal: soc, cmp: ge, threshold: 1.0}
"""
    )
    fine = compile_spec(doc, sample_period_s=1.0)
    coarse = compile_spec(doc, sample_period_s=10.0)
    assert fine.monitors[0].history_window_len == 100
    assert coarse.monitors[0].history_window_len == 10


def test_nested_temporal_history_is_conservative() -> None:
    # always[0,20](eventually[0,30] p): the buffer bound sums the nested horizons.
    doc = _doc(
        """
    - kind: temporal
      id: t
      temporal:
        formula:
          op: always
          interval_s: {lo: 0.0, hi: 20.0}
          args:
            - op: eventually
              interval_s: {lo: 0.0, hi: 30.0}
              args:
                - {op: predicate, signal: soc, cmp: ge, threshold: 1.0}
"""
    )
    m = compile_spec(doc)
    assert m.monitors[0].history_window_len == 50  # 20 + 30, conservative upper bound


def test_compile_is_deterministic_and_sorted() -> None:
    doc = _doc(
        """
    - {kind: thermal_ceiling, id: zzz, thermal_ceiling: {signal: tk, limit_k: 1.0}}
    - {kind: thermal_floor, id: aaa, thermal_floor: {signal: tk, limit_k: 2.0}}
""",
        signals="[{key: tk, unit: K, source: observation}]",
    )
    m1 = compile_spec(doc)
    m2 = compile_spec(doc)
    assert m1 == m2
    assert m1.content_hash() == m2.content_hash()
    # scalar bounds are sorted by constraint id
    assert [b.constraint_id for b in m1.scalar_bounds] == ["aaa", "zzz"]


def test_reject_nonpositive_sample_period(anchor_document: SafetyDocument) -> None:
    with pytest.raises(CompileError, match="must be positive"):
        compile_spec(anchor_document, sample_period_s=0.0)


def test_resource_bounds_are_upper_bounds(anchor_compiled: CompiledSafetyModel) -> None:
    rb = anchor_compiled.resource_bounds
    assert rb.predicate_slot_count == len(anchor_compiled.predicate_table.atoms)
    assert rb.scalar_bound_count == len(anchor_compiled.scalar_bounds)
    assert rb.keep_out_term_count == len(anchor_compiled.keep_out_terms)
    assert rb.monitor_count == len(anchor_compiled.monitors)
    assert rb.max_history_len == max(mon.history_window_len for mon in anchor_compiled.monitors)
    assert rb.worst_case_term_count == (
        rb.predicate_slot_count
        + rb.keep_out_term_count
        + sum(mon.node_count for mon in anchor_compiled.monitors)
    )


def test_safe_pose_lowers_to_position_in_keepout_frame() -> None:
    # The authored safe pose is lowered to a bare position vector in the keep-out frame — the
    # target the verified retreat (safe_state) backup steers toward (RM-P1-GUARD-04).
    doc = load_safety_spec(
        """
safety_version: "0.1"
safety:
  id: t
  name: t
  safe_pose: {frame: MOON_ME, position_m: {x: 40.0, y: -1200.0, z: 3.0}}
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
    )
    m = compile_spec(doc)
    assert m.safe_pose is not None
    assert m.safe_pose.frame == "MOON_ME"
    assert m.safe_pose.position == [40.0, -1200.0, 3.0]


def test_no_safe_pose_lowers_to_none() -> None:
    m = compile_spec(
        _doc("    - {kind: energy_floor, id: e, energy_floor: {signal: soc, floor_j: 1.0}}")
    )
    assert m.safe_pose is None


# --- admissible_directives lowering (RFC-0004 Amendment 2) -------------------------------------


def test_admissible_directives_lower_sorted_and_deduplicated() -> None:
    doc = load_safety_spec(
        """
safety_version: "0.1"
safety:
  id: t
  name: t
  admissible_directives:
    modes: [safe_hold, drill, safe_hold]
    tasks: [standby, charge, standby]
  signals: [{key: soc, unit: J, source: observation}]
  constraints:
    - kind: energy_floor
      id: c_soc
      energy_floor: {signal: soc, floor_j: 1.0}
"""
    )
    m = compile_spec(doc, sample_period_s=1.0)
    assert m.admissible_directives is not None
    # Sorted + deduplicated, so the lowering is byte-for-byte reproducible (the golden gate) and the
    # core's membership scan is over a canonical set.
    assert m.admissible_directives.modes == ["drill", "safe_hold"]
    assert m.admissible_directives.tasks == ["charge", "standby"]


def test_a_silent_spec_lowers_to_no_grant() -> None:
    # And absence means the empty set, NOT "the configured allowlist stands" — the deliberate
    # asymmetry with `action_limits`, whose absent members do leave the configured ceiling standing.
    m = compile_spec(
        _doc("    - {kind: energy_floor, id: c, energy_floor: {signal: soc, floor_j: 1.0}}")
    )
    assert m.admissible_directives is None
    assert m.action_limits.max_velocity_mps is None  # the ceiling's identity really is +inf
