"""Content-addressing + Protobuf wire round-trip are byte-exact and hash-stable."""

from __future__ import annotations

from astro_mine.guard.spec import (
    CompiledSafetyModel,
    SafetyDocument,
    compiled_content_hash,
    compiled_from_wire,
    compiled_to_wire,
    from_wire,
    spec_content_hash,
    to_wire,
)


def test_spec_content_hash_form(anchor_document: SafetyDocument) -> None:
    h = spec_content_hash(anchor_document)
    assert h.startswith("sha256:")
    assert len(h) == len("sha256:") + 64
    # stable across repeated computation
    assert spec_content_hash(anchor_document) == h
    assert anchor_document.content_hash() == h


def test_compiled_content_hash_form(anchor_compiled: CompiledSafetyModel) -> None:
    h = compiled_content_hash(anchor_compiled)
    assert h.startswith("sha256:")
    assert compiled_content_hash(anchor_compiled) == h


def test_spec_wire_roundtrip_is_exact(anchor_document: SafetyDocument) -> None:
    wire = to_wire(anchor_document)
    assert isinstance(wire, bytes)
    back = from_wire(wire)
    assert back == anchor_document
    # byte-stable: re-serializing the round-tripped doc reproduces the same bytes
    assert to_wire(back) == wire
    assert back.content_hash() == anchor_document.content_hash()


def test_compiled_wire_roundtrip_is_exact(anchor_compiled: CompiledSafetyModel) -> None:
    wire = compiled_to_wire(anchor_compiled)
    back = compiled_from_wire(wire)
    assert back == anchor_compiled
    assert compiled_to_wire(back) == wire
    assert back.content_hash() == anchor_compiled.content_hash()


def test_collision_pair_roundtrips_when_set() -> None:
    from astro_mine.guard.spec import load_safety_spec

    doc = load_safety_spec(
        """
safety_version: "0.1"
safety:
  id: cp
  name: cp
  signals: [{key: s, unit: J, source: observation}]
  constraints:
    - kind: keep_out
      id: ko
      keep_out:
        margin_m: 1.0
        collision_pair: [rover_a, rover_b]
        volume:
          shape: sphere
          sphere: {frame: F, center_m: {x: 0, y: 0, z: 0}, radius_m: 1.0}
"""
    )
    assert doc.safety.constraints[0].keep_out is not None
    assert doc.safety.constraints[0].keep_out.collision_pair == ("rover_a", "rover_b")
    # survives the proto round-trip (a set pair), and an unset pair stays None
    assert from_wire(to_wire(doc)) == doc


def test_two_specs_differ_by_hash(anchor_document: SafetyDocument) -> None:
    other = anchor_document.model_copy(deep=True)
    other.safety.constraints[0].power_floor.floor_w += 1.0  # type: ignore[union-attr]
    assert other.content_hash() != anchor_document.content_hash()


def test_safe_pose_roundtrips_through_wire() -> None:
    # A compiled model carrying a safe pose (retreat target) round-trips byte-exactly through the
    # protobuf wire form, and the target survives (RM-P1-GUARD-04).
    from astro_mine.guard.spec import compile_spec, load_safety_spec

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
    model = compile_spec(doc)
    wire = compiled_to_wire(model)
    back = compiled_from_wire(wire)
    assert back == model
    assert back.safe_pose is not None
    assert back.safe_pose.position == [40.0, -1200.0, 3.0]
    assert compiled_to_wire(back) == wire  # byte-stable
