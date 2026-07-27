"""Allocation IR Protobuf wire form — exact, byte-stable round-trip (RM-P1-ALLOC-01).

The canonical cross-language encoding Sim/Bench and solver backends consume (allocate.md §5).
A model -> bytes -> model round-trip must be exact, and the bytes must be deterministic, so a
content hash over the wire form is portable (the Core messages.wire pattern) and the anchor
request round-trips through the IR losslessly (acceptance criterion).
"""

from __future__ import annotations

import pytest

from astro_mine.allocate import (
    AllocationIR,
    DecisionVariable,
    ObjectiveSense,
    VariableKind,
    VariableSemantic,
    compile_request,
    ir_from_wire,
    ir_to_proto,
    ir_to_wire,
)
from tests.allocate.factories import anchor_request, unwindowed_request

_ANCHOR_IR = compile_request(anchor_request())
_UNWINDOWED_IR = compile_request(unwindowed_request())


@pytest.mark.parametrize("ir", [_ANCHOR_IR, _UNWINDOWED_IR], ids=["anchor", "unwindowed"])
def test_round_trip_is_exact(ir: AllocationIR) -> None:
    assert ir_from_wire(ir_to_wire(ir)) == ir


@pytest.mark.parametrize("ir", [_ANCHOR_IR, _UNWINDOWED_IR], ids=["anchor", "unwindowed"])
def test_serialization_is_byte_stable(ir: AllocationIR) -> None:
    assert ir_to_wire(ir) == ir_to_wire(ir)


def test_optional_absence_round_trips() -> None:
    # A start-time variable carries no asset_ref, and an unbounded variable no lower/upper;
    # absence must survive the wire (proto3 `optional` presence).
    ir = AllocationIR(
        objective_sense=ObjectiveSense.MINIMIZE,
        variables=[
            DecisionVariable(
                id="start::t", kind=VariableKind.CONTINUOUS, semantic=VariableSemantic.START_TIME
            )
        ],
    )
    restored = ir_from_wire(ir_to_wire(ir))
    assert restored == ir
    var = restored.variables[0]
    assert var.asset_ref is None
    assert var.lower is None and var.upper is None


def test_zero_valued_bounds_round_trip() -> None:
    # 0.0 is a *present* bound (distinct from absent) and must survive as 0.0, not drop out.
    ir = AllocationIR(
        objective_sense=ObjectiveSense.MAXIMIZE,
        variables=[
            DecisionVariable(
                id="assign::t::a",
                kind=VariableKind.BINARY,
                lower=0.0,
                upper=1.0,
                semantic=VariableSemantic.ASSIGNMENT,
                task_ref="t",
                asset_ref="a",
            )
        ],
    )
    restored = ir_from_wire(ir_to_wire(ir))
    assert restored.variables[0].lower == 0.0
    assert restored.variables[0].upper == 1.0


def test_metadata_map_round_trips() -> None:
    assert ir_from_wire(ir_to_wire(_ANCHOR_IR)).metadata == {"request_id": "lunar-polar-ice-001"}


def test_to_proto_exposes_the_typed_message() -> None:
    msg = ir_to_proto(_ANCHOR_IR)
    assert msg.ir_version == "0.1.0"
    assert msg.objective_sense == "maximize"
    assert len(msg.variables) == len(_ANCHOR_IR.variables)


def test_content_hash_stable_across_wire_round_trip() -> None:
    assert ir_from_wire(ir_to_wire(_ANCHOR_IR)).content_hash() == _ANCHOR_IR.content_hash()
