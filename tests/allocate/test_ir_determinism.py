"""IR determinism: a fixed request compiles to a byte-stable IR (RM-P1-ALLOC-01).

Same model + same seed ⇒ same plan (allocate.md §8); the prerequisite is that the IR
compiled from a fixed request is byte-stable and independent of input ordering — the
foundation of the seeded golden-plan gate (RM-P1-ALLOC-07).
"""

from __future__ import annotations

from astro_mine.allocate import compile_request, ir_to_wire
from tests.allocate.factories import anchor_request


def test_compile_is_byte_stable_across_repeats() -> None:
    ir_a = compile_request(anchor_request())
    ir_b = compile_request(anchor_request())
    assert ir_a == ir_b
    assert ir_a.content_hash() == ir_b.content_hash()
    assert ir_to_wire(ir_a) == ir_to_wire(ir_b)


def test_compile_is_independent_of_task_and_asset_ordering() -> None:
    request = anchor_request()
    shuffled = request.model_copy(
        update={
            "tasks": list(reversed(request.tasks)),
            "assets": list(reversed(request.assets)),
        }
    )
    canonical = compile_request(request)
    from_shuffled = compile_request(shuffled)
    assert canonical == from_shuffled
    assert ir_to_wire(canonical) == ir_to_wire(from_shuffled)
