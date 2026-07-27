"""Determinism of constrained compilation (RM-P1-ALLOC-03 acceptance; feeds RM-P1-ALLOC-07).

Constraint compilation from fixed, content-hashed inputs is **byte-stable**: the same
``(request, context, config, cost-table)`` compiles to a byte-identical augmented IR (identical
content hash and wire bytes), and any config/cost-table change alters the IR hash — so the
golden-plan gate can pin the exact model a plan was produced from.
"""

from __future__ import annotations

from astro_mine.allocate import (
    ConstraintConfig,
    ConstraintContext,
    compile_with_constraints,
    ir_to_wire,
)
from astro_mine.allocate.constraints import CostTable
from astro_mine.allocate.constraints.config import CommsPolicy, TerrainPolicy
from astro_mine.core.units import J2000_EPOCH
from tests.allocate import constraint_factories as F
from tests.allocate.factories import anchor_request

_CONFIG = ConstraintConfig(
    comms=CommsPolicy(relay_required_task_ids=frozenset({"haul-to-plant"}), epoch0=J2000_EPOCH)
)
_COSTS = F.cost_table(
    {
        ("haul-to-plant", "hauler-1"): (600.0, 1.0e6),
        ("excavate-crater-a", "excavator-1"): (900.0, 3.0e6),
    }
)


def _ctx() -> ConstraintContext:
    return F.context(
        world=F.FakeWorld(slope_deg=4.0),
        contacts=F.contact_plan({"hauler-1": (5000.0, 9000.0)}),
        resource=F.FakeField(mean=0.4, variance=0.02),
    )


def test_same_inputs_compile_to_a_byte_identical_ir() -> None:
    request = anchor_request()
    a = compile_with_constraints(request, _ctx(), config=_CONFIG, costs=_COSTS)
    b = compile_with_constraints(request, _ctx(), config=_CONFIG, costs=_COSTS)
    assert a.ir.content_hash() == b.ir.content_hash()
    assert ir_to_wire(a.ir) == ir_to_wire(b.ir)  # byte-for-byte


def test_changing_the_config_changes_the_ir_hash() -> None:
    request = anchor_request()
    base = compile_with_constraints(request, _ctx(), config=_CONFIG, costs=_COSTS)
    tighter = ConstraintConfig(
        comms=_CONFIG.comms, terrain=TerrainPolicy(default_max_slope_deg=10.0)
    )
    changed = compile_with_constraints(request, _ctx(), config=tighter, costs=_COSTS)
    assert base.ir.content_hash() != changed.ir.content_hash()
    assert (
        base.ir.metadata["constraint_config_hash"] != changed.ir.metadata["constraint_config_hash"]
    )


def test_changing_the_cost_table_changes_the_ir_hash() -> None:
    request = anchor_request()
    base = compile_with_constraints(request, _ctx(), config=_CONFIG, costs=_COSTS)
    dearer = F.cost_table({("excavate-crater-a", "excavator-1"): (900.0, 9.0e6)})
    changed = compile_with_constraints(request, _ctx(), config=_CONFIG, costs=dearer)
    assert base.ir.content_hash() != changed.ir.content_hash()


def test_input_content_hashes_are_stable_across_instances() -> None:
    # The config/cost-table are content-addressed value objects — two equal instances hash equally.
    assert ConstraintConfig().content_hash() == ConstraintConfig().content_hash()
    assert (
        _COSTS.content_hash()
        == F.cost_table(
            {
                ("haul-to-plant", "hauler-1"): (600.0, 1.0e6),
                ("excavate-crater-a", "excavator-1"): (900.0, 3.0e6),
            }
        ).content_hash()
    )
    assert CostTable().content_hash() != _COSTS.content_hash()
