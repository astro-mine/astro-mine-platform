"""Anchor profile builder — SADF threshold binding, safe_pose, sample period (RM-P1-GUARD-04)."""

from __future__ import annotations

from astro_mine.guard.models import (
    ANCHOR_MAX_HISTORY_CAP,
    ANCHOR_SAMPLE_PERIOD_S,
    NIGHT_SURVIVAL_ENERGY_FRACTION,
    SadfBudgets,
    WorldsFleetSignalResolver,
    anchor_core_config,
    build_anchor_resolver,
    compile_anchor,
    load_anchor_document,
    with_sadf_thresholds,
    with_safe_pose,
)
from astro_mine.guard.spec.enums import ConstraintKind
from astro_mine.guard.spec.model import SafePose, Vec3
from tests.guard.models_fixtures import sadf_document


def _threshold(model_or_doc, cid: str) -> float:
    for c in model_or_doc.safety.constraints:
        if c.id == cid:
            payload = getattr(
                c,
                {
                    "c_power_floor": "power_floor",
                    "c_energy_floor": "energy_floor",
                    "c_thermal_ceiling": "thermal_ceiling",
                    "c_thermal_floor": "thermal_floor",
                    "c_anchor_torque": "torque_ceiling",
                }[cid],
            )
            for attr in ("floor_w", "floor_j", "limit_k", "max_nm"):
                if hasattr(payload, attr):
                    return float(getattr(payload, attr))
    raise AssertionError(cid)


def test_load_anchor_document() -> None:
    doc = load_anchor_document()
    assert doc.safety.id == "anchor-lunar-polar-v0"


def test_with_sadf_thresholds_rebinds_from_budgets() -> None:
    doc = load_anchor_document()
    budgets = SadfBudgets.from_document(sadf_document(capacity_j=1_000_000.0))
    bound = with_sadf_thresholds(doc, budgets)
    assert _threshold(bound, "c_power_floor") == 15.0
    assert _threshold(bound, "c_thermal_ceiling") == 320.0
    assert _threshold(bound, "c_thermal_floor") == 100.0  # survival min
    assert _threshold(bound, "c_anchor_torque") == 42.0
    # energy floor = fraction * capacity
    assert _threshold(bound, "c_energy_floor") == 1_000_000.0 * NIGHT_SURVIVAL_ENERGY_FRACTION
    # the original document is unchanged (pure function)
    assert _threshold(doc, "c_anchor_torque") == 40.0


def test_with_sadf_thresholds_leaves_absent_budgets_untouched() -> None:
    doc = load_anchor_document()
    budgets = SadfBudgets.from_document(
        sadf_document(floor_w=None, torque_nm=None, capacity_j=None)
    )
    bound = with_sadf_thresholds(doc, budgets)
    # No SADF floor/torque/capacity ⇒ the authored anchor thresholds are preserved.
    assert _threshold(bound, "c_power_floor") == 15.0
    assert _threshold(bound, "c_anchor_torque") == 40.0
    assert _threshold(bound, "c_energy_floor") == _threshold(doc, "c_energy_floor")


def test_with_safe_pose_overrides_the_authored_target() -> None:
    doc = load_anchor_document()
    authored = doc.safety.safe_pose  # the anchor authors a charging pose at (60, 0, 5)
    assert authored is not None
    pose = SafePose(frame="MOON_ME", position_m=Vec3(x=40.0, y=0.0, z=0.0))
    out = with_safe_pose(doc, pose)
    assert out.safety.safe_pose == pose  # overridden
    assert doc.safety.safe_pose == authored  # original untouched (pure function)


def test_compile_anchor_uses_survival_period_and_carries_safe_pose() -> None:
    pose = SafePose(frame="MOON_ME", position_m=Vec3(x=45.0, y=0.0, z=3.0))
    budgets = SadfBudgets.from_document(sadf_document())
    model = compile_anchor(budgets=budgets, safe_pose=pose)
    assert model.sample_period_s == ANCHOR_SAMPLE_PERIOD_S == 1.0
    assert model.safe_pose is not None
    assert model.safe_pose.position == [45.0, 0.0, 3.0]
    # SADF torque budget flowed into the compiled torque ceiling.
    torque_atoms = [
        model.predicate_table.atoms[b.atom_index].threshold
        for b in model.scalar_bounds
        if b.constraint_id == "c_anchor_torque"
    ]
    assert torque_atoms == [42.0]


def test_anchor_core_config_cap_covers_survival_window() -> None:
    cfg = anchor_core_config()
    assert cfg.max_history_cap == ANCHOR_MAX_HISTORY_CAP
    # The 14-day survival window at 1 Hz fits under the cap.
    model = compile_anchor()
    assert model.resource_bounds.max_history_len <= cfg.max_history_cap
    # An override still lands.
    assert anchor_core_config(u_max=5.0).u_max == 5.0


def test_build_anchor_resolver_binds_worlds_signals() -> None:
    budgets = SadfBudgets.from_document(sadf_document())
    resolver = build_anchor_resolver(budgets=budgets)
    assert isinstance(resolver, WorldsFleetSignalResolver)
    assert resolver.budgets is budgets
    # charging_window_active is Worlds-bound; with no terrain/position it fails safe to NaN.
    import math

    assert math.isnan(resolver.resolve(["charging_window_active"], None)[0])


def test_constraint_kinds_are_exhaustive_for_the_anchor() -> None:
    # Sanity: the anchor exercises every scalar kind the rebinder handles.
    kinds = {c.kind for c in load_anchor_document().safety.constraints}
    assert {
        ConstraintKind.POWER_FLOOR,
        ConstraintKind.ENERGY_FLOOR,
        ConstraintKind.THERMAL_CEILING,
        ConstraintKind.THERMAL_FLOOR,
        ConstraintKind.TORQUE_CEILING,
    } <= kinds
