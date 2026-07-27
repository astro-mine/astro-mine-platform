"""Property-based tests for the compiler (guard.md §9.3 — the Hypothesis analogue of the
Rust core's proptest gate).

For any randomly-generated **valid** SafetySpec the compiler must be:

1. **total** — it never raises (a validated spec always lowers);
2. **deterministic** — two compiles produce an identical content hash;
3. **sound on bounds** — the static-bounds pass never under-counts (every resource bound is
   at least the real requirement), so a pre-allocating core is never handed too small a buffer;
4. **complete** — every declared constraint appears in the compiled IR.
"""

from __future__ import annotations

import math
from typing import Any

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from astro_mine.guard.spec import SafetyDocument, compile_spec, validate_safety_spec
from astro_mine.guard.spec.ir import CompiledNode

_floats = st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
_pos_floats = st.floats(min_value=1e-3, max_value=1e6, allow_nan=False, allow_infinity=False)
_nonneg_floats = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)
_horizon = st.floats(min_value=0.0, max_value=1e5, allow_nan=False, allow_infinity=False)
_cmp = st.sampled_from(["lt", "le", "gt", "ge"])
_on_uncertain = st.sampled_from(["fallback", "hold", "safe_state"])


@st.composite
def _vec3(draw: st.DrawFn) -> dict[str, float]:
    return {"x": draw(_floats), "y": draw(_floats), "z": draw(_floats)}


@st.composite
def _interval(draw: st.DrawFn) -> dict[str, float]:
    lo = draw(_horizon)
    hi = lo + draw(st.floats(min_value=0.0, max_value=1e5, allow_nan=False, allow_infinity=False))
    return {"lo": lo, "hi": hi}


@st.composite
def _predicate(draw: st.DrawFn, keys: list[str]) -> dict[str, Any]:
    return {
        "op": "predicate",
        "signal": draw(st.sampled_from(keys)),
        "cmp": draw(_cmp),
        "threshold": draw(_floats),
    }


@st.composite
def _formula(draw: st.DrawFn, keys: list[str], depth: int) -> dict[str, Any]:
    if depth <= 0:
        return draw(_predicate(keys))
    op = draw(st.sampled_from(["predicate", "not", "and", "or", "always", "eventually", "until"]))
    if op == "predicate":
        return draw(_predicate(keys))
    if op == "not":
        return {"op": "not", "args": [draw(_formula(keys, depth - 1))]}
    if op in ("and", "or"):
        n = draw(st.integers(min_value=2, max_value=3))
        return {"op": op, "args": [draw(_formula(keys, depth - 1)) for _ in range(n)]}
    if op in ("always", "eventually"):
        return {
            "op": op,
            "interval_s": draw(_interval()),
            "args": [draw(_formula(keys, depth - 1))],
        }
    # until
    return {
        "op": "until",
        "interval_s": draw(_interval()),
        "args": [draw(_formula(keys, depth - 1)), draw(_formula(keys, depth - 1))],
    }


@st.composite
def _keep_out(draw: st.DrawFn) -> dict[str, Any]:
    shape = draw(st.sampled_from(["box", "sphere", "half_space"]))
    if shape == "box":
        vol = {
            "shape": "box",
            "box": {"frame": "F", "center_m": draw(_vec3()), "dimensions_m": draw(_vec3())},
        }
    elif shape == "sphere":
        vol = {
            "shape": "sphere",
            "sphere": {"frame": "F", "center_m": draw(_vec3()), "radius_m": draw(_pos_floats)},
        }
    else:
        # guarantee a non-zero normal
        nx = draw(st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False))
        vol = {
            "shape": "half_space",
            "half_space": {
                "frame": "F",
                "normal": {"x": nx, "y": draw(_floats), "z": draw(_floats)},
                "offset_m": draw(_floats),
            },
        }
    return {"margin_m": draw(_nonneg_floats), "volume": vol}


@st.composite
def _constraint(draw: st.DrawFn, keys: list[str], cid: str) -> dict[str, Any]:
    kind = draw(
        st.sampled_from(
            [
                "power_floor",
                "energy_floor",
                "thermal_ceiling",
                "thermal_floor",
                "torque_ceiling",
                "kinematic_limit",
                "keep_out",
                "temporal",
            ]
        )
    )
    base: dict[str, Any] = {"kind": kind, "id": cid, "on_uncertain": draw(_on_uncertain)}
    signal = draw(st.sampled_from(keys))
    if kind == "power_floor":
        base["power_floor"] = {"signal": signal, "floor_w": draw(_floats)}
    elif kind == "energy_floor":
        base["energy_floor"] = {"signal": signal, "floor_j": draw(_floats)}
    elif kind == "thermal_ceiling":
        base["thermal_ceiling"] = {"signal": signal, "limit_k": draw(_floats)}
    elif kind == "thermal_floor":
        base["thermal_floor"] = {"signal": signal, "limit_k": draw(_floats)}
    elif kind == "torque_ceiling":
        base["torque_ceiling"] = {"signal": signal, "max_nm": draw(_pos_floats)}
    elif kind == "kinematic_limit":
        km: dict[str, Any] = {"signal": signal}
        which = draw(st.sampled_from(["v", "a", "both"]))
        if which in ("v", "both"):
            km["max_velocity_mps"] = draw(_nonneg_floats)
        if which in ("a", "both"):
            km["max_accel_mps2"] = draw(_nonneg_floats)
        base["kinematic_limit"] = km
    elif kind == "keep_out":
        base["keep_out"] = draw(_keep_out())
    else:  # temporal
        base["temporal"] = {"formula": draw(_formula(keys, draw(st.integers(0, 3))))}
    return base


@st.composite
def _document(draw: st.DrawFn) -> dict[str, Any]:
    n_sig = draw(st.integers(min_value=1, max_value=4))
    keys = [f"sig{i}" for i in range(n_sig)]
    signals = [{"key": k, "unit": "u", "source": "observation"} for k in keys]
    n_con = draw(st.integers(min_value=1, max_value=6))
    constraints = [draw(_constraint(keys, f"c{i}")) for i in range(n_con)]
    return {
        "safety_version": "0.1",
        "safety": {"id": "hyp", "name": "hyp", "signals": signals, "constraints": constraints},
    }


def _max_single_horizon(node: CompiledNode) -> int:
    """A lower bound on a monitor's true history need: the largest single horizon in the tree."""
    here = node.interval_hi_samples or 0
    return max([here, *(_max_single_horizon(c) for c in node.args)])


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(data=_document())
def test_compile_is_total_deterministic_sound_and_complete(data: dict[str, Any]) -> None:
    # Generated specs are valid by construction: the loader accepts them (dict path, so the
    # exact generated floats are validated — not a YAML-reparse of them).
    validate_safety_spec(data)
    doc = SafetyDocument.model_validate(data)

    # 1. total: compilation never raises.
    m = compile_spec(doc)

    # 2. deterministic.
    assert compile_spec(doc).content_hash() == m.content_hash()

    # 3. sound bounds — never under-counts.
    rb = m.resource_bounds
    assert rb.predicate_slot_count == len(m.predicate_table.atoms)
    assert rb.scalar_bound_count == len(m.scalar_bounds)
    assert rb.keep_out_term_count == len(m.keep_out_terms)
    assert rb.monitor_count == len(m.monitors)
    assert rb.max_history_len >= max((mon.history_window_len for mon in m.monitors), default=0)
    for mon in m.monitors:
        assert mon.history_window_len >= _max_single_horizon(mon.root)
        for idx in mon.predicate_indices:
            assert 0 <= idx < rb.predicate_slot_count
    assert rb.worst_case_term_count >= rb.predicate_slot_count + rb.keep_out_term_count

    # every atom references a real signal slot; every scalar bound a real atom.
    n_signals = len(m.predicate_table.signals)
    for atom in m.predicate_table.atoms:
        assert 0 <= atom.signal_index < n_signals
        assert math.isfinite(atom.threshold)
    for b in m.scalar_bounds:
        assert 0 <= b.atom_index < rb.predicate_slot_count

    # 4. complete — every declared constraint id appears in the IR.
    covered = (
        {b.constraint_id for b in m.scalar_bounds}
        | {t.constraint_id for t in m.keep_out_terms}
        | {mon.constraint_id for mon in m.monitors}
    )
    assert covered == {c.id for c in doc.safety.constraints}
