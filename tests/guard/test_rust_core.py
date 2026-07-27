"""End-to-end binding test: compile the anchor SafetySpec, hand its canonical wire form to the
Rust safety core (RM-P1-GUARD-02) over the PyO3 boundary, and check the arbiter's fail-safe
behaviour from Python.

This exercises the full contract seam: `compile_spec` → `compiled_to_wire` → the Rust core's
protobuf decode → per-tick certification. The core itself is validated exhaustively in the Rust
crate (`rust/tests/`); this proves the same core is reachable and behaves through the binding.
"""

from __future__ import annotations

import math
from importlib import resources

import pytest

from astro_mine.guard.spec.compiler import compile_spec
from astro_mine.guard.spec.loader import load_safety_spec
from astro_mine.guard.spec.wire import compiled_to_wire

_core = pytest.importorskip(
    "astro_mine.guard._core",
    reason="Rust safety core extension not built (run `maturin develop` / `uv sync`)",
)

ANCHOR = resources.files("astro_mine.guard.reference").joinpath(
    "safety_specs", "anchor.safety.yaml"
)

# Signal order is the sorted signal-key table the compiler emits (see compiler._signal_index).
SIGNALS = [
    "anchor_torque_nm",
    "battery_soc_j",
    "charging_window_active",
    "chassis_temp_k",
    "power_available_w",
    "traverse_speed_mps",
]


def _safe_signals() -> list[float]:
    """A signal vector satisfying every scalar bound and temporal clause."""
    return [10.0, 200_000.0, 1.0, 250.0, 20.0, 0.05]


def _core_from_anchor() -> object:
    doc = load_safety_spec(ANCHOR.read_text())
    # A coarse sample period keeps the two 14-day survival monitors' history windows tiny
    # (≈10 samples) so the pre-allocated ring buffers stay small for the test.
    model = compile_spec(doc, sample_period_s=120_960.0)
    wire = compiled_to_wire(model)
    return _core.SafetyCore.from_wire(wire, u_max=20.0)


def test_loads_and_reports_provenance() -> None:
    core = _core_from_anchor()
    assert core.spec_id == "anchor-lunar-polar-v0"
    assert core.spec_content_hash.startswith("sha256:")
    # Keep-out geometry is 3-D (MOON_ME frame).
    assert core.spatial_dim == 3


def test_benign_action_far_from_keepout_is_certified() -> None:
    core = _core_from_anchor()
    v = core.step(
        signals=_safe_signals(),
        position=[40.0, 0.0, 0.0],  # 40 m out; lander keep-out is a 33 m sphere at the origin
        velocity=[0.0, 0.0, 0.0],
        proposed_action=[0.1, 0.0, 0.0],
    )
    assert v["layer"] in ("primary", "shield")
    assert v["reason"] in ("certified", "shield_corrected")
    assert all(math.isfinite(a) for a in v["certified_action"])


def test_adversarial_action_into_keepout_is_not_passed_through() -> None:
    core = _core_from_anchor()
    # Near the lander sphere boundary (h ≈ 1 m), moving inward, commanding full inward thrust.
    v = core.step(
        signals=_safe_signals(),
        position=[34.0, 0.0, 0.0],
        velocity=[-3.0, 0.0, 0.0],
        proposed_action=[-20.0, 0.0, 0.0],
    )
    # The arbiter must not pass the inward proposal through: either the shield corrected it
    # (outward x-acceleration) or it fell back — never the raw adversarial action.
    assert v["layer"] in ("shield", "backup")
    assert v["reason"] in ("shield_corrected", "qp_uncertifiable")
    action = v["certified_action"]
    assert all(math.isfinite(a) and abs(a) <= 20.0 + 1e-6 for a in action)
    if v["layer"] == "shield":
        assert action[0] > -20.0  # was pushed away from the raw inward command
    assert math.isfinite(v["min_barrier_margin"])


def test_scalar_floor_violation_forces_backup() -> None:
    core = _core_from_anchor()
    signals = _safe_signals()
    signals[4] = 5.0  # power_available_w below the 15 W survival floor
    v = core.step(
        signals=signals,
        position=[40.0, 0.0, 0.0],
        velocity=[0.0, 0.0, 0.0],
        proposed_action=[0.0, 0.0, 0.0],
    )
    assert v["layer"] == "backup"
    assert v["reason"] == "scalar_violated"
    assert "c_power_floor" in v["fired"]
    assert v["backup_kind"] == "brake_to_stop"


def test_nan_signal_fails_safe() -> None:
    core = _core_from_anchor()
    signals = _safe_signals()
    signals[1] = float("nan")
    v = core.step(
        signals=signals,
        position=[40.0, 0.0, 0.0],
        velocity=[0.0, 0.0, 0.0],
        proposed_action=[0.0, 0.0, 0.0],
    )
    assert v["layer"] == "backup"
    assert v["reason"] == "bad_input"
    assert all(math.isfinite(a) for a in v["certified_action"])


def test_rejecting_unbounded_history_is_fail_closed() -> None:
    # Compiling at the native 1 s period yields a 14-day (1_209_600-sample) window that exceeds
    # the default history cap — the core must refuse to load rather than allocate unboundedly.
    doc = load_safety_spec(ANCHOR.read_text())
    wire = compiled_to_wire(compile_spec(doc, sample_period_s=1.0))
    with pytest.raises(ValueError):
        _core.SafetyCore.from_wire(wire, max_history_cap=1024)
