"""Contract stubs: each Sim subpackage exists and its placeholder is wired.

Pins the public surface and the not-yet-implemented entry points so the backlog has a
concrete starting point. Replace the placeholder checks with real behavior as each
RM-P0-SIM-* item lands (``runtime`` is implemented by RM-P0-SIM-01).
"""

from __future__ import annotations

from astro_mine.sim import (
    comms,
    coupling,
    engines,
    power_thermal,
    recording,
    runtime,
    scheduler,
    sensors,
    validation,
)


def test_subpackages_import() -> None:
    modules = (
        runtime,
        engines,
        coupling,
        scheduler,
        sensors,
        power_thermal,
        comms,
        recording,
        validation,
    )
    for module in modules:
        assert module is not None


def test_regime_engine_contract_is_exported() -> None:
    assert hasattr(engines, "RegimeEngine")


def test_runtime_stepping_core_is_implemented() -> None:
    # RM-P0-SIM-01 landed: the deterministic stepping core is wired, not stubbed.
    for name in ("Simulator", "Scenario", "SimClock", "RngStreams", "Trace", "run_episode"):
        assert hasattr(runtime, name)
