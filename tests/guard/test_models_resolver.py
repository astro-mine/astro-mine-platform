"""WorldsFleetSignalResolver — signal resolution with fail-safe NaN (RM-P1-GUARD-04)."""

from __future__ import annotations

import math

from astro_mine.core.world.model import IlluminationState
from astro_mine.guard.models import WorldsFleetSignalResolver, WorldsSignalKind, WorldsTerrain
from astro_mine.guard.wrap.shield import SignalResolver
from tests.guard.conftest import make_observation
from tests.guard.models_fixtures import FakeWorldProvider


def _resolver(**kwargs: object) -> WorldsFleetSignalResolver:
    return WorldsFleetSignalResolver(**kwargs)  # type: ignore[arg-type]


def test_is_a_signal_resolver() -> None:
    assert isinstance(_resolver(), SignalResolver)


def test_worlds_signals_resolve_at_observation_position() -> None:
    terrain = WorldsTerrain(FakeWorldProvider(illumination=IlluminationState.SHADOW))
    resolver = WorldsFleetSignalResolver(
        terrain=terrain,
        worlds_bindings={
            "charging_window_active": WorldsSignalKind.CHARGING_WINDOW,
            "surface_temp_k": WorldsSignalKind.SURFACE_TEMPERATURE_K,
        },
    )
    obs = make_observation("rover", position=(10.0, 0.0, 0.0))
    values = resolver.resolve(["charging_window_active", "surface_temp_k"], obs)
    assert values[0] == 0.0  # PSR / shadow ⇒ no charging window
    assert values[1] == 110.0  # surface temperature from the provider


def test_observation_fallback_for_unbound_keys() -> None:
    resolver = WorldsFleetSignalResolver()
    obs = make_observation("rover", signals={"chassis_temp_k": 250.0})
    values = resolver.resolve(["chassis_temp_k", "unknown_signal"], obs)
    assert values[0] == 250.0  # from the sensor reading
    assert math.isnan(values[1])  # unresolved ⇒ NaN (fail-safe)


def test_worlds_signal_without_position_is_nan() -> None:
    terrain = WorldsTerrain(FakeWorldProvider())
    resolver = WorldsFleetSignalResolver(
        terrain=terrain,
        worlds_bindings={"charging_window_active": WorldsSignalKind.CHARGING_WINDOW},
    )
    # No observation ⇒ no position ⇒ the Worlds signal cannot be sampled ⇒ NaN.
    assert math.isnan(resolver.resolve(["charging_window_active"], None)[0])


def test_worlds_signal_without_terrain_is_nan() -> None:
    resolver = WorldsFleetSignalResolver(
        worlds_bindings={"charging_window_active": WorldsSignalKind.CHARGING_WINDOW}
    )
    obs = make_observation("rover")
    assert math.isnan(resolver.resolve(["charging_window_active"], obs)[0])


def test_world_sampling_error_resolves_to_nan() -> None:
    # A provider fault (out-of-raster query) must not crash the control loop — it fails closed.
    terrain = WorldsTerrain(FakeWorldProvider(raise_on_sample=True))
    resolver = WorldsFleetSignalResolver(
        terrain=terrain,
        worlds_bindings={"slope_deg": WorldsSignalKind.SLOPE_DEG},
    )
    obs = make_observation("rover", position=(1.0, 2.0, 3.0))
    assert math.isnan(resolver.resolve(["slope_deg"], obs)[0])


def test_sadf_static_binding_resolves() -> None:
    resolver = WorldsFleetSignalResolver(sadf_bindings={"static_floor_w": 15.0})
    assert resolver.resolve(["static_floor_w"], None) == [15.0]


def test_resolve_preserves_order_and_length() -> None:
    resolver = WorldsFleetSignalResolver()
    obs = make_observation("rover", signals={"a": 1.0, "b": 2.0})
    out = resolver.resolve(["b", "a", "missing"], obs)
    assert out[0] == 2.0
    assert out[1] == 1.0
    assert math.isnan(out[2])
