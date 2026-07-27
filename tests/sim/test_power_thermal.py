"""RM-P0-SIM-07 — per-asset power/thermal evolution and the night-survival constraint.

Covers the acceptance criteria: across a lunar day/night cycle an asset's state-of-charge and
temperature evolve plausibly and a survival check is computable (and an asset survives ≥1 night
in reduced form). Also covers the coupled thermostat-heater drain, the loads-only (externally
powered) asset, the reference Core WorldProvider, and the Simulator integration.
"""

from __future__ import annotations

import pytest

from astro_mine.core.messages.model import ActionBatch
from astro_mine.core.sadf.enums import PowerSourceKind
from astro_mine.core.sadf.model import (
    ModeLoad,
    PowerBudget,
    PowerSource,
    PowerStorage,
    Range,
    ThermalBudget,
)
from astro_mine.core.units import MOON_BODY_FIXED, Epoch
from astro_mine.core.units.enums import TimeScale
from astro_mine.core.world import (
    Illumination,
    IlluminationState,
    RegolithParams,
    SurfacePoint,
    WorldProvider,
    check_world_provider,
)
from astro_mine.sim.power_thermal import (
    PowerThermalModel,
    PowerThermalState,
    ReferenceWorldProvider,
    default_initial_temperature,
)
from astro_mine.sim.runtime import AgentSpec, Scenario, Simulator


def _surface(
    *, flux: float = 1361.0, shadow: bool = False, ground_temp_k: float = 350.0
) -> SurfacePoint:
    return SurfacePoint(
        frame=MOON_BODY_FIXED,
        elevation_m=0.0,
        surface_normal=(0.0, 0.0, 1.0),
        gravity=(0.0, 0.0, -1.62),
        illumination=Illumination(
            state=IlluminationState.SHADOW if shadow else IlluminationState.LIT,
            solar_flux_w_m2=flux,
        ),
        temperature_k=ground_temp_k,
        regolith=RegolithParams(),
    )


_DAY = _surface(flux=1361.0, shadow=False, ground_temp_k=350.0)
_NIGHT = _surface(flux=0.0, shadow=True, ground_temp_k=100.0)


def _power(
    *,
    solar_w: float = 200.0,
    storage: bool = True,
    floor_w: float = 25.0,
    external_w: float = 0.0,
) -> PowerBudget:
    sources = [PowerSource(name="solar", kind=PowerSourceKind.SOLAR, nominal_power_w=solar_w)]
    if external_w:
        sources.append(
            PowerSource(name="ext", kind=PowerSourceKind.EXTERNAL, nominal_power_w=external_w)
        )
    return PowerBudget(
        sources=sources,
        storage=(
            [PowerStorage(name="b", capacity_j=1.0e6, max_charge_w=150.0, max_discharge_w=300.0)]
            if storage
            else []
        ),
        floor_w=floor_w,
        loads_by_mode=[ModeLoad(mode="idle", power_w=30.0), ModeLoad(mode="drive", power_w=120.0)],
    )


def _thermal() -> ThermalBudget:
    return ThermalBudget(
        operating_range_k=Range(min=120.0, max=330.0),
        survival_range_k=Range(min=95.0, max=360.0),
        dissipation_w=60.0,
        radiator_area_m2=0.4,
        heater_power_w=30.0,
        surface_coupling=True,
    )


# --- power balance ---------------------------------------------------------------


def test_battery_discharges_at_night_and_charges_in_sunlight() -> None:
    model = PowerThermalModel(_power())
    state = PowerThermalState(soc_j=5.0e5, temperature_k=None)

    night = model.step(state, 100.0, _NIGHT, "idle")
    assert night.soc_j == pytest.approx(5.0e5 - 30.0 * 100.0)  # idle load, no solar

    day = model.step(state, 100.0, _DAY, "idle")
    # generation 200 - load 30 = +170 W, clamped to the 150 W charge limit.
    assert day.soc_j == pytest.approx(5.0e5 + 150.0 * 100.0)


def test_solar_generation_is_gated_off_in_shadow() -> None:
    model = PowerThermalModel(_power())
    state = PowerThermalState(soc_j=5.0e5, temperature_k=None)
    shadowed = _surface(flux=1361.0, shadow=True)  # in shadow despite a flux value
    stepped = model.step(state, 100.0, shadowed, "idle")
    assert stepped.soc_j == pytest.approx(5.0e5 - 30.0 * 100.0)  # no solar in the PSR/eclipse


def test_a_higher_load_mode_drains_faster() -> None:
    model = PowerThermalModel(_power())
    state = PowerThermalState(soc_j=5.0e5, temperature_k=None)
    idle = model.step(state, 100.0, _NIGHT, "idle")
    drive = model.step(state, 100.0, _NIGHT, "drive")
    assert drive.soc_j < idle.soc_j  # drive draws 120 W vs idle 30 W


def test_an_unknown_mode_falls_back_to_the_housekeeping_floor() -> None:
    model = PowerThermalModel(_power(floor_w=40.0))
    state = PowerThermalState(soc_j=5.0e5, temperature_k=None)
    stepped = model.step(state, 100.0, _NIGHT, "no-such-mode")
    assert stepped.soc_j == pytest.approx(5.0e5 - 40.0 * 100.0)  # floor_w, not zero


def test_a_loads_only_asset_tracks_no_battery() -> None:
    model = PowerThermalModel(_power(storage=False, external_w=5000.0))
    assert model.has_storage is False
    state = PowerThermalState(soc_j=999.0, temperature_k=None)
    assert model.step(state, 100.0, _DAY, "drive").soc_j == 999.0  # externally supplied


def test_unbounded_storage_limits_do_not_clamp_the_net_power() -> None:
    power = PowerBudget(
        sources=[PowerSource(name="solar", kind=PowerSourceKind.SOLAR, nominal_power_w=1000.0)],
        storage=[PowerStorage(name="b", capacity_j=1.0e9)],  # no charge/discharge limits declared
        floor_w=0.0,
        loads_by_mode=[ModeLoad(mode="idle", power_w=0.0)],
    )
    stepped = PowerThermalModel(power).step(
        PowerThermalState(soc_j=0.0, temperature_k=None), 100.0, _DAY, "idle"
    )
    assert stepped.soc_j == pytest.approx(1000.0 * 100.0)  # full generation, no charge-limit clamp


# --- the coupled thermostat heater -----------------------------------------------


def test_the_thermostat_heater_engages_below_the_operating_floor_and_costs_battery() -> None:
    model = PowerThermalModel(_power(), _thermal())
    cold = PowerThermalState(soc_j=1.0e6, temperature_k=110.0)  # below the 120 K operating floor
    warm = PowerThermalState(soc_j=1.0e6, temperature_k=200.0)  # above it — no heating

    cold_step = model.step(cold, 100.0, _NIGHT, "idle")
    warm_step = model.step(warm, 100.0, _NIGHT, "idle")
    # heating adds the 30 W heater to the 30 W idle load, so the cold asset drains faster.
    assert cold_step.soc_j == pytest.approx(1.0e6 - 60.0 * 100.0)
    assert warm_step.soc_j == pytest.approx(1.0e6 - 30.0 * 100.0)
    assert cold_step.soc_j < warm_step.soc_j


# --- thermal balance + survival --------------------------------------------------


def test_a_hot_asset_cools_toward_the_cold_night_surface() -> None:
    model = PowerThermalModel(_power(), _thermal())
    hot = PowerThermalState(soc_j=1.0e6, temperature_k=300.0)
    assert model.step(hot, 1000.0, _NIGHT, "idle").temperature_k < 300.0  # radiates + couples down


def test_an_asset_without_a_thermal_budget_has_no_temperature() -> None:
    model = PowerThermalModel(_power())
    assert model.step(PowerThermalState(1.0e6, None), 100.0, _DAY, "idle").temperature_k is None


def test_survival_predicate_fails_on_depletion_or_out_of_range_temperature() -> None:
    model = PowerThermalModel(_power(), _thermal())
    assert model.survived(PowerThermalState(soc_j=1.0e6, temperature_k=200.0)) is True
    assert model.survived(PowerThermalState(soc_j=0.0, temperature_k=200.0)) is False  # depleted
    assert model.survived(PowerThermalState(soc_j=1.0e6, temperature_k=90.0)) is False  # < 95 K


def test_default_initial_temperature_is_the_operating_midpoint() -> None:
    assert default_initial_temperature(None) is None
    assert default_initial_temperature(_thermal()) == pytest.approx(225.0)  # (120 + 330) / 2


def test_model_rejects_a_non_positive_heat_capacity() -> None:
    with pytest.raises(ValueError, match="heat_capacity_j_per_k must be > 0"):
        PowerThermalModel(_power(), _thermal(), heat_capacity_j_per_k=0.0)


# --- day/night cycle (acceptance criterion 1) ------------------------------------


def _epoch(seconds: float) -> Epoch:
    return Epoch(tdb_seconds=seconds, scale=TimeScale.TDB)


def test_state_of_charge_and_temperature_track_a_full_day_night_cycle() -> None:
    # Step the same model through the reference world's noon and midnight: the battery charges
    # in full sun and drains in the dark, and the asset runs warmer by day than by night.
    provider = ReferenceWorldProvider(period_s=2400.0, day_temp_k=350.0, night_temp_k=100.0)
    model = PowerThermalModel(_power(), _thermal())
    start = PowerThermalState(soc_j=5.0e5, temperature_k=225.0)

    noon = model.step(start, 100.0, provider.sample((0.0, 0.0, 0.0), epoch=_epoch(1200.0)), "idle")
    midnight = model.step(start, 100.0, provider.sample((0.0, 0.0, 0.0), epoch=_epoch(0.0)), "idle")

    assert noon.soc_j > start.soc_j  # charging in full sun
    assert midnight.soc_j < start.soc_j  # draining in the dark
    assert noon.temperature_k is not None and midnight.temperature_k is not None
    assert noon.temperature_k > midnight.temperature_k  # warmer surface coupling by day


def test_a_well_provisioned_asset_survives_a_sustained_night() -> None:
    # The M0.2 night-survival exit: a full battery with the thermostat heater holds charge and
    # temperature through a sustained night — survived() stays True the whole way.
    provider = ReferenceWorldProvider(period_s=2400.0, night_temp_k=100.0)
    model = PowerThermalModel(_power(), _thermal())
    midnight = provider.sample((0.0, 0.0, 0.0), epoch=_epoch(0.0))
    state = PowerThermalState(soc_j=1.0e6, temperature_k=200.0)
    for _ in range(20):
        state = model.step(state, 100.0, midnight, "idle")
        assert model.survived(state)  # battery not depleted, temperature within survival range


# --- the reference world provider satisfies the Core contract --------------------


def test_reference_world_provider_satisfies_the_core_contract() -> None:
    provider = ReferenceWorldProvider()
    assert isinstance(provider, WorldProvider)
    check_world_provider(provider)  # the consumer-driven Core WorldProvider contract test
    assert provider.frame == MOON_BODY_FIXED


def test_reference_world_provider_is_sunlit_at_noon_and_dark_at_night() -> None:
    provider = ReferenceWorldProvider(period_s=1000.0, day_temp_k=350.0, night_temp_k=100.0)
    noon = provider.sample((0.0, 0.0, 0.0), epoch=_epoch(500.0))  # phase 0.5
    assert noon.illumination.state is IlluminationState.LIT
    assert noon.illumination.solar_flux_w_m2 == pytest.approx(1361.0)
    assert noon.temperature_k == pytest.approx(350.0)

    midnight = provider.sample((0.0, 0.0, 0.0), epoch=_epoch(0.0))  # phase 0.0
    assert midnight.illumination.state is IlluminationState.SHADOW
    assert midnight.illumination.solar_flux_w_m2 == 0.0
    assert midnight.temperature_k == pytest.approx(100.0)
    # the reference world is flat — no occluding terrain, full line-of-sight.
    assert provider.ray_intersect((0.0, 0.0, 10.0), (0.0, 0.0, -1.0)) is None
    assert provider.line_of_sight((0.0, 0.0, 1.0), (100.0, 0.0, 1.0)) is True


def test_reference_world_provider_rejects_a_non_positive_period() -> None:
    with pytest.raises(ValueError, match="period_s must be > 0"):
        ReferenceWorldProvider(period_s=0.0)


# --- Simulator integration -------------------------------------------------------


def _budgeted_rover(*, soc_j: float = 1.0e6) -> AgentSpec:
    return AgentSpec(
        agent_id="rover", battery_soc_j=soc_j, mode="idle", power=_power(), thermal=_thermal()
    )


def test_simulator_evolves_a_budgeted_agents_soc_and_temperature() -> None:
    scenario = Scenario(name="budgeted", agents=(_budgeted_rover(),), dt_s=3600.0, horizon_steps=2)
    sim = Simulator(scenario)
    initial = sim.reset().observations["rover"].self_state
    assert initial.temperature_k == pytest.approx(225.0)  # operating-midpoint default
    assert initial.battery_soc_j == pytest.approx(1.0e6)

    stepped = sim.step(ActionBatch()).observations["rover"].self_state
    assert stepped.temperature_k is not None and stepped.temperature_k != initial.temperature_k
    # power/thermal owns the battery — the drop is the idle/heater load, not the engine's 1 W draw.
    assert stepped.battery_soc_j < initial.battery_soc_j
    assert initial.battery_soc_j - stepped.battery_soc_j > 1.0


def test_simulator_mixes_budgeted_and_unbudgeted_agents() -> None:
    scenario = Scenario(
        name="mixed",
        agents=(
            _budgeted_rover(),
            AgentSpec(agent_id="bare", battery_soc_j=1.0e6),  # un-budgeted: kinematic-engine draw
        ),
        dt_s=3600.0,
        horizon_steps=1,
    )
    sim = Simulator(scenario)
    sim.reset()
    observations = sim.step(ActionBatch()).observations
    # the budgeted rover evolves via power/thermal; the bare agent keeps the engine's draw.
    assert observations["rover"].self_state.temperature_k is not None
    assert observations["bare"].self_state.temperature_k is None
    assert observations["bare"].self_state.battery_soc_j == pytest.approx(1.0e6 - 3600.0)


def test_an_unbudgeted_agent_keeps_the_engine_draw_and_has_no_temperature() -> None:
    scenario = Scenario(name="bare", agents=(AgentSpec(agent_id="a", battery_soc_j=100.0),))
    sim = Simulator(scenario)
    sim.reset()
    state = sim.step(ActionBatch()).observations["a"].self_state
    assert state.temperature_k is None  # no thermal budget
    assert state.battery_soc_j == pytest.approx(99.0)  # the kinematic engine's 1 W placeholder draw


def test_a_loads_only_plant_evolves_temperature_but_freezes_its_battery() -> None:
    plant = AgentSpec(
        agent_id="plant",
        battery_soc_j=500.0,
        mode="idle",
        power=_power(storage=False, external_w=5000.0),
        thermal=_thermal(),
    )
    sim = Simulator(Scenario(name="isru", agents=(plant,), dt_s=3600.0, horizon_steps=2))
    sim.reset()
    state = sim.step(ActionBatch()).observations["plant"].self_state
    assert state.battery_soc_j == pytest.approx(500.0)  # external supply, not battery-bound
    assert state.temperature_k is not None  # thermal still evolves


def test_battery_floor_termination_uses_the_power_thermal_soc() -> None:
    # A nearly-empty battery at night drains below its floor via power/thermal, not the engine.
    rover = AgentSpec(
        agent_id="rover",
        battery_soc_j=2000.0,
        battery_floor_j=0.0,
        mode="drive",
        power=_power(),
        thermal=_thermal(),
    )
    sim = Simulator(Scenario(name="dying", agents=(rover,), dt_s=3600.0, horizon_steps=3))
    sim.reset()
    # drive load 120 W * 3600 s = 432 kJ/step >> 2 kJ start, with no solar at the J2000 night.
    result = sim.step(ActionBatch())
    assert result.terminations["rover"] is True
    assert result.observations["rover"].self_state.battery_soc_j == 0.0  # clamped, not negative
