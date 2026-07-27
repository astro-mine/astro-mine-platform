"""RM-P1-SIM-02 — ISRU extraction/storage: the reduced-order process model + its episode wiring.

Covers the acceptance criteria: an anchor run reports monotonic stored-water (kg) + its energy
cost in the (MCAP-equivalent) trace so Bench can score ``water_mass`` / ``energy_per_kg``
non-degenerately, and two clean runs are byte-identical (determinism). Also unit-tests the
reduced-order model (mode gating, capacity, abundance scaling, monotonicity, validators).
"""

from __future__ import annotations

import pytest

from astro_mine.core.messages.enums import (
    ActionKind,
    ExcavationPattern,
    ExcavationTool,
    TaskKind,
)
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    ExcavateTask,
    ModeCommand,
    TaskDirective,
    Vec3,
    Volume,
)
from astro_mine.sim.isru import DEFAULT_EXTRACTION_MODES, IsruModel, IsruState
from astro_mine.sim.logistics import Material
from astro_mine.sim.runtime import Scenario, run_episode
from astro_mine.sim.sensors import ReferenceResourceField

# --- IsruModel unit ---------------------------------------------------------------


def _model(**kwargs: float | None) -> IsruModel:
    params: dict[str, float | None] = {
        "extraction_rate_kg_s": 2.0,
        "specific_energy_j_per_kg": 1000.0,
    }
    params.update(kwargs)
    return IsruModel(**params)  # type: ignore[arg-type]


def test_initial_state_is_empty() -> None:
    assert IsruModel(extraction_rate_kg_s=1.0, specific_energy_j_per_kg=1.0).initial_state() == (
        IsruState(0.0, 0.0)
    )


def _stock(mass_kg: float = 100.0, grade: float = 1.0) -> Material:
    """Feedstock delivered to the plant, at a grade — pure water at the default."""
    return Material(mass_kg=mass_kg, water_fraction=grade)


def test_extraction_without_feedstock_produces_nothing() -> None:
    """The whole point of #64: a mode string is no longer sufficient.

    Before this, `step` produced water from `mode` alone, so a policy that flipped the plant to
    `extract` manufactured a confident `water_mass` no excavation and no haulage had earned.
    """
    model = _model()
    state = IsruState(5.0, 5000.0)
    after, remaining = model.step(state, 1.0, "excavate", Material())
    assert after == state
    assert remaining == Material()


def test_extraction_accumulates_water_and_energy_from_delivered_feedstock() -> None:
    model = _model()
    s1, left = model.step(model.initial_state(), 1.0, "excavate", _stock())
    assert s1.stored_water_kg == pytest.approx(2.0)
    assert s1.energy_used_j == pytest.approx(2000.0)
    assert left.mass_kg == pytest.approx(98.0)  # the regolith it consumed is gone
    s2, left2 = model.step(s1, 1.0, "excavate", left)
    assert s2.stored_water_kg == pytest.approx(4.0)  # monotonic
    assert s2.energy_used_j == pytest.approx(4000.0)
    assert left2.mass_kg == pytest.approx(96.0)


def test_the_regolith_consumed_matches_the_water_it_yielded() -> None:
    # Half-grade feedstock: 4 kg of regolith must be processed for 2 kg of water. Mass is not
    # created — the chain has to balance, or `water_mass` is a number with no material behind it.
    model = _model()
    before = _stock(mass_kg=100.0, grade=0.5)
    after, left = model.step(model.initial_state(), 1.0, "excavate", before)
    assert after.stored_water_kg == pytest.approx(2.0)
    assert before.mass_kg - left.mass_kg == pytest.approx(4.0)
    assert left.water_fraction == pytest.approx(0.5)  # the grade of what remains is unchanged


def test_a_barren_feedstock_yields_nothing() -> None:
    model = _model()
    state, left = model.step(model.initial_state(), 1.0, "excavate", _stock(grade=0.0))
    assert state == IsruState(0.0, 0.0)
    assert left.mass_kg == pytest.approx(100.0)  # and none of it was consumed


def test_extraction_is_bounded_by_the_water_actually_delivered() -> None:
    # 1 kg of regolith at 50% grade carries 0.5 kg of water; the 2 kg/s rate cannot exceed it.
    model = _model()
    state, left = model.step(model.initial_state(), 1.0, "excavate", _stock(1.0, 0.5))
    assert state.stored_water_kg == pytest.approx(0.5)
    assert left.mass_kg == pytest.approx(0.0)


def test_non_extraction_mode_is_a_no_op() -> None:
    model = _model()
    state = IsruState(5.0, 5000.0)
    assert model.step(state, 1.0, "idle", _stock())[0] == state
    assert model.step(state, 1.0, None, _stock())[0] == state


def test_non_positive_dt_is_a_no_op() -> None:
    model = _model()
    state = IsruState(5.0, 5000.0)
    assert model.step(state, 0.0, "excavate", _stock())[0] == state


def test_capacity_caps_the_tank() -> None:
    model = _model(capacity_kg=3.0)
    s1, left = model.step(model.initial_state(), 1.0, "excavate", _stock())  # +2 -> 2
    s2, left = model.step(s1, 1.0, "excavate", left)  # would be +2 -> capped at 3
    assert s2.stored_water_kg == pytest.approx(3.0)
    assert s2.energy_used_j == pytest.approx(3000.0)  # energy only for the 1 kg actually stored
    s3, _ = model.step(s2, 1.0, "excavate", left)  # full tank -> unchanged
    assert s3 == s2


def test_zero_rate_extracts_nothing() -> None:
    model = _model(extraction_rate_kg_s=0.0)
    assert model.step(model.initial_state(), 1.0, "excavate", _stock())[0] == IsruState(0.0, 0.0)


def test_material_rejects_an_impossible_grade() -> None:
    # The clamp that used to live in `step` belongs to the material now: a mass fraction outside
    # [0, 1] is not a thing that can be dug, so it fails at construction rather than silently.
    with pytest.raises(ValueError, match="water fraction"):
        Material(mass_kg=1.0, water_fraction=1.5)
    with pytest.raises(ValueError, match="water fraction"):
        Material(mass_kg=1.0, water_fraction=-0.1)
    with pytest.raises(ValueError, match="mass"):
        Material(mass_kg=-1.0)


def test_default_extraction_modes() -> None:
    assert "excavate" in DEFAULT_EXTRACTION_MODES and "idle" not in DEFAULT_EXTRACTION_MODES


@pytest.mark.parametrize(
    "kwargs",
    [
        {"extraction_rate_kg_s": -1.0},
        {"specific_energy_j_per_kg": -1.0},
        {"capacity_kg": -1.0},
    ],
)
def test_negative_parameters_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match=">= 0"):
        _model(**kwargs)


# --- episode integration ----------------------------------------------------------


def _isru_scenario(mode: str = "excavate", horizon: int = 4, **isru: object) -> Scenario:
    spec: dict[str, object] = {"extraction_rate_kg_s": 2.0, "specific_energy_j_per_kg": 1000.0}
    spec.update(isru)
    return Scenario.from_mapping(
        {
            "name": "isru-anchor",
            "seed": 3,
            "dt_s": 1.0,
            "horizon_steps": horizon,
            "agents": [
                {
                    "agent_id": "plant",
                    "mode": mode,
                    "battery_soc_j": 1.0e9,  # survive the horizon so the series is non-degenerate
                    "isru": spec,
                    "sensors": [{"name": "tank", "kind": "resource_storage", "frame": "body"}],
                }
            ],
        }
    )


def _tank_series(trace: object) -> list[tuple[float, float]]:
    """The (stored_kg, energy_j) reading of the ``tank`` sensor across every frame."""
    series: list[tuple[float, float]] = []
    for frame in trace.frames:  # type: ignore[attr-defined]
        for obs in frame["observations"].values():
            for reading in obs["sensors"]:
                if reading["sensor"] == "tank" and reading["valid"]:
                    series.append((reading["values"][0], reading["values"][1]))
    return series


def _chain_scenario(horizon: int = 6, *, separation_m: float = 0.0) -> Scenario:
    """A digger and a plant — the value chain in miniature.

    ``separation_m`` puts the two apart. Inside the transfer radius the plant is fed; outside it,
    nothing arrives however hard the excavator digs, which is the property that makes `water_mass`
    mean something.
    """
    return Scenario.from_mapping(
        {
            "name": "isru-chain",
            "seed": 3,
            "dt_s": 1.0,
            "horizon_steps": horizon,
            "agents": [
                {
                    "agent_id": "digger",
                    "battery_soc_j": 1.0e9,
                    "initial_position_m": [separation_m, 0.0, 0.0],
                    "dynamics": {"kind": "granular", "max_dig_rate_m3_s": 1.0e-3},
                },
                {
                    "agent_id": "plant",
                    "mode": "extract",
                    "battery_soc_j": 1.0e9,
                    "isru": {"extraction_rate_kg_s": 2.0, "specific_energy_j_per_kg": 1000.0},
                    "sensors": [{"name": "tank", "kind": "resource_storage", "frame": "body"}],
                },
            ],
        }
    )


class _DigAndExtract:
    """Commands the digger to excavate and holds the plant in an extraction mode."""

    def decide(self, observations: object, context: object) -> ActionBatch:
        return ActionBatch(
            actions=[
                Action(
                    agent_id="digger",
                    kind=ActionKind.TASK,
                    task=TaskDirective(
                        task_kind=TaskKind.EXCAVATE,
                        excavate=ExcavateTask(
                            region=Volume(
                                frame="MOON_ME",
                                center_m=Vec3(x=0.0, y=0.0, z=0.0),
                                dimensions_m=Vec3(x=1.0, y=1.0, z=1.0),
                            ),
                            tool=ExcavationTool.BUCKET,
                            pattern=ExcavationPattern.TRENCH,
                            target_volume_m3=None,
                        ),
                    ),
                ),
                Action(agent_id="plant", kind=ActionKind.MODE, mode=ModeCommand(mode="extract")),
            ]
        )


def test_a_plant_fed_by_a_digger_accumulates_water_and_energy() -> None:
    # The value chain, end to end: regolith is dug, delivered, and converted. Every kilogram in
    # the tank is traceable to material that came out of the ground.
    field = ReferenceResourceField()
    trace = run_episode(_chain_scenario(), seed=3, policy=_DigAndExtract(), resource_field=field)
    series = _tank_series(trace)
    water = [w for w, _ in series]
    energy = [e for _, e in series]
    assert water == sorted(water) and energy == sorted(energy)  # both monotonic non-decreasing
    assert water[-1] > 0.0 and energy[-1] > 0.0  # non-degenerate
    assert energy[-1] / water[-1] == pytest.approx(1000.0)  # specific energy, as declared


def test_a_plant_out_of_reach_of_the_digger_gets_nothing() -> None:
    """The gate that makes the number defensible.

    Same digger, same extraction mode, same field — only the distance changes. If water still
    accumulated here it would mean the plant was mining the ground under its own footprint, which
    is exactly the defect #64 exists to remove.
    """
    trace = run_episode(
        _chain_scenario(separation_m=10_000.0),
        seed=3,
        policy=_DigAndExtract(),
        resource_field=ReferenceResourceField(),
    )
    assert {w for w, _ in _tank_series(trace)} == {0.0}


def test_an_extraction_mode_alone_manufactures_nothing() -> None:
    # No digger at all: the plant is in `extract` for the whole run and stores nothing. This is
    # the regression that matters — it passed for the wrong reason before #64.
    trace = run_episode(_isru_scenario(), seed=3, resource_field=ReferenceResourceField())
    assert {w for w, _ in _tank_series(trace)} == {0.0}


def test_isru_run_is_deterministic() -> None:
    a = run_episode(_isru_scenario(), seed=3)
    b = run_episode(_isru_scenario(), seed=3)
    assert a.content_hash == b.content_hash


def test_the_grade_comes_from_where_it_was_dug() -> None:
    """Extraction scales by the field at the *excavator*, not under the plant.

    The reference field peaks at 0.1 at the origin, so a digger sitting there yields material at
    that grade and the plant converts against it. Previously the abundance was sampled at the
    plant's own pose, which made `water_mass` a siting constant — it moved with the terrain rather
    than with anything the swarm did.
    """
    trace = run_episode(
        _chain_scenario(), seed=3, policy=_DigAndExtract(), resource_field=ReferenceResourceField()
    )
    water = [w for w, _ in _tank_series(trace)]
    # Yield is bounded by the water content of the delivered regolith (grade 0.1), never by the
    # 2.0 kg/s nameplate rate, which a plant mining its own footprint would have hit.
    assert 0.0 < water[-1] < 2.0


def test_idle_asset_stores_nothing() -> None:
    series = _tank_series(run_episode(_isru_scenario(mode="idle"), seed=3))
    assert {w for w, _ in series} == {0.0}


def test_non_isru_agent_is_unaffected() -> None:
    scenario = Scenario.from_mapping(
        {"name": "plain", "seed": 1, "horizon_steps": 2, "agents": [{"agent_id": "rover"}]}
    )
    # No isru block, no resource_storage sensor: the run just works (byte-identical to before).
    assert run_episode(scenario, seed=1).content_hash == run_episode(scenario, seed=1).content_hash


def test_mixed_fleet_only_the_isru_asset_carries_the_channel() -> None:
    # The anchor shape: an ISRU plant alongside a plain rover — only the plant has ISRU state at
    # all, and the rover is untouched by the ISRU evolution.
    scenario = Scenario.from_mapping(
        {
            "name": "mixed",
            "seed": 3,
            "horizon_steps": 2,
            "agents": [
                {
                    "agent_id": "plant",
                    "mode": "excavate",
                    "battery_soc_j": 1.0e9,
                    "isru": {"extraction_rate_kg_s": 2.0, "specific_energy_j_per_kg": 1000.0},
                    "sensors": [{"name": "tank", "kind": "resource_storage", "frame": "body"}],
                },
                {"agent_id": "rover", "battery_soc_j": 1.0e9},
            ],
        }
    )
    final = run_episode(scenario, seed=3).frames[-1]["observations"]
    # The plant carries the ISRU channel and the rover does not — that is what this test is for.
    # The tank reads zero because nothing was delivered to it: an extraction mode is no longer
    # sufficient (#64), so a fleet with no digger produces no water however it is configured.
    assert final["plant"]["sensors"][0]["values"][0] == 0.0
    assert len(final["plant"]["sensors"][0]["values"]) == 2  # mass and energy, the gauge contract
    assert final["rover"]["sensors"] == []  # the rover has no ISRU channel


# --- gauge dispatch: a declared resource target must not turn the tank into a field probe (#61) ---


def test_a_storage_gauge_with_a_declared_resource_target_still_reports_the_tank() -> None:
    """The defect this pins: `render_sensor` routed *any* sensor declaring a `resource` target to
    the field-sample model, so a plant's tank rendered a noisy draw of the ice field tagged
    `unit="kg"` — and Bench's `water_mass`, which filters on exactly `(species, "kg")`, scored a
    mass fraction as kilograms (astro-mine-sim#61).

    A `resource_storage` gauge legitimately declares a target: it says *what the tank holds*, not
    *what to sample*. The tell is the shape — the gauge emits `[stored_kg, energy_j]`, the field
    sampler emits one value.
    """
    field = ReferenceResourceField(species="water", peak=0.5, length_scale_m=1e6)
    scenario = Scenario.model_validate(
        {
            "name": "gauge-dispatch",
            "dt_s": 1.0,
            "horizon_steps": 3,
            "agents": [
                {
                    "agent_id": "plant",
                    "initial_position_m": [0.0, 0.0, 0.0],
                    "battery_soc_j": 1.0e9,
                    "isru": {"extraction_rate_kg_s": 1.0, "nominal_abundance": 1.0},
                    "sensors": [
                        {
                            "name": "tank",
                            "kind": "resource_storage",
                            "frame": "body",
                            "resource": {"species": "water", "si_unit": "kg"},
                        }
                    ],
                }
            ],
        }
    )

    frames = run_episode(scenario, seed=5, resource_field=field).frames
    reading = frames[-1]["observations"]["plant"]["sensors"][0]

    assert len(reading["values"]) == 2, "the gauge was bypassed by the field sampler"
    assert reading["unit"] == "kg"
    assert reading["resource_species"] == "water"
    # The plant never entered an extraction mode, so the tank is empty — and must say 0.0 rather
    # than the 0.5 the field would have reported at this position.
    assert reading["values"][0] == 0.0


def test_the_gauge_reports_its_declared_species_and_unit() -> None:
    """Species/unit come off the SADF `ResourceTarget`, not from literals baked into the model."""
    scenario = Scenario.model_validate(
        {
            "name": "gauge-declares",
            "dt_s": 1.0,
            "horizon_steps": 1,
            "agents": [
                {
                    "agent_id": "plant",
                    "initial_position_m": [0.0, 0.0, 0.0],
                    "battery_soc_j": 1.0e9,
                    "isru": {"extraction_rate_kg_s": 1.0},
                    "sensors": [
                        {
                            "name": "tank",
                            "kind": "resource_storage",
                            "frame": "body",
                            "resource": {"species": "oxygen", "si_unit": "kg"},
                        }
                    ],
                }
            ],
        }
    )

    reading = run_episode(scenario, seed=1).frames[-1]["observations"]["plant"]["sensors"][0]
    assert reading["resource_species"] == "oxygen"


def test_a_prospecting_sensor_with_a_target_still_samples_the_field() -> None:
    """The rule is narrow: a kind with a *registered resource model*, or with no self-state model,
    keeps sampling the field. Only gauge kinds — self-state-only — are rerouted."""
    field = ReferenceResourceField(
        species="water_equivalent_hydrogen", peak=0.4, length_scale_m=1e6
    )
    scenario = Scenario.model_validate(
        {
            "name": "prospect-sampling",
            "dt_s": 1.0,
            "horizon_steps": 1,
            "agents": [
                {
                    "agent_id": "rover",
                    "initial_position_m": [0.0, 0.0, 0.0],
                    "battery_soc_j": 1.0e9,
                    "sensors": [
                        {
                            "name": "neutron",
                            "kind": "neutron_spectrometer",
                            "frame": "body",
                            "resource": {
                                "species": "water_equivalent_hydrogen",
                                "si_unit": "mass_fraction",
                            },
                        }
                    ],
                }
            ],
        }
    )

    reading = run_episode(scenario, seed=2, resource_field=field).frames[-1]["observations"][
        "rover"
    ]["sensors"][0]
    assert len(reading["values"]) == 1  # a field sample, not a gauge
    assert reading["values"][0] > 0.0
