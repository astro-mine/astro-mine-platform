"""RM-P0-SIM-01 — the deterministic stepping core.

Covers acceptance criterion 1 (byte-identical traces under a fixed seed) and criterion 2
(the Core Environment API contract via ``check_environment``), plus the scenario loader,
the SPICE-time clock, and the seeded RNG units. Criterion 3 (Core-version compatibility)
lives in ``test_core_compat.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from astro_mine.core.env import check_environment
from astro_mine.core.messages.model import ActionBatch, Observation
from astro_mine.core.units import INERTIAL_J2000, J2000_EPOCH, MOON_BODY_FIXED, Epoch
from astro_mine.core.units.enums import TimeScale
from astro_mine.sim.runtime import (
    CORE_INTERFACES,
    AgentSpec,
    GranularDynamics,
    KinematicDynamics,
    ManipulationDynamics,
    MobilityDynamics,
    OrbitalDynamics,
    RngStreams,
    Scenario,
    SimClock,
    Simulator,
    load_scenario,
    run_episode,
)

DATA = Path(__file__).parent / "data"


def _scenario(**overrides: object) -> Scenario:
    base: dict[str, object] = {
        "name": "unit",
        "agents": (
            AgentSpec(agent_id="a", velocity_mps=(1.0, 0.0, 0.0), battery_soc_j=100.0),
            AgentSpec(agent_id="b", velocity_mps=(0.0, 2.0, 0.0), battery_soc_j=50.0),
        ),
        "seed": 7,
        "dt_s": 0.5,
        "horizon_steps": 3,
    }
    base.update(overrides)
    return Scenario(**base)  # type: ignore[arg-type]


# --- scenario loader -------------------------------------------------------------


def test_scenario_loads_from_json_file_with_defaults() -> None:
    scenario = load_scenario(DATA / "scenario.json")
    assert scenario.name == "two-rover-smoke"
    assert tuple(a.agent_id for a in scenario.agents) == ("rover-a", "rover-b")
    assert scenario.dt_s == 0.5
    assert scenario.horizon_steps == 4
    # frame/epoch are not in the document — the lunar-anchor defaults apply.
    assert scenario.frame == MOON_BODY_FIXED
    assert scenario.start_epoch == J2000_EPOCH


def test_scenario_rejects_empty_agents() -> None:
    with pytest.raises(ValidationError, match="at least one agent"):
        Scenario(name="empty", agents=())


def test_scenario_rejects_duplicate_agent_ids() -> None:
    with pytest.raises(ValidationError, match="duplicate agent ids"):
        Scenario(name="dup", agents=(AgentSpec(agent_id="x"), AgentSpec(agent_id="x")))


def test_scenario_rejects_unknown_field() -> None:
    with pytest.raises(ValidationError):
        Scenario.from_mapping({"name": "x", "agents": [{"agent_id": "a"}], "bogus": 1})


# --- per-agent dynamics discriminator (RM-P0-SIM-03) -----------------------------


def test_agent_dynamics_defaults_to_kinematic() -> None:
    # backward compatible: a bare agent (no dynamics block) is the reference engine.
    assert isinstance(AgentSpec(agent_id="a").dynamics, KinematicDynamics)


def test_dynamics_discriminator_selects_each_variant() -> None:
    scenario = Scenario.from_mapping(
        {
            "name": "fleet",
            "agents": [
                {"agent_id": "relay", "dynamics": {"kind": "orbital"}},
                {
                    "agent_id": "rover",
                    "dynamics": {
                        "kind": "mobility",
                        "mass_kg": 200.0,
                        "max_speed_mps": 0.5,
                        "max_traction_n": 300.0,
                    },
                },
                {
                    "agent_id": "arm",
                    "dynamics": {
                        "kind": "manipulation",
                        "joints": [{"name": "j0", "joint_type": "revolute", "rate_limit": 0.5}],
                    },
                },
                {
                    "agent_id": "digger",
                    "dynamics": {"kind": "granular", "max_dig_rate_m3_s": 0.01},
                },
            ],
        }
    )
    kinds = {a.agent_id: type(a.dynamics) for a in scenario.agents}
    assert kinds == {
        "relay": OrbitalDynamics,
        "rover": MobilityDynamics,
        "arm": ManipulationDynamics,
        "digger": GranularDynamics,
    }


def test_mobility_dynamics_requires_physical_params() -> None:
    with pytest.raises(ValidationError):
        Scenario.from_mapping(
            {"name": "x", "agents": [{"agent_id": "r", "dynamics": {"kind": "mobility"}}]}
        )


def test_manipulation_requires_at_least_one_joint() -> None:
    with pytest.raises(ValidationError, match="at least one joint"):
        ManipulationDynamics(joints=())


def test_unknown_dynamics_kind_fails_loudly() -> None:
    with pytest.raises(ValidationError):
        Scenario.from_mapping(
            {"name": "x", "agents": [{"agent_id": "r", "dynamics": {"kind": "warp_drive"}}]}
        )


# --- clock -----------------------------------------------------------------------


def test_clock_advances_and_renders_absolute_epoch() -> None:
    clock = SimClock(start_epoch=J2000_EPOCH, dt_s=0.5)
    assert clock.tick == 0
    assert clock.now_epoch() == Epoch(tdb_seconds=0.0, scale=TimeScale.TDB)

    stepped = clock.advanced()
    assert stepped.tick == 1
    assert stepped.sim_time_s == 0.5
    assert stepped.now_epoch().tdb_seconds == 0.5

    variable = stepped.advanced(dt_s=2.0)
    assert variable.tick == 2
    assert variable.sim_time_s == 2.5


def test_clock_rejects_nonpositive_dt() -> None:
    with pytest.raises(ValueError, match="dt_s must be > 0"):
        SimClock(start_epoch=J2000_EPOCH, dt_s=0.0)


# --- rng -------------------------------------------------------------------------


def test_rng_streams_are_reproducible_and_independent() -> None:
    a1 = [RngStreams(7).stream("agent").random() for _ in range(3)]
    a2 = [RngStreams(7).stream("agent").random() for _ in range(3)]
    assert a1 == a2  # same root seed + name -> identical sequence

    streams = RngStreams(7)
    assert streams.root_seed == 7
    assert streams.stream("agent") is streams.stream("agent")  # cached per name
    assert streams.stream("a").random() != streams.stream("b").random()  # independent


# --- episode loop / Environment contract -----------------------------------------


def test_agents_are_empty_until_reset() -> None:
    sim = Simulator(_scenario())
    assert sim.possible_agents == ("a", "b")
    assert sim.agents == ()


def test_reset_is_deterministic_and_yields_core_observations() -> None:
    sim = Simulator(_scenario())
    first = sim.reset(seed=1)
    second = sim.reset(seed=1)
    assert sim.agents == ("a", "b")
    assert all(isinstance(obs, Observation) for obs in first.observations.values())
    dump = {a: o.model_dump(mode="json") for a, o in first.observations.items()}
    again = {a: o.model_dump(mode="json") for a, o in second.observations.items()}
    assert dump == again


def test_step_advances_clock_battery_and_truncates_at_horizon() -> None:
    scenario = _scenario(horizon_steps=3)
    sim = Simulator(scenario)
    sim.reset()  # no explicit seed -> falls back to the scenario seed

    results = [sim.step(ActionBatch()) for _ in range(3)]
    sim_times = [r.sim_time_s for r in results]
    assert sim_times == [0.5, 1.0, 1.5]
    assert all(r.dt_s == 0.5 for r in results)
    # battery decays linearly; the agent moves along +x.
    last = results[-1].observations["a"].self_state
    assert last.battery_soc_j is not None and last.battery_soc_j < 100.0
    assert last.pose.translation_m.x > 0.0
    # only the horizon step truncates.
    assert [r.truncations["a"] for r in results] == [False, False, True]


def test_environment_conformance() -> None:
    # Acceptance criterion 2: honors the Core Environment API contract.
    check_environment(Simulator(_scenario()))


# --- reproducibility: byte-identical traces (acceptance criterion 1) -------------


def test_same_seed_traces_are_byte_identical() -> None:
    scenario = _scenario()
    trace_a = run_episode(scenario)
    trace_b = run_episode(scenario)
    assert trace_a.to_canonical_json() == trace_b.to_canonical_json()


def test_different_seed_traces_diverge() -> None:
    scenario = _scenario()
    assert (
        run_episode(scenario, seed=1).to_canonical_json()
        != run_episode(scenario, seed=2).to_canonical_json()
    )


def test_trace_content_hash_is_stable_and_seed_sensitive() -> None:
    scenario = _scenario()
    h1 = run_episode(scenario).content_hash
    assert h1 == run_episode(scenario).content_hash  # same seed -> same digest
    assert len(h1) == 64 and all(c in "0123456789abcdef" for c in h1)
    assert run_episode(scenario, seed=1).content_hash != run_episode(scenario, seed=2).content_hash


def test_trace_records_reproducibility_provenance() -> None:
    trace = run_episode(_scenario(), seed=5)
    assert trace.provenance["seed"] == 5
    assert trace.provenance["scenario"] == "unit"
    # the envelope RM-P0-SIM-09 extends; SIM-01 stamps the interfaces it is built against.
    assert trace.provenance["core_interfaces"] == CORE_INTERFACES


# --- per-agent frame (heterogeneous assets) --------------------------------------


def test_per_agent_frame_overrides_scenario_default() -> None:
    scenario = _scenario(
        agents=(
            AgentSpec(agent_id="orbiter", frame=INERTIAL_J2000),
            AgentSpec(agent_id="rover"),  # inherits the scenario default frame
        ),
    )
    observations = Simulator(scenario).reset().observations
    assert observations["orbiter"].self_state.frame == INERTIAL_J2000
    assert observations["rover"].self_state.frame == MOON_BODY_FIXED


# --- termination at the battery floor (shrinking active set) ----------------------


def test_agent_terminates_at_battery_floor_and_leaves_active_set() -> None:
    scenario = _scenario(
        agents=(
            AgentSpec(agent_id="dying", battery_soc_j=0.4),  # floor 0.0, draw 1.0 W
            AgentSpec(agent_id="alive", battery_soc_j=100.0),
        ),
        dt_s=1.0,
        horizon_steps=2,
    )
    sim = Simulator(scenario)
    sim.reset()

    first = sim.step(ActionBatch())  # dying drains 0.4 -> clamped to the floor -> terminates
    assert first.terminations == {"dying": True, "alive": False}
    assert "dying" in first.observations  # the terminal observation is still emitted
    assert first.observations["dying"].self_state.battery_soc_j == 0.0  # clamped, not negative
    assert sim.agents == ("alive",)  # dropped from the active set

    second = sim.step(ActionBatch())  # dying is gone; only alive remains
    assert "dying" not in second.observations
    assert set(second.terminations) == {"alive"}
