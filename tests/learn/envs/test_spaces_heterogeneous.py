"""Per-agent spaces are keyed by the agent's SADF capabilities (RM-P1-LEARN-01).

Heterogeneous agents get *different* observation/action Dict key sets, derived from the
Core-typed :class:`~astro_mine.core.sadf.model.Asset` (capabilities + sensor suite), and
the observation-mask / comms-channel are exposed as first-class ``infos`` fields.
"""

from __future__ import annotations

from gymnasium import spaces

from astro_mine.core.messages.enums import ActionKind, ExcavationTool, SampleMethod, TaskKind
from astro_mine.core.messages.model import ActionBatch
from astro_mine.core.sadf.enums import CapabilityTag, CommsBand, SensorKind
from astro_mine.core.sadf.model import Asset, Comms, Identity, ModeLoad, PowerBudget, Sensor
from astro_mine.learn.envs import SwarmEnv, observation_mask
from astro_mine.learn.envs.adapter.encode import decode_action
from astro_mine.learn.envs.adapter.spaces import build_agent_spec
from tests.learn.fakes import AGENTS, FakeSwarmWorld


def test_observation_space_keys_are_capability_specific(swarm_env: SwarmEnv) -> None:
    rover = swarm_env.observation_space("rover")
    digger = swarm_env.observation_space("digger")
    relay = swarm_env.observation_space("relay")
    assert isinstance(rover, spaces.Dict)

    # rover: imaging + neutron sensing blocks + comms + neighbors + self_state
    assert set(rover.spaces) == {
        "self_state",
        "sensing.imaging",
        "prospecting.neutron",
        "comms",
        "neighbors",
    }
    # digger: lidar only, and NO comms block (it declares no radio)
    assert set(digger.spaces) == {"self_state", "sensing.lidar", "neighbors"}
    assert "comms" not in digger.spaces
    # relay: imaging + comms (relay radio), no prospecting/excavation
    assert set(relay.spaces) == {"self_state", "sensing.imaging", "comms", "neighbors"}


def test_action_space_keys_are_capability_specific(swarm_env: SwarmEnv) -> None:
    rover = swarm_env.action_space("rover")
    digger = swarm_env.action_space("digger")
    assert isinstance(rover, spaces.Dict)
    # rover is mobile (goto) and prospects → kind selects among {mode, goto, prospect}
    assert set(rover.spaces) == {"kind", "mode", "goto"}
    assert rover.spaces["kind"].n == 3  # mode, goto, prospect
    # digger is mobile (goto) and excavates → {mode, goto, excavate}
    assert set(digger.spaces) == {"kind", "mode", "goto"}
    assert digger.spaces["kind"].n == 3  # mode, goto, excavate


def test_spaces_are_the_same_object_each_call(swarm_env: SwarmEnv) -> None:
    # PettingZoo/Gymnasium identity contract: same object per agent, per call.
    for agent in AGENTS:
        assert swarm_env.observation_space(agent) is swarm_env.observation_space(agent)
        assert swarm_env.action_space(agent) is swarm_env.action_space(agent)


def test_mask_and_comms_are_first_class_infos(swarm_env: SwarmEnv) -> None:
    _, infos = swarm_env.reset(seed=0)
    for agent in swarm_env.agents:
        assert "observation_mask" in infos[agent]
        assert "comms" in infos[agent]
        assert "earth_contact" in infos[agent]
    # rover's comms channel lists its reachable peer (relay), never the denied one (digger)
    rover_peers = [link.peer for link in infos["rover"]["comms"]]
    assert rover_peers == ["relay"]
    assert infos["digger"]["comms"] == []  # digger has no radio


def test_observation_mask_helper_reads_observable_flag() -> None:
    world = FakeSwarmWorld()
    world.reset(seed=0)
    # relay is unobservable when tick % 3 == 2 → step to tick 2.
    world.step(ActionBatch())  # tick 1
    result = world.step(ActionBatch())  # tick 2
    mask = observation_mask(result)  # StepResult input form
    assert mask["relay"] is False
    assert mask["rover"] is True


def _all_modality_asset() -> Asset:
    return Asset(
        identity=Identity(id="omni", name="Omni Hopper", version="0.1.0", kind="hopper"),
        capabilities=[
            CapabilityTag.MOBILITY_HOP,
            CapabilityTag.EXCAVATION_DRILL,
            CapabilityTag.SAMPLE_COLLECTION_SCOOP,
            CapabilityTag.PROSPECTING_NIR,
            CapabilityTag.ISRU_ELECTROLYSIS,
            CapabilityTag.SENSING_THERMAL,
            CapabilityTag.COMMS_DTN,
        ],
        root_frame="body",
        sensors=[
            Sensor(name="nir", kind=SensorKind.NIR_SPECTROMETER, frame="body"),
            Sensor(name="temp", kind=SensorKind.THERMAL_SENSOR, frame="body"),
        ],
        comms=[Comms(name="dtn", band=CommsBand.UHF)],
        power=PowerBudget(
            loads_by_mode=[ModeLoad(mode="cruise", power_w=5.0), ModeLoad(mode="dig", power_w=25.0)]
        ),
    )


def test_declared_sensing_capability_without_a_sensor_yields_no_block() -> None:
    # An asset may declare a sensing capability it has no matching Sensor for; that
    # capability must NOT produce an observation block (the block is sensor-backed).
    asset = Asset(
        identity=Identity(id="blind", name="Blind", version="0.1.0", kind="rover"),
        capabilities=[CapabilityTag.MOBILITY_WHEELED, CapabilityTag.SENSING_LIDAR],
        root_frame="body",
        sensors=[],  # declares sensing.lidar but ships no lidar sensor
    )
    spec = build_agent_spec("blind", asset, ("blind",))
    assert "sensing.lidar" not in spec.observation_space.spaces
    assert set(spec.observation_space.spaces) == {"self_state", "neighbors"}


def test_all_action_modalities_decode_to_valid_core_actions() -> None:
    spec = build_agent_spec("omni", _all_modality_asset(), ("omni", "peer"))
    # A pure-hop mobility gives a hop block but no goto block.
    assert set(spec.action_space.spaces) == {"kind", "mode", "hop"}
    assert spec.modalities == ("mode", "hop", "excavate", "sample", "prospect", "isru")
    assert spec.mode_names == ("cruise", "dig")  # from loads_by_mode, not the default
    assert spec.excavation_tool is ExcavationTool.DRILL
    assert spec.sample_method is SampleMethod.SCOOP
    assert "nir_spectrometer" in spec.prospect_sensor_kinds

    base = {"hop": [0.5, -0.5, 0.0], "mode": 1}
    expected = {
        "mode": (ActionKind.MODE, None),
        "hop": (ActionKind.TASK, TaskKind.HOP),
        "excavate": (ActionKind.TASK, TaskKind.EXCAVATE),
        "sample": (ActionKind.TASK, TaskKind.SAMPLE),
        "prospect": (ActionKind.TASK, TaskKind.PROSPECT),
        "isru": (ActionKind.TASK, TaskKind.CUSTOM),
    }
    for kind_idx, modality in enumerate(spec.modalities):
        action = decode_action({"kind": kind_idx, **base}, spec)
        exp_kind, exp_task = expected[modality]
        assert action.kind is exp_kind
        assert action.agent_id == "omni"
        if exp_task is None:
            assert action.mode is not None
            assert action.mode.mode == "dig"  # mode index 1
        else:
            assert action.task is not None
            assert action.task.task_kind is exp_task
