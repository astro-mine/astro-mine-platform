# SPDX-License-Identifier: Apache-2.0
"""Per-agent Gymnasium spaces built from a SADF asset (RM-P1-LEARN-01).

The Core Environment protocol exposes neither spaces nor SADF assets (its surface is
only ``possible_agents``/``agents``/``reset``/``step``; core.md §3). Learn therefore
derives the per-agent observation/action *spaces* from the asset's declared
:class:`~astro_mine.core.sadf.model.Asset` — the Core-typed capability list and sensor
suite — never from a Sim/Fleet import. This keeps the narrow waist intact: the adapter
reaches the world only through ``astro_mine.core``.

Spaces are **capability-keyed**, so heterogeneous agents get different key sets
(learn.md §3; the SADF ``CapabilityTag`` vocabulary is the space-dict key substrate,
sadf/enums.py). The observation Dict always carries a ``self_state`` block; one block
per declared sensing/prospecting capability that the agent actually has a sensor for;
a ``comms`` block when the agent has a radio; and a fixed-width ``neighbors`` block.
The action Dict carries a ``kind`` selector over the agent's task/mode modalities
(the tagged-union Core :class:`~astro_mine.core.messages.model.Action` is single-choice
per step), a ``mode`` selector, and continuous ``goto``/``hop`` target blocks when the
agent is mobile.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.enums import ExcavationTool, SampleMethod
from astro_mine.core.sadf.enums import CapabilityTag, SensorKind
from astro_mine.core.sadf.model import Asset

__all__ = [
    "AgentSpaceSpec",
    "build_agent_spec",
]

# --- fixed widths / bounds (SI, generous finite bounds so encoded samples clip in) ---

#: ``self_state`` block: pose x/y/z (m), battery state-of-charge (J), temperature (K).
_POS_BOUND = 1.0e6
_BATTERY_BOUND = 1.0e12
_TEMP_BOUND = 1.0e4
#: Per-sensor reading slots packed into a sensing/prospecting block (pad/truncate).
_SENSOR_VALUE_WIDTH = 4
_SENSOR_BOUND = 1.0e6
#: Per-neighbour slots: presence flag + relative-frame pose x/y/z.
_NEIGHBOR_WIDTH = 4

# --- capability → sensor-kind / task-modality vocabularies -------------------------

#: Sensing/prospecting capability tags that key an observation block, mapped to the
#: SADF sensor kinds that back them. A capability only yields a block when the agent
#: also declares at least one matching sensor.
_SENSING_CAPABILITY_KINDS: dict[CapabilityTag, frozenset[SensorKind]] = {
    CapabilityTag.SENSING_IMAGING: frozenset({SensorKind.IMAGING}),
    CapabilityTag.SENSING_RANGING: frozenset({SensorKind.RANGEFINDER}),
    CapabilityTag.SENSING_LIDAR: frozenset({SensorKind.LIDAR}),
    CapabilityTag.SENSING_IMU: frozenset({SensorKind.IMU}),
    CapabilityTag.SENSING_ODOMETRY: frozenset({SensorKind.ODOMETRY}),
    CapabilityTag.SENSING_CONTACT: frozenset({SensorKind.CONTACT}),
    CapabilityTag.SENSING_ALTIMETRY: frozenset({SensorKind.ALTIMETER}),
    CapabilityTag.SENSING_THERMAL: frozenset({SensorKind.THERMAL_SENSOR}),
    CapabilityTag.PROSPECTING_NEUTRON: frozenset({SensorKind.NEUTRON_SPECTROMETER}),
    CapabilityTag.PROSPECTING_NIR: frozenset({SensorKind.NIR_SPECTROMETER}),
    CapabilityTag.PROSPECTING_GPR: frozenset({SensorKind.GPR}),
    CapabilityTag.PROSPECTING_MASS_SPEC: frozenset({SensorKind.MASS_SPECTROMETER}),
    CapabilityTag.PROSPECTING_DRILL_ASSAY: frozenset({SensorKind.DRILL_ASSAY}),
}

#: Mobility tags that grant a continuous GOTO target block (everything but pure hop).
_GOTO_MOBILITY: frozenset[CapabilityTag] = frozenset(
    {
        CapabilityTag.MOBILITY_WHEELED,
        CapabilityTag.MOBILITY_TRACKED,
        CapabilityTag.MOBILITY_LEGGED,
        CapabilityTag.MOBILITY_ORBITER,
        CapabilityTag.MOBILITY_ROCKET,
    }
)
_EXCAVATION_TAGS: frozenset[CapabilityTag] = frozenset(
    {
        CapabilityTag.EXCAVATION_BUCKET,
        CapabilityTag.EXCAVATION_AUGER,
        CapabilityTag.EXCAVATION_SCOOP,
        CapabilityTag.EXCAVATION_DRILL,
    }
)
_SAMPLE_TAGS: frozenset[CapabilityTag] = frozenset(
    {
        CapabilityTag.SAMPLE_COLLECTION_DRILL,
        CapabilityTag.SAMPLE_COLLECTION_SCOOP,
        CapabilityTag.SAMPLE_COLLECTION_AUGER,
        CapabilityTag.SAMPLE_COLLECTION_PNEUMATIC,
        CapabilityTag.SAMPLE_COLLECTION_CAPTURE_BAG,
    }
)
_PROSPECTING_TAGS: frozenset[CapabilityTag] = frozenset(
    {
        CapabilityTag.PROSPECTING_NEUTRON,
        CapabilityTag.PROSPECTING_NIR,
        CapabilityTag.PROSPECTING_GPR,
        CapabilityTag.PROSPECTING_MASS_SPEC,
        CapabilityTag.PROSPECTING_DRILL_ASSAY,
    }
)
_ISRU_TAGS: frozenset[CapabilityTag] = frozenset(
    {
        CapabilityTag.ISRU_THERMAL_EXTRACTION,
        CapabilityTag.ISRU_ELECTROLYSIS,
        CapabilityTag.ISRU_PURIFICATION,
        CapabilityTag.ISRU_STORAGE,
    }
)
_COMMS_TAGS: frozenset[CapabilityTag] = frozenset(
    {
        CapabilityTag.COMMS_RELAY,
        CapabilityTag.COMMS_DIRECT_TO_EARTH,
        CapabilityTag.COMMS_DSN,
        CapabilityTag.COMMS_DTN,
    }
)

#: First-match excavation tool / sample method for a declared capability (decoder use).
_EXCAVATION_TOOL_FOR_TAG: dict[CapabilityTag, ExcavationTool] = {
    CapabilityTag.EXCAVATION_BUCKET: ExcavationTool.BUCKET,
    CapabilityTag.EXCAVATION_AUGER: ExcavationTool.AUGER,
    CapabilityTag.EXCAVATION_SCOOP: ExcavationTool.SCOOP,
    CapabilityTag.EXCAVATION_DRILL: ExcavationTool.DRILL,
}
_SAMPLE_METHOD_FOR_TAG: dict[CapabilityTag, SampleMethod] = {
    CapabilityTag.SAMPLE_COLLECTION_DRILL: SampleMethod.DRILL,
    CapabilityTag.SAMPLE_COLLECTION_SCOOP: SampleMethod.SCOOP,
    CapabilityTag.SAMPLE_COLLECTION_AUGER: SampleMethod.AUGER,
    CapabilityTag.SAMPLE_COLLECTION_PNEUMATIC: SampleMethod.PNEUMATIC,
    CapabilityTag.SAMPLE_COLLECTION_CAPTURE_BAG: SampleMethod.CAPTURE_BAG,
}

_DEFAULT_MODES: tuple[str, ...] = ("idle", "active")


@dataclass(frozen=True)
class AgentSpaceSpec:
    """Everything the encoder/decoder needs for one agent, plus its cached spaces.

    Built once per agent from its SADF :class:`~astro_mine.core.sadf.model.Asset` and
    the environment's ``possible_agents``. The ``observation_space``/``action_space``
    objects are cached here so the adapter can hand back the *same* object each call
    (the PettingZoo/Gymnasium identity contract)."""

    agent_id: AgentId
    root_frame: str
    peers: tuple[AgentId, ...]
    #: Ordered sensor names backing each declared sensing/prospecting capability block.
    sensor_names_by_capability: dict[CapabilityTag, tuple[str, ...]]
    has_comms: bool
    #: Ordered action modalities the ``kind`` selector chooses among.
    modalities: tuple[str, ...]
    mode_names: tuple[str, ...]
    excavation_tool: ExcavationTool
    sample_method: SampleMethod
    prospect_sensor_kinds: tuple[str, ...]
    observation_space: spaces.Dict
    action_space: spaces.Dict


def _box(low: list[float], high: list[float]) -> spaces.Box:
    return spaces.Box(
        low=np.asarray(low, dtype=np.float32),
        high=np.asarray(high, dtype=np.float32),
        dtype=np.float32,
    )


def _self_state_box() -> spaces.Box:
    return _box(
        low=[-_POS_BOUND, -_POS_BOUND, -_POS_BOUND, 0.0, 0.0],
        high=[_POS_BOUND, _POS_BOUND, _POS_BOUND, _BATTERY_BOUND, _TEMP_BOUND],
    )


def _sensor_block_box(n_sensors: int) -> spaces.Box:
    width = n_sensors * _SENSOR_VALUE_WIDTH
    return _box([-_SENSOR_BOUND] * width, [_SENSOR_BOUND] * width)


def _comms_box(n_peers: int) -> spaces.Box:
    # per-peer reachability flag (0/1) + earth-contact flag (0/1)
    return _box([0.0] * (n_peers + 1), [1.0] * (n_peers + 1))


def _neighbors_box(n_peers: int) -> spaces.Box:
    low: list[float] = []
    high: list[float] = []
    for _ in range(n_peers):
        low.extend([0.0, -_POS_BOUND, -_POS_BOUND, -_POS_BOUND])
        high.extend([1.0, _POS_BOUND, _POS_BOUND, _POS_BOUND])
    return _box(low, high)


def _mode_names(asset: Asset) -> tuple[str, ...]:
    if asset.power is not None and asset.power.loads_by_mode:
        seen: dict[str, None] = {}
        for load in asset.power.loads_by_mode:
            seen.setdefault(load.mode, None)
        if seen:
            return tuple(seen)
    return _DEFAULT_MODES


def _first_by_value(tags: frozenset[CapabilityTag], present: set[CapabilityTag]) -> CapabilityTag:
    return sorted(tags & present, key=lambda t: t.value)[0]


def build_agent_spec(
    agent_id: AgentId, asset: Asset, possible_agents: tuple[AgentId, ...]
) -> AgentSpaceSpec:
    """Derive one agent's capability-keyed observation/action spaces from its asset.

    Reads only the Core-typed :class:`~astro_mine.core.sadf.model.Asset` (capabilities +
    sensor suite), never a Sim/Fleet type. Sensing/prospecting blocks appear only for a
    capability the agent both *declares* and has a *sensor* for, so the key set is
    genuinely per-agent."""
    caps = set(asset.capabilities)
    peers = tuple(a for a in possible_agents if a != agent_id)
    n_peers = len(peers)

    # Observation blocks -----------------------------------------------------------
    obs_dict: dict[str, spaces.Space[NDArray[np.float32]]] = {"self_state": _self_state_box()}
    sensors_by_name = {s.name: s for s in asset.sensors}
    sensor_names_by_capability: dict[CapabilityTag, tuple[str, ...]] = {}
    for cap in sorted(_SENSING_CAPABILITY_KINDS, key=lambda t: t.value):
        if cap not in caps:
            continue
        kinds = _SENSING_CAPABILITY_KINDS[cap]
        names = tuple(sorted(name for name, s in sensors_by_name.items() if s.kind in kinds))
        if not names:
            continue
        sensor_names_by_capability[cap] = names
        obs_dict[cap.value] = _sensor_block_box(len(names))

    has_comms = bool(caps & _COMMS_TAGS) or bool(asset.comms)
    if has_comms:
        obs_dict["comms"] = _comms_box(n_peers)
    obs_dict["neighbors"] = _neighbors_box(n_peers)

    # Action modalities + blocks ---------------------------------------------------
    modalities: list[str] = ["mode"]
    action_dict: dict[str, spaces.Space[NDArray[np.float32]]] = {}
    if caps & _GOTO_MOBILITY:
        modalities.append("goto")
        action_dict["goto"] = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    if CapabilityTag.MOBILITY_HOP in caps:
        modalities.append("hop")
        action_dict["hop"] = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    if caps & _EXCAVATION_TAGS:
        modalities.append("excavate")
    if caps & _SAMPLE_TAGS:
        modalities.append("sample")
    if caps & _PROSPECTING_TAGS:
        modalities.append("prospect")
    if caps & _ISRU_TAGS:
        modalities.append("isru")

    mode_names = _mode_names(asset)
    action_dict["kind"] = spaces.Discrete(len(modalities))
    action_dict["mode"] = spaces.Discrete(len(mode_names))

    excavation_tool = (
        _EXCAVATION_TOOL_FOR_TAG[_first_by_value(_EXCAVATION_TAGS, caps)]
        if caps & _EXCAVATION_TAGS
        else ExcavationTool.BUCKET
    )
    sample_method = (
        _SAMPLE_METHOD_FOR_TAG[_first_by_value(_SAMPLE_TAGS, caps)]
        if caps & _SAMPLE_TAGS
        else SampleMethod.SCOOP
    )
    prospect_sensor_kinds = tuple(
        sorted(
            {
                s.kind.value
                for cap in _PROSPECTING_TAGS & caps
                for s in asset.sensors
                if s.kind in _SENSING_CAPABILITY_KINDS.get(cap, frozenset())
            }
        )
    )

    return AgentSpaceSpec(
        agent_id=agent_id,
        root_frame=asset.root_frame,
        peers=peers,
        sensor_names_by_capability=sensor_names_by_capability,
        has_comms=has_comms,
        modalities=tuple(modalities),
        mode_names=mode_names,
        excavation_tool=excavation_tool,
        sample_method=sample_method,
        prospect_sensor_kinds=prospect_sensor_kinds,
        observation_space=spaces.Dict(obs_dict),
        action_space=spaces.Dict(action_dict),
    )
