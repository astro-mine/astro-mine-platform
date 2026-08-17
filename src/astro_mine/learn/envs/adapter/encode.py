# SPDX-License-Identifier: Apache-2.0
"""Observation → space sample and space sample → Core action codecs (RM-P1-LEARN-01).

The two pure functions that bridge the Core message vocabulary and the RL tensor
world, keyed off an :class:`~astro_mine.learn.envs.adapter.spaces.AgentSpaceSpec`:

- :func:`encode_observation` renders a Core
  :class:`~astro_mine.core.messages.model.Observation` into a fixed-shape, fixed-dtype
  numpy sample that is guaranteed ``in`` the agent's declared observation space
  (deterministic ordering; every block clipped into its Box).
- :func:`decode_action` turns one numpy action-dict sample back into a single Core
  :class:`~astro_mine.core.messages.model.Action` (a tagged union — the ``kind``
  selector picks which task/mode the sample encodes).

:func:`zero_observation` is the neutral, in-space sample the adapter emits for a masked
or unobservable agent, so masking leaks no sensor content.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.enums import ActionKind, ExcavationPattern, TaskKind
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    ExcavateTask,
    GotoTask,
    HopTask,
    ModeCommand,
    Observation,
    ProspectTask,
    Quat,
    SampleTask,
    TaskDirective,
    Transform,
    Vec3,
    Volume,
)
from astro_mine.learn.envs.adapter.spaces import AgentSpaceSpec

__all__ = [
    "decode_action",
    "encode_action_batch",
    "encode_observation",
    "zero_observation",
]

_SENSOR_VALUE_WIDTH = 4
_POS_BOUND = 1.0e6  # denormalizes the [-1, 1] mobility target blocks back to metres
_IDENTITY_QUAT = Quat(x=0.0, y=0.0, z=0.0, w=1.0)

ObsSample = dict[str, "NDArray[np.float32]"]


def _clip(values: list[float], box: spaces.Box) -> NDArray[np.float32]:
    arr = np.asarray(values, dtype=np.float32)
    # box.low / box.high are float32, so clipping into them then casting is a no-op that
    # *guarantees* the result is ``in`` the Box (no dtype/rounding escape).
    return np.clip(arr, box.low, box.high).astype(np.float32)


def encode_observation(obs: Observation, spec: AgentSpaceSpec) -> ObsSample:
    """Render a Core observation into the agent's declared space sample.

    The returned dict has exactly the observation space's keys, each a float32 array
    clipped into its Box — so ``obs in spec.observation_space`` always holds."""
    space = spec.observation_space
    out: ObsSample = {}

    ss = obs.self_state
    out["self_state"] = _clip(
        [
            ss.pose.translation_m.x,
            ss.pose.translation_m.y,
            ss.pose.translation_m.z,
            ss.battery_soc_j if ss.battery_soc_j is not None else 0.0,
            ss.temperature_k if ss.temperature_k is not None else 0.0,
        ],
        _box(space, "self_state"),
    )

    readings = {r.sensor: r for r in obs.sensors}
    for cap, names in spec.sensor_names_by_capability.items():
        vec: list[float] = []
        for name in names:
            reading = readings.get(name)
            values = list(reading.values) if reading is not None else []
            values = (values + [0.0] * _SENSOR_VALUE_WIDTH)[:_SENSOR_VALUE_WIDTH]
            vec.extend(values)
        out[cap.value] = _clip(vec, _box(space, cap.value))

    if spec.has_comms:
        reachable = {link.peer: link.reachable for link in (obs.comms.links if obs.comms else [])}
        comms_vec = [1.0 if reachable.get(peer) else 0.0 for peer in spec.peers]
        comms_vec.append(1.0 if (obs.comms is not None and obs.comms.earth_contact) else 0.0)
        out["comms"] = _clip(comms_vec, _box(space, "comms"))

    neighbors = {n.agent_id: n for n in obs.neighbors}
    nb_vec: list[float] = []
    for peer in spec.peers:
        n = neighbors.get(peer)
        if n is not None:
            nb_vec.extend(
                [1.0, n.pose.translation_m.x, n.pose.translation_m.y, n.pose.translation_m.z]
            )
        else:
            nb_vec.extend([0.0, 0.0, 0.0, 0.0])
    out["neighbors"] = _clip(nb_vec, _box(space, "neighbors"))

    return out


def zero_observation(spec: AgentSpaceSpec) -> ObsSample:
    """The neutral, in-space observation for a masked/unobservable agent (no leak).

    Every Box in the observation space contains its zero vector, so this is always
    ``in`` the space."""
    return {
        key: np.zeros(subspace.shape, dtype=np.float32)
        for key, subspace in spec.observation_space.spaces.items()
        if isinstance(subspace, spaces.Box)
    }


def _box(space: spaces.Dict, key: str) -> spaces.Box:
    subspace = space.spaces[key]
    assert isinstance(subspace, spaces.Box)  # every observation block is a Box
    return subspace


def decode_action(sample: Mapping[str, Any], spec: AgentSpaceSpec) -> Action:
    """Decode one numpy action-dict sample into a single Core :class:`Action`.

    The ``kind`` index selects the modality (the Core Action is a tagged union, so a
    step encodes exactly one decision); ``goto``/``hop`` blocks are denormalized from
    ``[-1, 1]`` back to metres, and the discrete task modalities map to a whole-target
    :class:`~astro_mine.core.messages.model.TaskDirective` with the agent's declared
    tool/method."""
    modality = spec.modalities[int(sample["kind"])]
    frame = spec.root_frame

    if modality == "mode":
        mode_name = spec.mode_names[int(sample["mode"])]
        return Action(
            agent_id=spec.agent_id, kind=ActionKind.MODE, mode=ModeCommand(mode=mode_name)
        )

    if modality == "goto":
        xyz = _denormalize(sample["goto"])
        task = TaskDirective(
            task_kind=TaskKind.GOTO,
            goto=GotoTask(
                target_frame=frame,
                target_pose=Transform(translation_m=Vec3(**xyz), rotation_quat_xyzw=_IDENTITY_QUAT),
            ),
        )
    elif modality == "hop":
        xyz = _denormalize(sample["hop"])
        task = TaskDirective(
            task_kind=TaskKind.HOP, hop=HopTask(launch_frame=frame, target_point_m=Vec3(**xyz))
        )
    elif modality == "excavate":
        task = TaskDirective(
            task_kind=TaskKind.EXCAVATE,
            excavate=ExcavateTask(
                region=_unit_volume(frame),
                tool=spec.excavation_tool,
                pattern=ExcavationPattern.TRENCH,
            ),
        )
    elif modality == "sample":
        task = TaskDirective(
            task_kind=TaskKind.SAMPLE,
            sample=SampleTask(
                site_frame=frame,
                target_point_m=Vec3(x=0.0, y=0.0, z=0.0),
                method=spec.sample_method,
            ),
        )
    elif modality == "prospect":
        task = TaskDirective(
            task_kind=TaskKind.PROSPECT,
            prospect=ProspectTask(
                region=_unit_volume(frame), sensor_kinds=list(spec.prospect_sensor_kinds)
            ),
        )
    elif modality == "isru":
        # No ISRU TaskKind in the v0.1 catalog: an ISRU process is a CUSTOM directive.
        task = TaskDirective(task_kind=TaskKind.CUSTOM, directive="isru_process")
    else:  # pragma: no cover - modalities is closed over the branches above
        raise ValueError(f"unknown action modality {modality!r}")

    return Action(agent_id=spec.agent_id, kind=ActionKind.TASK, task=task)


def encode_action_batch(
    actions: Mapping[AgentId, Mapping[str, Any]], specs: Mapping[AgentId, AgentSpaceSpec]
) -> ActionBatch:
    """Decode a per-agent action-sample map into a Core :class:`ActionBatch`."""
    return ActionBatch(
        actions=[decode_action(sample, specs[agent]) for agent, sample in actions.items()]
    )


def _denormalize(block: Any) -> dict[str, float]:
    arr = np.asarray(block, dtype=np.float64) * _POS_BOUND
    return {"x": float(arr[0]), "y": float(arr[1]), "z": float(arr[2])}


def _unit_volume(frame: str) -> Volume:
    return Volume(
        frame=frame,
        center_m=Vec3(x=0.0, y=0.0, z=0.0),
        dimensions_m=Vec3(x=1.0, y=1.0, z=1.0),
    )
