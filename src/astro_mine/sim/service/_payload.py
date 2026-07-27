"""The per-tick observation payload codec — FlatBuffers (conventions.md §3, §4; sim.md §6, §11).

The platform splits its wire formats deliberately (conventions.md §3): **Protobuf/gRPC** is the
*control plane* (``reset`` / ``step`` RPCs, service contracts), and **FlatBuffers/Cap'n Proto**
carry the **per-tick sensor/telemetry payloads** — the high-rate stream where a Protobuf decode per
agent per tick would dominate the step at swarm scale. sim.md §11 names that split as the
recommendation ("per-tick payload encoding: FlatBuffers/Cap'n Proto for per-tick sensor/telemetry;
Protobuf for control-plane RPCs"), so the ``EnvironmentService`` follows it: the RPC envelopes are
Protobuf and the observation frame inside them is this FlatBuffers buffer.

FlatBuffers' point is that reading a field is a **pointer offset**, not a parse: a consumer that
only wants one agent's battery SoC never touches the rest of the frame. This module encodes an
``ObservationFrame`` directly against the FlatBuffers builder/table API rather than through
generated code, so the repo needs no ``flatc`` binary in its toolchain (Python is the only build
dependency); the schema below *is* the contract, and :func:`decode_frame` is its reader.

.. code-block:: text

    table SensorReading { name:string; values:[double]; unit:string; species:string; valid:bool; }
    table AgentObservation {
        agent_id:string;              // 0 pose:[double];                // 1  (7: x y z  qx qy qz
        qw) velocity:[double];            // 2  (3, empty when unset) battery_soc_j:double;
        // 3 has_battery:bool;             // 4 temperature_k:double;         // 5
        has_temperature:bool;         // 6 mode:string;                  // 7
        sensors:[SensorReading];      // 8
    } table ObservationFrame {
        tick:int;                     // 0 sim_time_s:double;            // 1
        agents:[AgentObservation];    // 2
    }

Encoding is **deterministic**: agents are emitted in the caller's order and every field is written
unconditionally, so the same observations always produce byte-identical buffers — a served run stays
as reproducible as an in-process one (conventions.md §11).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import flatbuffers
import flatbuffers.number_types as N
from flatbuffers import encode, packer
from flatbuffers import table as fb_table

from astro_mine.core.messages.model import (
    Observation,
    Quat,
    SensorReading,
    StateSample,
    Transform,
    Vec3,
)
from astro_mine.core.units import MOON_BODY_FIXED

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from astro_mine.core.units import ReferenceFrame

__all__ = ["decode_frame", "encode_frame"]


#: A FlatBuffers table field lives at vtable byte offset ``4 + 2 * slot``.
def _vt(slot: int) -> int:
    return 4 + 2 * slot


def _float_vector(builder: flatbuffers.Builder, values: Sequence[float]) -> int:
    """Write a ``[double]`` vector and return its offset (FlatBuffers vectors build in reverse)."""
    builder.StartVector(8, len(values), 8)
    for value in reversed(values):
        builder.PrependFloat64(float(value))
    return int(builder.EndVector())


def _offset_vector(builder: flatbuffers.Builder, offsets: Sequence[int]) -> int:
    """Write a vector of table offsets and return its offset."""
    builder.StartVector(4, len(offsets), 4)
    for offset in reversed(offsets):
        builder.PrependUOffsetTRelative(offset)
    return int(builder.EndVector())


def _encode_sensor(builder: flatbuffers.Builder, reading: SensorReading) -> int:
    name = builder.CreateString(reading.sensor)
    unit = builder.CreateString(reading.unit or "")
    species = builder.CreateString(reading.resource_species or "")
    values = _float_vector(builder, list(reading.values or []))
    builder.StartObject(5)
    builder.PrependUOffsetTRelativeSlot(0, name, 0)
    builder.PrependUOffsetTRelativeSlot(1, values, 0)
    builder.PrependUOffsetTRelativeSlot(2, unit, 0)
    builder.PrependUOffsetTRelativeSlot(3, species, 0)
    builder.PrependBoolSlot(4, reading.valid, False)
    return int(builder.EndObject())


def _encode_agent(builder: flatbuffers.Builder, observation: Observation) -> int:
    state = observation.self_state
    # Depth-first: every child (strings, vectors, nested tables) must be written before the table
    # that references it — the FlatBuffers builder grows backwards.
    sensor_offsets = [_encode_sensor(builder, r) for r in observation.sensors]
    sensors = _offset_vector(builder, sensor_offsets)
    agent_id = builder.CreateString(observation.agent_id)
    mode = builder.CreateString(state.mode or "")
    t, q = state.pose.translation_m, state.pose.rotation_quat_xyzw
    pose = _float_vector(builder, [t.x, t.y, t.z, q.x, q.y, q.z, q.w])
    v = state.linear_velocity_mps
    velocity = _float_vector(builder, [] if v is None else [v.x, v.y, v.z])

    builder.StartObject(9)
    builder.PrependUOffsetTRelativeSlot(0, agent_id, 0)
    builder.PrependUOffsetTRelativeSlot(1, pose, 0)
    builder.PrependUOffsetTRelativeSlot(2, velocity, 0)
    builder.PrependFloat64Slot(3, state.battery_soc_j or 0.0, 0.0)
    builder.PrependBoolSlot(4, state.battery_soc_j is not None, False)
    builder.PrependFloat64Slot(5, state.temperature_k or 0.0, 0.0)
    builder.PrependBoolSlot(6, state.temperature_k is not None, False)
    builder.PrependUOffsetTRelativeSlot(7, mode, 0)
    builder.PrependUOffsetTRelativeSlot(8, sensors, 0)
    return int(builder.EndObject())


def encode_frame(observations: Mapping[str, Observation]) -> bytes:
    """Encode one tick's per-agent observations into a FlatBuffers ``ObservationFrame`` buffer.

    The high-rate half of the served Environment (conventions.md §3): a consumer reads a field by
    pointer offset instead of parsing the frame. Deterministic — agents are emitted in the mapping's
    order and every field is written, so identical observations yield identical bytes."""
    builder = flatbuffers.Builder(1024)
    ordered = list(observations.values())
    tick = ordered[0].tick if ordered else 0
    sim_time_s = ordered[0].sim_time_s if ordered else 0.0

    agent_offsets = [_encode_agent(builder, observation) for observation in ordered]
    agents = _offset_vector(builder, agent_offsets)

    builder.StartObject(3)
    builder.PrependInt32Slot(0, tick, 0)
    builder.PrependFloat64Slot(1, sim_time_s, 0.0)
    builder.PrependUOffsetTRelativeSlot(2, agents, 0)
    builder.Finish(builder.EndObject())
    return bytes(builder.Output())


def _string(table: fb_table.Table, slot: int) -> str:
    offset = table.Offset(_vt(slot))
    if offset == 0:
        return ""
    raw = table.String(offset + table.Pos)
    return raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)


def _floats(table: fb_table.Table, slot: int) -> list[float]:
    offset = table.Offset(_vt(slot))
    if offset == 0:
        return []
    start = table.Vector(offset)
    return [float(table.Get(N.Float64Flags, start + i * 8)) for i in range(table.VectorLen(offset))]


def _float(table: fb_table.Table, slot: int) -> float:
    offset = table.Offset(_vt(slot))
    return 0.0 if offset == 0 else float(table.Get(N.Float64Flags, offset + table.Pos))


def _bool(table: fb_table.Table, slot: int) -> bool:
    offset = table.Offset(_vt(slot))
    return False if offset == 0 else bool(table.Get(N.BoolFlags, offset + table.Pos))


def _int(table: fb_table.Table, slot: int) -> int:
    offset = table.Offset(_vt(slot))
    return 0 if offset == 0 else int(table.Get(N.Int32Flags, offset + table.Pos))


def _child(table: fb_table.Table, position: int) -> fb_table.Table:
    """The nested table at ``position`` (a vector element's slot)."""
    return fb_table.Table(table.Bytes, table.Indirect(position))


def _decode_sensor(table: fb_table.Table) -> SensorReading:
    unit = _string(table, 2)
    species = _string(table, 3)
    return SensorReading(
        sensor=_string(table, 0),
        values=_floats(table, 1),
        unit=unit or None,
        resource_species=species or None,
        valid=_bool(table, 4),
    )


def _decode_agent(
    table: fb_table.Table, *, tick: int, sim_time_s: float, frame: ReferenceFrame
) -> Observation:
    agent_id = _string(table, 0)
    pose = _floats(table, 1)
    velocity = _floats(table, 2)
    mode = _string(table, 7)

    sensors: list[SensorReading] = []
    sensors_offset = table.Offset(_vt(8))
    if sensors_offset:
        start = table.Vector(sensors_offset)
        for i in range(table.VectorLen(sensors_offset)):
            sensors.append(_decode_sensor(_child(table, start + i * 4)))

    state = StateSample(
        agent_id=agent_id,
        frame=frame,
        pose=Transform(
            translation_m=Vec3(x=pose[0], y=pose[1], z=pose[2]),
            rotation_quat_xyzw=Quat(x=pose[3], y=pose[4], z=pose[5], w=pose[6]),
        ),
        linear_velocity_mps=(
            Vec3(x=velocity[0], y=velocity[1], z=velocity[2]) if velocity else None
        ),
        battery_soc_j=_float(table, 3) if _bool(table, 4) else None,
        temperature_k=_float(table, 5) if _bool(table, 6) else None,
        mode=mode or None,
    )
    return Observation(
        tick=tick,
        sim_time_s=sim_time_s,
        agent_id=agent_id,
        self_state=state,
        sensors=sensors,
    )


def decode_frame(data: bytes, *, frame: ReferenceFrame = MOON_BODY_FIXED) -> dict[str, Observation]:
    """Decode an ``ObservationFrame`` buffer back into Core observations — the inverse of
    :func:`encode_frame`.

    ``frame`` is the reference frame the agents' poses are expressed in. It is a property of the
    *scenario*, not of the tick, so it rides on the service's ``Describe`` response rather than
    being repeated in every high-rate frame (the whole point of the split: the per-tick payload
    carries only
    what actually changes per tick)."""
    buffer = bytearray(data)
    root = fb_table.Table(buffer, encode.Get(packer.uoffset, buffer, 0))
    tick = _int(root, 0)
    sim_time_s = _float(root, 1)

    observations: dict[str, Observation] = {}
    agents_offset = root.Offset(_vt(2))
    if agents_offset:
        start = root.Vector(agents_offset)
        for i in range(root.VectorLen(agents_offset)):
            observation = _decode_agent(
                _child(root, start + i * 4), tick=tick, sim_time_s=sim_time_s, frame=frame
            )
            observations[observation.agent_id] = observation
    return observations
