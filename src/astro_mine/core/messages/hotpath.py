# SPDX-License-Identifier: Apache-2.0
"""Observation <-> Cap'n Proto hot-path wire form (zero-copy per-tick decode).

The per-tick observation family uses Cap'n Proto, not Protobuf, so streaming
sensor/telemetry payloads decode **zero-copy** at swarm scale (conventions.md §3;
core.md §8). The canonical schema is ``schema/observation.capnp`` (shipped in-package,
loaded at runtime — pycapnp needs no codegen step, which is what makes it the right
pre-buf choice; RM-P0-CORE-07 owns the codegen pipeline).

The Pydantic :class:`~astro_mine.core.messages.model.Observation` is the ergonomic API;
this module maps it to/from the Cap'n Proto encoding. Field names map 1:1 between
snake_case (Pydantic) and lowerCamelCase (Cap'n Proto); optional Float64 scalars carry
a companion ``has*`` flag, optional Text/struct fields use pointer presence.

- :func:`to_bytes` / :func:`from_bytes` — encode / full decode (round-trips exactly);
- :func:`reader` — a zero-copy reader over the bytes for hot-path field access without
  materializing the Python model.
"""

from __future__ import annotations

from functools import lru_cache
from importlib import resources
from typing import Any

import capnp

from astro_mine.core.messages.model import (
    CommsObservationMask,
    Observation,
    PeerLink,
    Quat,
    SensorReading,
    StateSample,
    Transform,
    Vec3,
)
from astro_mine.core.units import Epoch, FrameClass, ReferenceFrame, TimeScale

__all__ = ["from_bytes", "reader", "to_bytes", "to_proto_message"]

_SCHEMA_RESOURCE = "schema/observation.capnp"


@lru_cache(maxsize=1)
def _schema() -> Any:
    """Load the canonical Cap'n Proto schema (shipped inside the package)."""
    # as_file materializes the resource to a real path (wheel installs included);
    # capnp.load needs a filesystem path. The loaded module is cached for the process.
    with resources.as_file(
        resources.files("astro_mine.core.messages").joinpath(_SCHEMA_RESOURCE)
    ) as path:
        return capnp.load(str(path))


# --- encode ----------------------------------------------------------------------


def _vec3(v: Vec3) -> dict[str, float]:
    return {"x": v.x, "y": v.y, "z": v.z}


def _quat(q: Quat) -> dict[str, float]:
    return {"x": q.x, "y": q.y, "z": q.z, "w": q.w}


def _transform(t: Transform) -> dict[str, Any]:
    return {"translationM": _vec3(t.translation_m), "rotationQuatXyzw": _quat(t.rotation_quat_xyzw)}


def _frame(rf: ReferenceFrame) -> dict[str, Any]:
    d: dict[str, Any] = {"name": rf.name, "frameClass": rf.frame_class.value}
    if rf.center is not None:
        d["center"] = rf.center
    return d


def _epoch(e: Epoch) -> dict[str, Any]:
    return {"tdbSeconds": e.tdb_seconds, "scale": e.scale.value}


def _state(s: StateSample) -> dict[str, Any]:
    d: dict[str, Any] = {
        "agentId": s.agent_id,
        "frame": _frame(s.frame),
        "pose": _transform(s.pose),
        "hasBatterySocJ": s.battery_soc_j is not None,
        "batterySocJ": s.battery_soc_j or 0.0,
        "hasTemperatureK": s.temperature_k is not None,
        "temperatureK": s.temperature_k or 0.0,
    }
    if s.linear_velocity_mps is not None:
        d["linearVelocityMps"] = _vec3(s.linear_velocity_mps)
    if s.angular_velocity_rps is not None:
        d["angularVelocityRps"] = _vec3(s.angular_velocity_rps)
    if s.mode is not None:
        d["mode"] = s.mode
    return d


def _sensor(r: SensorReading) -> dict[str, Any]:
    d: dict[str, Any] = {
        "sensor": r.sensor,
        "values": list(r.values),
        "hasNoiseSigma": r.noise_sigma is not None,
        "noiseSigma": r.noise_sigma or 0.0,
        "valid": r.valid,
    }
    if r.unit is not None:
        d["unit"] = r.unit
    if r.resource_species is not None:
        d["resourceSpecies"] = r.resource_species
    return d


def _peer(p: PeerLink) -> dict[str, Any]:
    return {
        "peer": p.peer,
        "reachable": p.reachable,
        "hasRateBps": p.rate_bps is not None,
        "rateBps": p.rate_bps or 0.0,
        "hasLatencyS": p.latency_s is not None,
        "latencyS": p.latency_s or 0.0,
        "hasMarginDb": p.margin_db is not None,
        "marginDb": p.margin_db or 0.0,
    }


def _comms(c: CommsObservationMask) -> dict[str, Any]:
    return {
        "agentId": c.agent_id,
        "links": [_peer(p) for p in c.links],
        "earthContact": c.earth_contact,
    }


def _observation_dict(obs: Observation) -> dict[str, Any]:
    d: dict[str, Any] = {
        "tick": obs.tick,
        "simTimeS": obs.sim_time_s,
        "agentId": obs.agent_id,
        "observable": obs.observable,
        "selfState": _state(obs.self_state),
        "sensors": [_sensor(s) for s in obs.sensors],
        "neighbors": [_state(s) for s in obs.neighbors],
    }
    if obs.comms is not None:
        d["comms"] = _comms(obs.comms)
    if obs.epoch is not None:
        d["epoch"] = _epoch(obs.epoch)
    return d


def to_proto_message(obs: Observation) -> Any:
    """Build the Cap'n Proto message for an observation (not yet serialized)."""
    return _schema().Observation.new_message(**_observation_dict(obs))


def to_bytes(obs: Observation) -> bytes:
    """Encode an observation to its canonical Cap'n Proto wire bytes."""
    data: bytes = to_proto_message(obs).to_bytes()
    return data


# --- decode ----------------------------------------------------------------------


def _r_vec3(r: Any) -> Vec3:
    return Vec3(x=r.x, y=r.y, z=r.z)


def _r_transform(r: Any) -> Transform:
    return Transform(
        translation_m=_r_vec3(r.translationM),
        rotation_quat_xyzw=Quat(
            x=r.rotationQuatXyzw.x,
            y=r.rotationQuatXyzw.y,
            z=r.rotationQuatXyzw.z,
            w=r.rotationQuatXyzw.w,
        ),
    )


def _r_frame(r: Any) -> ReferenceFrame:
    return ReferenceFrame(
        name=r.name,
        frame_class=FrameClass(r.frameClass),
        center=r.center if r._has("center") else None,
    )


def _r_epoch(r: Any) -> Epoch:
    return Epoch(tdb_seconds=r.tdbSeconds, scale=TimeScale(r.scale))


def _r_state(r: Any) -> StateSample:
    return StateSample(
        agent_id=r.agentId,
        frame=_r_frame(r.frame),
        pose=_r_transform(r.pose),
        linear_velocity_mps=_r_vec3(r.linearVelocityMps) if r._has("linearVelocityMps") else None,
        angular_velocity_rps=_r_vec3(r.angularVelocityRps)
        if r._has("angularVelocityRps")
        else None,
        battery_soc_j=r.batterySocJ if r.hasBatterySocJ else None,
        temperature_k=r.temperatureK if r.hasTemperatureK else None,
        mode=r.mode if r._has("mode") else None,
    )


def _r_sensor(r: Any) -> SensorReading:
    return SensorReading(
        sensor=r.sensor,
        values=list(r.values),
        unit=r.unit if r._has("unit") else None,
        resource_species=r.resourceSpecies if r._has("resourceSpecies") else None,
        noise_sigma=r.noiseSigma if r.hasNoiseSigma else None,
        valid=r.valid,
    )


def _r_peer(r: Any) -> PeerLink:
    return PeerLink(
        peer=r.peer,
        reachable=r.reachable,
        rate_bps=r.rateBps if r.hasRateBps else None,
        latency_s=r.latencyS if r.hasLatencyS else None,
        margin_db=r.marginDb if r.hasMarginDb else None,
    )


def _r_comms(r: Any) -> CommsObservationMask:
    return CommsObservationMask(
        agent_id=r.agentId,
        links=[_r_peer(p) for p in r.links],
        earth_contact=r.earthContact,
    )


def _from_reader(r: Any) -> Observation:
    return Observation(
        tick=r.tick,
        sim_time_s=r.simTimeS,
        agent_id=r.agentId,
        observable=r.observable,
        self_state=_r_state(r.selfState),
        sensors=[_r_sensor(s) for s in r.sensors],
        comms=_r_comms(r.comms) if r._has("comms") else None,
        neighbors=[_r_state(s) for s in r.neighbors],
        epoch=_r_epoch(r.epoch) if r._has("epoch") else None,
    )


def from_bytes(data: bytes) -> Observation:
    """Decode Cap'n Proto wire bytes into a typed :class:`Observation`."""
    with _schema().Observation.from_bytes(data) as r:
        return _from_reader(r)


def reader(data: bytes) -> Any:
    """Return a zero-copy Cap'n Proto reader over ``data`` for hot-path field access
    without materializing the Python model (use as a context manager)."""
    return _schema().Observation.from_bytes(data)
