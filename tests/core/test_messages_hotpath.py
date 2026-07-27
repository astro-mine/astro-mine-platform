"""Hot-path observation tests (RM-P0-CORE-04): Cap'n Proto zero-copy decode.

Covers the acceptance criterion: a per-tick sensor/telemetry payload decodes zero-copy
via the chosen Cap'n Proto schema. Also exercises exact round-trip and the None-vs-value
presence handling (optional primitives via `has*` flags; optional pointers via presence).
"""

from __future__ import annotations

from astro_mine.core.messages import hotpath
from astro_mine.core.messages import model as m
from astro_mine.core.units import MOON_BODY_FIXED, Epoch, TimeScale


def _full_observation() -> m.Observation:
    return m.Observation(
        tick=42,
        sim_time_s=12.5,
        agent_id="rover-1",
        observable=True,
        self_state=m.StateSample(
            agent_id="rover-1",
            frame=MOON_BODY_FIXED,
            pose=m.Transform(
                translation_m=m.Vec3(x=1.0, y=2.0, z=3.0),
                rotation_quat_xyzw=m.Quat(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
            linear_velocity_mps=m.Vec3(x=0.1, y=0.0, z=0.0),
            angular_velocity_rps=m.Vec3(x=0.0, y=0.0, z=0.05),
            battery_soc_j=3.0e6,
            temperature_k=240.0,
            mode="prospect",
        ),
        sensors=[
            m.SensorReading(
                sensor="neutron",
                values=[0.08, 0.11],
                unit="mass_fraction",
                resource_species="water_equivalent_hydrogen",
                noise_sigma=0.08,
                valid=True,
            ),
            m.SensorReading(sensor="imu", values=[]),
        ],
        comms=m.CommsObservationMask(
            agent_id="rover-1",
            links=[m.PeerLink(peer="relay-1", reachable=True, rate_bps=2.0e6, latency_s=1.3)],
            earth_contact=False,
        ),
        neighbors=[
            m.StateSample(
                agent_id="rover-2",
                frame=MOON_BODY_FIXED,
                pose=m.Transform(
                    translation_m=m.Vec3(x=5.0, y=0.0, z=0.0),
                    rotation_quat_xyzw=m.Quat(x=0.0, y=0.0, z=0.0, w=1.0),
                ),
            )
        ],
        epoch=Epoch(tdb_seconds=8.1e8, scale=TimeScale.TDB),
    )


def _minimal_observation() -> m.Observation:
    """Partial observation — most optionals absent (comms-denied, no telemetry)."""
    return m.Observation(
        tick=1,
        sim_time_s=0.0,
        agent_id="rover-9",
        observable=False,
        self_state=m.StateSample(
            agent_id="rover-9",
            frame=MOON_BODY_FIXED,
            pose=m.Transform(
                translation_m=m.Vec3(x=0.0, y=0.0, z=0.0),
                rotation_quat_xyzw=m.Quat(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        ),
    )


def test_full_observation_roundtrips_exactly() -> None:
    obs = _full_observation()
    data = hotpath.to_bytes(obs)
    assert hotpath.from_bytes(data) == obs


def test_minimal_observation_preserves_none_vs_value() -> None:
    """Optional primitives (battery/temperature/epoch) and pointers (velocity/mode/comms)
    that are None stay None across the round-trip — not collapsed to 0.0/empty."""
    obs = _minimal_observation()
    restored = hotpath.from_bytes(hotpath.to_bytes(obs))
    assert restored == obs
    assert restored.self_state.battery_soc_j is None
    assert restored.self_state.temperature_k is None
    assert restored.self_state.linear_velocity_mps is None
    assert restored.self_state.mode is None
    assert restored.comms is None
    assert restored.epoch is None


def test_zero_copy_decode_reads_nested_fields() -> None:
    """Acceptance: the hot-path payload decodes zero-copy — nested fields are readable
    straight from the reader without materializing the Python model."""
    obs = _full_observation()
    data = hotpath.to_bytes(obs)
    with hotpath.reader(data) as r:
        assert r.agentId == "rover-1"
        assert r.tick == 42
        # nested list + scalar access without building the Pydantic model
        assert r.sensors[0].sensor == "neutron"
        assert list(r.sensors[0].values) == [0.08, 0.11]
        assert r.selfState.batterySocJ == 3.0e6
        # typed frame/epoch decode as nested structs (RM-P0-CORE-06)
        assert r.selfState.frame.name == "MOON_ME"
        assert r.selfState.frame.frameClass == "body_fixed"
        assert r.epoch.tdbSeconds == 8.1e8
        assert r.epoch.scale == "tdb"
        assert r.comms.links[0].peer == "relay-1"
        assert r.neighbors[0].agentId == "rover-2"


def test_encoding_is_stable() -> None:
    obs = _full_observation()
    assert hotpath.to_bytes(obs) == hotpath.to_bytes(hotpath.from_bytes(hotpath.to_bytes(obs)))
