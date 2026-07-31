"""The MCAP → EpisodeTrace decoder (RM-P0-BENCH-07).

Decodes a **golden real-Sim** recording (`tests/data/`, produced offline by astro-mine-sim — Bench
never imports Sim) into a scorable :class:`EpisodeTrace`, and covers the provenance/seed extraction,
the default scoring context, and the error paths — the latter built with `mcap` directly, Sim-free.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from mcap.writer import Writer

from astro_mine.bench.metrics import REFERENCE_METRICS, score
from astro_mine.bench.recording import (
    FRAMES_TOPIC,
    DecodedRecording,
    RecordingError,
    decode_recording,
)
from astro_mine.core.scoring import ScoringContext

from ._factories import make_observation

_GOLDEN = Path(__file__).parent / "data" / "anchor-lunar-polar-ice-prospecting-v1.mcap"

ANCHOR_METRICS = frozenset(
    {
        "water_mass",
        "energy_per_kg",
        "information_gain",
        "psr_area_characterized",
        "nights_survived",
        "comms_robustness",
        "discovery_latency",
    }
)


def _write_mcap(
    path: Path, *, topic: str, frames: list[dict[str, object]], provenance: dict[str, object] | None
) -> None:
    """Write a minimal MCAP (a synthetic Sim-shaped recording) for the error-path tests."""
    with path.open("wb") as handle:
        writer = Writer(handle)
        writer.start(profile="astro-mine-sim", library="test")
        schema_id = writer.register_schema(name="frame", encoding="jsonschema", data=b"{}")
        channel_id = writer.register_channel(
            topic=topic, message_encoding="json", schema_id=schema_id
        )
        for sequence, frame in enumerate(frames):
            writer.add_message(
                channel_id=channel_id,
                log_time=sequence,
                data=json.dumps(frame).encode(),
                publish_time=sequence,
                sequence=sequence,
            )
        if provenance is not None:
            writer.add_attachment(
                create_time=0,
                log_time=0,
                name="provenance.json",
                media_type="application/json",
                data=json.dumps(provenance).encode(),
            )
        writer.finish()


def _frame(agent_id: str = "rover") -> dict[str, object]:
    observation = make_observation(0, 0.0, agent_id=agent_id, water_kg=1.0)
    return {"kind": "step", "observations": {agent_id: observation.model_dump(mode="json")}}


# --- the golden real-Sim fixture ----------------------------------------------------------------


def test_decode_golden_recording() -> None:
    decoded = decode_recording(_GOLDEN)
    assert isinstance(decoded, DecodedRecording)
    assert decoded.seed == 1001
    assert decoded.content_hash.startswith("05da9e79")  # provenance content hash, pinned by fixture
    observations = decoded.trace.observations
    assert observations  # reset + per-step, flattened across agents
    assert {o.agent_id for o in observations} == {"rover", "relay"}
    species = {r.resource_species for o in observations for r in o.sensors}
    assert "water_equivalent_hydrogen" in species  # the neutron discovery channel
    assert any(o.comms is not None for o in observations)  # comms mask channel
    assert any(o.self_state.battery_soc_j is not None for o in observations)  # survival channel


def test_decoded_trace_is_scorable() -> None:
    context = ScoringContext(
        discovery_species="water_equivalent_hydrogen",
        discovery_threshold=0.05,
        night_intervals=((0.0, 900.0),),
        survivable_temperature_k=100.0,
    )
    decoded = decode_recording(_GOLDEN, context=context)
    card = score(
        {decoded.seed or 0: decoded.trace},
        list(REFERENCE_METRICS),
        scenario_id="lunar-polar-ice-prospecting-v1",
        runner="mcap-recording/0.1.0",
    )
    assert {m.metric for m in card.metrics} == ANCHOR_METRICS
    by_metric = {m.metric: m for m in card.metrics}
    assert by_metric["discovery_latency"].value is not None  # neutron detection decoded + scored
    assert by_metric["nights_survived"].value == 1.0
    assert 0.0 < by_metric["comms_robustness"].value <= 1.0  # type: ignore[operator]


def test_decode_defaults_to_an_empty_context() -> None:
    assert decode_recording(_GOLDEN).trace.context == ScoringContext()


# --- provenance + error paths (synthetic MCAPs, no Sim) -----------------------------------------


def test_absent_provenance_yields_no_seed(tmp_path: Path) -> None:
    path = tmp_path / "no-prov.mcap"
    _write_mcap(path, topic=FRAMES_TOPIC, frames=[_frame()], provenance=None)
    decoded = decode_recording(path)
    assert decoded.seed is None
    assert decoded.content_hash == ""
    assert len(decoded.trace.observations) == 1


def test_provenance_without_seed_yields_no_seed(tmp_path: Path) -> None:
    path = tmp_path / "no-seed.mcap"
    _write_mcap(
        path,
        topic=FRAMES_TOPIC,
        frames=[_frame()],
        provenance={"content_hash": "sha256:x", "run": {}},
    )
    decoded = decode_recording(path)
    assert decoded.seed is None
    assert decoded.content_hash == "sha256:x"


def test_missing_frames_channel_raises(tmp_path: Path) -> None:
    path = tmp_path / "wrong-topic.mcap"
    _write_mcap(path, topic="/other/topic", frames=[_frame()], provenance=None)
    with pytest.raises(RecordingError, match="not a Sim recording"):
        decode_recording(path)


def test_empty_recording_raises(tmp_path: Path) -> None:
    path = tmp_path / "empty.mcap"
    _write_mcap(path, topic=FRAMES_TOPIC, frames=[], provenance=None)
    with pytest.raises(RecordingError):
        decode_recording(path)
