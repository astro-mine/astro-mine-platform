"""RM-P0-SIM-09 — headless + interactive MCAP recording with provenance stamping.

Proves the deliverables and acceptance criteria (sim.md §3, §5):

- a headless run emits a well-formed MCAP that reads back (the file Bench scores), recording every
  per-tick frame, the seed, every input content hash, and every engine version/tier;
- the provenance is sufficient to reproduce the run byte-for-byte — the recording carries the same
  ``Trace.content_hash`` the determinism gate (RM-P0-SIM-10) compares, and recording never perturbs
  it;
- headless and interactive are one runtime — the interactive ``on_frame`` sink sees exactly the
  frames the batch trace holds, from the same stepping loop.
"""

from __future__ import annotations

from pathlib import Path

from astro_mine.core.units import INERTIAL_J2000
from astro_mine.sim.recording import (
    FRAMES_TOPIC,
    PROVENANCE_ATTACHMENT,
    open_recording,
    read_recording,
    record_episode,
    run_provenance,
)
from astro_mine.sim.runtime import AgentSpec, OrbitalDynamics, Scenario, run_episode


def _scenario(name: str = "rec", seed: int = 7) -> Scenario:
    return Scenario(
        name=name,
        agents=(
            AgentSpec(agent_id="rover", velocity_mps=(1.0, 0.0, 0.0), battery_soc_j=100.0),
            AgentSpec(
                agent_id="relay",
                initial_position_m=(1_837_400.0, 0.0, 0.0),
                velocity_mps=(0.0, 1633.0, 0.0),
                battery_soc_j=5000.0,
                frame=INERTIAL_J2000,
                dynamics=OrbitalDynamics(),
            ),
        ),
        seed=seed,
        horizon_steps=6,
    )


# --- round-trip + Bench-readable MCAP ---------------------------------------------------------


def test_record_episode_emits_a_readable_mcap(tmp_path: Path) -> None:
    path = tmp_path / "run.mcap"
    trace = record_episode(_scenario(), path)
    assert path.stat().st_size > 0  # a real file on disk

    recording = read_recording(path)
    assert recording.frames == trace.frames  # every per-tick frame recovered in order
    assert len(recording.frames) == trace.frames.__len__() == 6 + 1  # reset + one per step
    assert recording.content_hash == trace.content_hash


def test_recording_does_not_perturb_determinism(tmp_path: Path) -> None:
    recorded = record_episode(_scenario(), tmp_path / "a.mcap")
    plain = run_episode(_scenario())
    assert recorded.content_hash == plain.content_hash  # the sink is observe-only


# --- the provenance envelope: every input hash, engine version/tier, seed (acceptance) --------


def test_provenance_records_inputs_engines_and_seed(tmp_path: Path) -> None:
    path = tmp_path / "run.mcap"
    record_episode(_scenario(seed=11), path)
    run = read_recording(path).provenance["run"]

    assert run["seed"] == 11
    # every input is content-addressed (here: the scenario)
    assert run["source_content_hashes"]["scenario"]
    # every engine the run routed through, by version — read off the engines that **actually ran**,
    # not off the kinds the scenario declared (#65). This assertion used to hold by coincidence:
    # `engine_versions` was derived from `AgentSpec.dynamics.kind`, while a single `KinematicEngine`
    # stepped every agent, so the recording named an orbital engine that never executed. The rover
    # is kinematic and the relay orbital, and now both engines are really built.
    assert run["engine_versions"] == {
        "astro-mine.sim.kinematic": "0.1.0",
        "astro-mine.sim.orbital": "0.1.0",
    }
    # ...and its tier + error-budget outcome (the scheduler's fidelity selection)
    relay_fidelity = run["fidelity"]["relay"]
    assert relay_fidelity["tier"] and "implied_error_rungs" in relay_fidelity


def test_changing_any_input_changes_the_recorded_hashes(tmp_path: Path) -> None:
    a = record_episode(_scenario(name="one"), tmp_path / "a.mcap")
    b = record_episode(_scenario(name="two"), tmp_path / "b.mcap")
    ra, rb = read_recording(tmp_path / "a.mcap"), read_recording(tmp_path / "b.mcap")
    assert ra.content_hash != rb.content_hash
    assert (
        ra.provenance["run"]["source_content_hashes"]["scenario"]
        != rb.provenance["run"]["source_content_hashes"]["scenario"]
    )
    assert a.content_hash != b.content_hash


def test_provenance_carries_the_trace_content_hash(tmp_path: Path) -> None:
    path = tmp_path / "run.mcap"
    trace = record_episode(_scenario(), path)
    envelope = read_recording(path).provenance
    assert envelope["content_hash"] == trace.content_hash  # one canonical artifact, not two


def test_resource_field_hash_is_stamped_when_present(tmp_path: Path) -> None:
    class _Field:
        content_hash = "deadbeef"

        def mean(self, position: object, *, epoch: object = None) -> float:
            return 0.0

        def variance(self, position: object, *, epoch: object = None) -> float:
            return 1.0

        def quantile(self, position: object, q: float, *, epoch: object = None) -> float:
            return 0.0

        def sample(self, position: object, **_: object) -> tuple[float, ...]:
            return (0.0,)

    sc = Scenario(name="rf", agents=(AgentSpec(agent_id="r1"),), horizon_steps=2)
    trace = record_episode(sc, tmp_path / "rf.mcap", resource_field=_Field())  # type: ignore[arg-type]
    assert trace.provenance["source_content_hashes"]["resource_field"] == "deadbeef"


# --- headless and interactive are one runtime -------------------------------------------------


def test_interactive_sink_sees_exactly_the_trace_frames(tmp_path: Path) -> None:
    seen: list[dict[str, object]] = []
    trace = record_episode(_scenario(), tmp_path / "run.mcap", on_frame=seen.append)
    assert tuple(seen) == trace.frames  # same loop, the live sink and the batch trace agree


def test_seed_override_is_recorded(tmp_path: Path) -> None:
    trace = record_episode(_scenario(seed=1), tmp_path / "run.mcap", seed=99)
    assert trace.seed == 99
    assert read_recording(tmp_path / "run.mcap").provenance["run"]["seed"] == 99


# --- envelope construction + lower-level writer -----------------------------------------------


def test_run_provenance_uses_an_explicit_environment() -> None:
    trace = run_episode(_scenario())
    envelope = run_provenance(trace, environment={"sim_version": "1.2.3"})
    assert envelope["environment"] == {"sim_version": "1.2.3"}
    # default path stamps the live interpreter fingerprint
    assert set(run_provenance(trace)["environment"]) == {"sim_version", "python", "platform"}


def test_recording_without_provenance_yields_an_empty_envelope(tmp_path: Path) -> None:
    path = tmp_path / "bare.mcap"
    with open_recording(path) as recording:
        recording.write_frame({"kind": "reset", "observations": {}})  # frame, no provenance stamp
    result = read_recording(path)
    assert result.content_hash == ""
    assert result.provenance == {}
    assert len(result.frames) == 1


def test_recording_topic_and_attachment_names_are_stable() -> None:
    # Bench reads these by name; keep them as published constants.
    assert FRAMES_TOPIC == "/sim/frames"
    assert PROVENANCE_ATTACHMENT == "provenance.json"
