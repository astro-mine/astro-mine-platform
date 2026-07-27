"""Wall-clock instrumentation — the real-time factor, and its exclusion from every hash (#51).

sim.md §10 lists "step wall-clock vs. sim-time (real-time factor)" as an intended observability
signal; before this it did not exist anywhere in Sim (every ``elapsed_s`` in the codebase is *sim*
time). It is the numerator of the surrogate speedup claim (surrogate.md §8; ``LUNAR-TR-002``).

The load-bearing property, asserted from several directions here: **timing never enters a content
hash**. Wall-clock is non-deterministic by nature, so a reproducibility key that moved with it would
fail the RM-P0-SIM-10 determinism gate on every run, on every machine. Bench's ``EpisodeRunner``
contract states the same rule from the other side — a conforming runner is a pure function of
``(scenario_hash, policy, seed)`` with no wall-clock.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from astro_mine.core.sadf.enums import FidelityTier
from astro_mine.sim.engines import RegimeEngine, kinematic_engine_factory
from astro_mine.sim.recording import read_recording, record_episode, run_provenance
from astro_mine.sim.runtime import (
    AgentSpec,
    EngineTiming,
    Scenario,
    TimedEngine,
    TimingRecorder,
    run_episode,
    timed_engine_factory,
)
from astro_mine.sim.runtime.rng import RngStreams


def _scenario(name: str = "timed", seed: int = 7, horizon: int = 6) -> Scenario:
    return Scenario(
        name=name,
        agents=(
            AgentSpec(agent_id="rover", velocity_mps=(1.0, 0.0, 0.0), battery_soc_j=100.0),
            AgentSpec(agent_id="hauler", velocity_mps=(0.0, 0.5, 0.0), battery_soc_j=100.0),
        ),
        seed=seed,
        horizon_steps=horizon,
    )


# --- the measurement --------------------------------------------------------------


def test_a_run_reports_its_wall_clock_and_real_time_factor() -> None:
    """The signal sim.md §10 asks for: modeled seconds vs. the wall-clock seconds spent on them."""
    scenario = _scenario(horizon=6)

    trace = run_episode(scenario)

    assert trace.timing is not None
    assert trace.timing["steps"] == 6  # one advance per step, none for the reset
    assert trace.timing["advance_wall_clock_s"] > 0.0
    assert trace.timing["sim_time_s"] == pytest.approx(6.0)  # 6 steps x 1 s
    # sim-seconds per wall-clock second: a reduced-order kinematic step easily beats real time.
    assert trace.timing["real_time_factor"] > 1.0
    assert trace.timing["mean_step_wall_clock_s"] == pytest.approx(
        trace.timing["advance_wall_clock_s"] / 6
    )


def test_the_timing_names_the_engine_and_the_tier_that_spent_it() -> None:
    """Per-*tier* wall-clock: the ratio of two tiers' numbers is the whole speedup claim, so a
    measurement that did not say which tier produced it would be unusable."""
    trace = run_episode(_scenario())

    assert trace.timing is not None
    assert trace.timing["tier"] == FidelityTier.KINEMATIC.value
    assert trace.timing["engine"]  # the engine's own declared name, not a label we invented here


def test_a_caller_supplied_recorder_keeps_the_handle() -> None:
    """The seam the speedup runner needs: comparing two tiers means holding both runs' numbers."""
    recorder = TimingRecorder()

    run_episode(_scenario(), timing=recorder)

    assert recorder.snapshot().steps == 6
    assert recorder.snapshot().advance_wall_clock_s > 0.0


# --- and it never touches a hash ---------------------------------------------------


def test_timing_is_not_serialized_into_the_determinism_key() -> None:
    """The canonical JSON enumerates its fields rather than reflecting off the dataclass, so a
    non-deterministic field cannot leak into the key merely by being added to the class."""
    trace = run_episode(_scenario())

    payload = json.loads(trace.to_canonical_json())
    assert "timing" not in payload
    assert set(payload) == {"provenance", "scenario", "seed", "frames"}


def test_a_differing_wall_clock_moves_neither_the_hash_nor_a_trace_s_identity() -> None:
    """The property the RM-P0-SIM-10 gate depends on, forced rather than hoped for.

    Two real runs *should* differ in wall-clock, but asserting that they do would make this test a
    coin-flip on a coarse clock. So the timings are substituted directly: same physics, wildly
    different durations, and neither the content hash nor dataclass equality may notice."""
    trace = run_episode(_scenario())
    slow = replace(trace, timing={"steps": 6, "advance_wall_clock_s": 900.0})
    fast = replace(trace, timing={"steps": 6, "advance_wall_clock_s": 0.001})

    assert slow.content_hash == fast.content_hash == trace.content_hash
    assert slow == fast  # `timing` is compare=False: it is not part of a trace's identity


def test_two_real_runs_reproduce_byte_for_byte_while_each_is_timed() -> None:
    """And the same holds end to end on the real thing: every run is timed, and every run still
    reproduces."""
    first = run_episode(_scenario())
    second = run_episode(_scenario())

    assert first.timing is not None and second.timing is not None
    assert first.content_hash == second.content_hash


def test_the_mcap_envelope_carries_timing_beside_the_hash_not_inside_it(tmp_path: Path) -> None:
    """Recorded for audit, excluded from the key — the same treatment the environment stamp gets,
    for the same reason."""
    path = tmp_path / "timed.mcap"

    trace = record_episode(_scenario(), path)

    envelope = run_provenance(trace)
    assert envelope["timing"] is not None
    assert envelope["timing"]["steps"] == 6
    # ... and the recorded hash is still the trace's own, untouched by the timing beside it.
    assert envelope["content_hash"] == trace.content_hash
    assert read_recording(path).content_hash == trace.content_hash
    # The deterministic run envelope stays free of it: `run` is what rides *inside* the hash.
    assert "timing" not in envelope["run"]


# --- the wrapper ------------------------------------------------------------------


def test_the_timed_engine_is_structurally_a_regime_engine() -> None:
    """It wraps by delegation, so it composes with any engine — reduced-order, DEM, surrogate, or
    the multi-domain coupler — without either side knowing."""
    inner = kinematic_engine_factory(_scenario(), RngStreams(0))

    timed = TimedEngine(inner, TimingRecorder())

    assert isinstance(timed, RegimeEngine)
    assert timed.inner is inner
    assert timed.descriptor == inner.descriptor


def test_the_wrapper_times_advance_and_leaves_the_physics_alone() -> None:
    """The instrument must not move what it measures: a timed run and an untimed one step the same
    engine to the same state."""
    scenario = _scenario()
    recorder = TimingRecorder()

    bare = run_episode(scenario, engine_factory=kinematic_engine_factory)
    wrapped = run_episode(
        scenario,
        engine_factory=timed_engine_factory(kinematic_engine_factory, recorder),
    )

    assert bare.content_hash == wrapped.content_hash  # identical physics, identical frames
    assert recorder.snapshot().steps == 6


def test_a_reset_starts_a_fresh_episode_and_banks_the_last() -> None:
    """The stepping core rebuilds the engine on every reset, so the recorder rewinds per episode —
    a re-run must measure *that* run, not the sum of every run before it."""
    recorder = TimingRecorder()

    run_episode(_scenario(), timing=recorder)
    first = recorder.snapshot()
    run_episode(_scenario(), timing=recorder)
    second = recorder.snapshot()

    assert first.steps == second.steps == 6  # not 12: the second run rewound
    assert len(recorder.episodes) == 1  # the first run, banked when the second rewound


# --- the honest-zero cases ---------------------------------------------------------


def test_an_unmeasurable_duration_reports_no_rate_rather_than_infinity() -> None:
    """A clock too coarse to see the work is an *absent* measurement, not an infinitely fast one."""
    zero = EngineTiming(
        engine="e", tier="kinematic", steps=0, advance_wall_clock_s=0.0, sim_time_s=0.0
    )

    assert zero.real_time_factor is None
    assert zero.mean_step_wall_clock_s is None
    assert TimingRecorder().as_provenance() is None  # nothing ran; nothing to report
