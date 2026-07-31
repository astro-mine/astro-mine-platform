"""The reproducibility harness + determinism gate (RM-P0-BENCH-04).

Covers the content-addressed Result/provenance model, the deterministic reference runner, lockfile
pinning, the reproduce/gate/replay API, and the gate's teeth: a non-deterministic runner must fail.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path

import pytest

from astro_mine.bench.harness import (
    LOCKFILE_ENV,
    REFERENCE_RUNNER_ID,
    DeterminismError,
    EnvironmentStamp,
    LockfileNotFound,
    Result,
    assert_reproducible,
    build_toolchain,
    environment_stamp,
    lockfile_digest,
    reference_runner,
    replay,
    reproduce,
    resolve_lockfile,
)
from astro_mine.bench.scenario import ResolvedScenario, ScenarioSpec, resolve_scenario
from astro_mine.bench.zoo import ANCHOR_SCENARIO_ID, load_scenario
from astro_mine.core.scoring import RunOutcome

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


class _FlakyRunner:
    """A deliberately non-deterministic runner: a fresh score on every call (trips the gate)."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, resolved: ResolvedScenario, seed: int) -> RunOutcome:
        self.calls += 1
        return RunOutcome(
            determinism_key=f"sha256:{self.calls:064d}", metrics={"m": float(self.calls)}
        )


def _constant_runner(resolved: ResolvedScenario, seed: int) -> RunOutcome:
    """A deterministic runner whose identity is derived from its function name."""
    return RunOutcome(determinism_key="sha256:" + "ab" * 32, metrics={"score": 1.0})


@pytest.fixture(scope="module")
def anchor() -> ScenarioSpec:
    return load_scenario(ANCHOR_SCENARIO_ID)


# --- reference runner ---------------------------------------------------------------------------


def test_reference_runner_is_deterministic_and_seed_sensitive(anchor: ScenarioSpec) -> None:
    resolved = resolve_scenario(anchor)
    first = reference_runner(resolved, 1001)
    assert reference_runner(resolved, 1001) == first  # same inputs -> byte-identical
    assert reference_runner(resolved, 1002).determinism_key != first.determinism_key
    assert set(first.metrics) == ANCHOR_METRICS  # one score per pinned metric
    assert all(0.0 <= v < 1.0 for v in first.metrics.values())


# --- lockfile / toolchain / environment ---------------------------------------------------------


def test_lockfile_digest_pins_the_lockfile(tmp_path: Path) -> None:
    real = lockfile_digest()
    assert real.startswith("sha256:")
    other = tmp_path / "uv.lock"
    other.write_text("different contents")
    assert lockfile_digest(other) != real  # content change -> different digest


def test_lockfile_digest_turns_on_contents_not_path(tmp_path: Path) -> None:
    """A consumer's lockfile digests the same wherever on disk it lives."""
    here = tmp_path / "a" / "uv.lock"
    there = tmp_path / "b" / "renamed.lock"
    for lockfile in (here, there):
        lockfile.parent.mkdir(parents=True)
        lockfile.write_text("the same pinned environment")
    assert lockfile_digest(here) == lockfile_digest(there)


def test_build_toolchain_carries_pinned_inputs() -> None:
    toolchain = build_toolchain("reference-runner/0.1.0")
    assert toolchain["lockfile"] == lockfile_digest()
    assert toolchain["runner"] == "reference-runner/0.1.0"
    assert "bench" in toolchain


def test_resolve_lockfile_prefers_an_explicit_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    explicit = tmp_path / "explicit.lock"
    explicit.write_text("pinned by the caller")
    from_env = tmp_path / "env.lock"
    from_env.write_text("pinned by the environment")
    monkeypatch.setenv(LOCKFILE_ENV, str(from_env))

    assert resolve_lockfile(explicit) == explicit  # the caller outranks the environment


def test_resolve_lockfile_falls_back_to_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from_env = tmp_path / "env.lock"
    from_env.write_text("pinned by the environment")
    monkeypatch.setenv(LOCKFILE_ENV, str(from_env))

    assert resolve_lockfile(start=tmp_path) == from_env  # outranks directory discovery


def test_resolve_lockfile_discovers_the_benchmarked_projects_lockfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An installed Bench pins its *consumer's* environment, found upward from the working dir.

    This is what a Sim-backed run needs: the dependency set that produced the physics is Sim's, not
    Bench's, so the lockfile is discovered from the project being benchmarked.
    """
    monkeypatch.delenv(LOCKFILE_ENV, raising=False)
    nested = tmp_path / "project" / "src" / "deep"
    nested.mkdir(parents=True)
    lockfile = tmp_path / "project" / "uv.lock"
    lockfile.write_text("the consumer's pinned environment")

    assert resolve_lockfile(start=nested) == lockfile


def test_resolve_lockfile_fails_closed_when_nothing_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unpinned environment cannot be attested, so the gate refuses rather than skipping."""
    monkeypatch.delenv(LOCKFILE_ENV, raising=False)
    with pytest.raises(LockfileNotFound, match=LOCKFILE_ENV):  # names the way out
        resolve_lockfile(start=tmp_path)


def test_resolve_lockfile_rejects_a_path_that_does_not_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(LOCKFILE_ENV, str(tmp_path / "absent.lock"))
    with pytest.raises(LockfileNotFound):
        resolve_lockfile()  # a dangling override is an error, not a fall-through to discovery
    with pytest.raises(LockfileNotFound):
        resolve_lockfile(tmp_path / "absent.lock")


def test_environment_stamp_records_the_machine() -> None:
    stamp = environment_stamp()
    assert stamp.python == platform.python_version()
    assert stamp.platform


# --- reproduce / gate ---------------------------------------------------------------------------


def test_anchor_reproduces_byte_for_byte(anchor: ScenarioSpec) -> None:
    report = reproduce(anchor)
    assert report.reproducible
    assert report.runs == 2
    assert len(set(report.result_hashes)) == 1
    assert report.result_hash == report.result.result_hash


def test_gate_returns_the_canonical_result(anchor: ScenarioSpec) -> None:
    result = assert_reproducible(anchor)
    assert result.scenario_id == ANCHOR_SCENARIO_ID
    assert result.runner == REFERENCE_RUNNER_ID
    assert tuple(s.seed for s in result.per_seed) == anchor.seeds.public
    assert len(result.content_hashes) == 9  # world + 6 fleet + 1 prospect + 1 link
    assert set(result.aggregate) == ANCHOR_METRICS


def test_reproduce_needs_two_runs(anchor: ScenarioSpec) -> None:
    with pytest.raises(ValueError, match="runs >= 2"):
        reproduce(anchor, runs=1)


def test_custom_runner_identity_from_name(anchor: ScenarioSpec) -> None:
    result = assert_reproducible(anchor, _constant_runner)
    assert result.runner == "_constant_runner"


def test_explicit_runner_id_is_recorded(anchor: ScenarioSpec) -> None:
    result = assert_reproducible(anchor, _constant_runner, runner_id="custom/2.0")
    assert result.runner == "custom/2.0"


def test_nondeterministic_runner_trips_the_gate(anchor: ScenarioSpec) -> None:
    report = reproduce(anchor, _FlakyRunner(), runner_id="flaky")
    assert not report.reproducible
    assert len(set(report.result_hashes)) == 2
    with pytest.raises(DeterminismError, match="did not reproduce"):
        assert_reproducible(anchor, _FlakyRunner(), runner_id="flaky")


# --- result hashing -----------------------------------------------------------------------------


def test_result_hash_excludes_the_environment_stamp(anchor: ScenarioSpec) -> None:
    result = assert_reproducible(anchor)
    relocated = result.model_copy(
        update={"environment": EnvironmentStamp(python="9.9.9", platform="other-os")}
    )
    assert relocated.result_hash == result.result_hash  # env stamp is not part of the key


def test_result_hash_changes_with_inputs(anchor: ScenarioSpec) -> None:
    result = assert_reproducible(anchor)
    bumped = result.model_copy(
        update={"aggregate": {**result.aggregate, "water_mass": result.aggregate["water_mass"] + 1}}
    )
    assert bumped.result_hash != result.result_hash


# --- replay (sampled re-execution) --------------------------------------------------------------


def test_replay_matches_a_stored_result(anchor: ScenarioSpec) -> None:
    result = assert_reproducible(anchor)
    assert replay(result) is True


def test_replay_detects_nondeterminism(anchor: ScenarioSpec) -> None:
    result = assert_reproducible(anchor)
    assert replay(result, _FlakyRunner()) is False


# --- the gate under an installed Bench ----------------------------------------------------------


def test_the_gate_pins_a_caller_supplied_lockfile(anchor: ScenarioSpec, tmp_path: Path) -> None:
    """The seam an installed Bench needs: pin the *consumer's* environment, not Bench's own.

    Without this, a consumer had no way to say which lockfile pinned the run except by patching a
    module global — which is exactly what astro-mine-sim was reduced to doing.
    """
    consumer = tmp_path / "uv.lock"
    consumer.write_text("the consumer's pinned environment")

    result = assert_reproducible(anchor, lockfile=consumer)

    assert result.environment_lockfile == lockfile_digest(consumer)
    assert result.environment_lockfile != lockfile_digest()  # not Bench's own lockfile


#: Run the determinism gate in a fresh interpreter, printing how the lockfile resolved.
_GATE = """
from astro_mine.bench.harness import LockfileNotFound, assert_reproducible
from astro_mine.bench.zoo import ANCHOR_SCENARIO_ID, load_scenario

try:
    result = assert_reproducible(load_scenario(ANCHOR_SCENARIO_ID))
except LockfileNotFound as exc:
    print("FAILED_CLOSED:", exc)
else:
    print("PINNED:", result.environment_lockfile)
"""


def test_the_gate_does_not_resolve_the_lockfile_from_benchs_install_location(
    tmp_path: Path,
) -> None:
    """Run the gate from a directory with no ``uv.lock`` above it — an installed Bench's shape.

    Bench used to derive the lockfile from its own ``__file__``, so from site-packages it walked up
    into the interpreter's lib directory and raised ``FileNotFoundError`` — the gate had never
    actually run from an installed Bench (bench.md §7 requires that tier to work). Resolution now
    ignores where Bench lives: with no lockfile in sight the gate fails closed with an actionable
    message, and naming one makes it pass — from any working directory.
    """
    workdir = tmp_path / "elsewhere"  # pytest's tmp tree has no uv.lock above it
    workdir.mkdir()
    lockfile = tmp_path / "consumer.lock"
    lockfile.write_text("the consumer's pinned environment")

    def run_gate(**overrides: str) -> str:
        env = {key: value for key, value in os.environ.items() if key != LOCKFILE_ENV}
        completed = subprocess.run(
            [sys.executable, "-c", _GATE],
            cwd=workdir,
            env={**env, **overrides},
            capture_output=True,
            text=True,
            check=True,
        )
        return completed.stdout.strip()

    unpinned = run_gate()
    assert unpinned.startswith("FAILED_CLOSED:")
    assert LOCKFILE_ENV in unpinned  # the error tells you how to fix it

    pinned = run_gate(**{LOCKFILE_ENV: str(lockfile)})
    assert pinned == f"PINNED: {lockfile_digest(lockfile)}"


def test_result_is_frozen(anchor: ScenarioSpec) -> None:
    result = assert_reproducible(anchor)
    with pytest.raises(Exception, match="frozen"):
        result.runner = "mutated"  # type: ignore[misc]
    assert isinstance(result, Result)
