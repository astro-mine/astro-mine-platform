"""Furnishing the SPICE kernel pool from the CLI and the environment (#80).

The behaviour under test is mostly about *not* doing things: not touching the pool when nothing is
configured, not furnishing the same kernel twice, and not letting a misconfiguration reach the user
as a traceback. The one positive case — a real furnish — is asserted through a stubbed
``load_metakernel``, because NAIF kernels are not shipped and are not in CI.
"""

from __future__ import annotations

import typing
from pathlib import Path

import pytest

from astro_mine.sim import kernels
from astro_mine.sim.kernels import (
    METAKERNEL_ENV,
    KernelConfigurationError,
    furnish_metakernel,
    scenario_epoch_window,
)
from astro_mine.sim.runtime import load_scenario

DATA = Path(__file__).parent / "data"
_SCENARIO = DATA / "scenario.json"


@pytest.fixture(autouse=True)
def _clear_furnished_registry() -> typing.Iterator[None]:
    """The furnished set is process-global, like the SPICE pool it mirrors — isolate each test."""
    kernels._furnished.clear()
    yield
    kernels._furnished.clear()


@pytest.fixture
def _no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(METAKERNEL_ENV, raising=False)


def _stub_load(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, object]]:
    """Record ``load_metakernel`` calls instead of touching the real, process-global SPICE pool."""
    calls: list[tuple[Path, object]] = []

    def _fake(path: str | Path, *, coverage: object = None) -> None:
        calls.append((Path(path), coverage))

    import astro_mine.spice as spice

    monkeypatch.setattr(spice, "load_metakernel", _fake)
    return calls


# --- the no-op path: nothing configured ---------------------------------------------------------


def test_no_metakernel_configured_is_a_no_op(
    monkeypatch: pytest.MonkeyPatch, _no_env: None
) -> None:
    """The zero-prerequisite path must stay zero-prerequisite (CX-LOCAL).

    `astro-mine-sim record` on the shipped reference scenario needs no geometry at all. If this
    ever furnishes, or raises, that path acquires a NAIF download as a prerequisite.
    """
    calls = _stub_load(monkeypatch)
    assert furnish_metakernel() is None
    assert calls == []


def test_empty_env_is_treated_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    """`export ASTRO_MINE_SPICE_METAKERNEL=` is a user clearing it, not a request to load ''."""
    calls = _stub_load(monkeypatch)
    monkeypatch.setenv(METAKERNEL_ENV, "")
    assert furnish_metakernel() is None
    assert calls == []


def test_no_op_does_not_read_the_scenario(monkeypatch: pytest.MonkeyPatch, _no_env: None) -> None:
    """With nothing configured, the scenario is never inspected — not even for its epoch window.

    Callers on the resolve path hand over whatever they hold; computing a window eagerly would make
    the no-op path fail on anything that is not a full Scenario.
    """
    _stub_load(monkeypatch)

    class _Exploding:
        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"scenario attribute {name!r} read on the no-op path")

    assert furnish_metakernel(scenario=typing.cast("typing.Any", _Exploding())) is None


# --- the furnish path ---------------------------------------------------------------------------


def test_flag_beats_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls = _stub_load(monkeypatch)
    flag = tmp_path / "flag.tm"
    flag.write_text("KPL/MK")
    env = tmp_path / "env.tm"
    env.write_text("KPL/MK")
    monkeypatch.setenv(METAKERNEL_ENV, str(env))

    assert furnish_metakernel(flag) == flag
    assert [c[0] for c in calls] == [flag]


def test_env_is_used_when_no_flag(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The seam that makes `bench score --runner sim` work without Bench ever naming SPICE."""
    calls = _stub_load(monkeypatch)
    env = tmp_path / "env.tm"
    env.write_text("KPL/MK")
    monkeypatch.setenv(METAKERNEL_ENV, str(env))

    assert furnish_metakernel() == env
    assert [c[0] for c in calls] == [env]


def test_furnishing_twice_is_a_no_op(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The pool is process-global and shared; a repeat furnish would stack duplicates."""
    calls = _stub_load(monkeypatch)
    mk = tmp_path / "k.tm"
    mk.write_text("KPL/MK")

    assert furnish_metakernel(mk) == mk
    assert furnish_metakernel(mk) == mk
    assert len(calls) == 1


def test_scenario_supplies_the_coverage_window(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A short kernel set should fail at furnish time, which needs the window passed through."""
    calls = _stub_load(monkeypatch)
    mk = tmp_path / "k.tm"
    mk.write_text("KPL/MK")
    scenario = load_scenario(_SCENARIO)

    furnish_metakernel(mk, scenario=scenario)

    (_, coverage) = calls[0]
    assert coverage == scenario_epoch_window(scenario)


# --- misconfiguration is actionable, never a traceback ------------------------------------------


def test_missing_metakernel_names_the_remedy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _stub_load(monkeypatch)
    with pytest.raises(KernelConfigurationError) as excinfo:
        furnish_metakernel(tmp_path / "absent.tm")

    message = str(excinfo.value)
    assert "--metakernel" in message
    assert METAKERNEL_ENV in message
    assert "naif.jpl.nasa.gov" in message


def test_spice_failure_keeps_its_diagnosis_and_gains_the_remedy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A coverage gap is Spice's diagnosis; the remedy is ours. The user needs both."""
    import astro_mine.spice as spice

    def _fail(path: str | Path, *, coverage: object = None) -> None:
        raise spice.SpiceKernelError("SPK pool does not cover [t0, t1)")

    monkeypatch.setattr(spice, "load_metakernel", _fail)
    mk = tmp_path / "k.tm"
    mk.write_text("KPL/MK")

    with pytest.raises(KernelConfigurationError) as excinfo:
        furnish_metakernel(mk)

    message = str(excinfo.value)
    assert "SPK pool does not cover" in message
    assert METAKERNEL_ENV in message


def test_a_failed_furnish_is_not_recorded_as_done(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A kernel that failed to load must be retryable, not remembered as furnished."""
    import astro_mine.spice as spice

    monkeypatch.setattr(
        spice,
        "load_metakernel",
        lambda *_a, **_k: (_ for _ in ()).throw(spice.SpiceKernelError("boom")),
    )
    mk = tmp_path / "k.tm"
    mk.write_text("KPL/MK")

    with pytest.raises(KernelConfigurationError):
        furnish_metakernel(mk)
    assert mk not in kernels._furnished


# --- the epoch window -----------------------------------------------------------------------


def test_epoch_window_spans_the_episode() -> None:
    scenario = load_scenario(_SCENARIO)
    window = scenario_epoch_window(scenario)

    assert window.start == scenario.start_epoch
    span = window.end.tdb_seconds - window.start.tdb_seconds
    assert span == pytest.approx(scenario.dt_s * scenario.horizon_steps)
