"""RM-P0-SIM-10 — the live external-oracle orbital regression against NASA GMAT.

The flight-grade half of the orbital validation (sim.md §10): the OrbitalEngine's RK4 two-body
propagation is regressed against a GMAT run of the *same* point-mass-Luna mission within an explicit
error budget. GMAT runs **live** — this test loads the committed ``two_body_moon.script``, overrides
the initial Cartesian state, runs GMAT via ``gmat-run``, and compares the engine to GMAT's reported
ephemeris.

It is a required gate in CI (where ``setup-gmat`` provides GMAT and the ``gmat`` dependency group
provides ``gmat-run``), and **skips** wherever GMAT or ``gmat-run`` is unavailable — e.g. a Python
3.13 dev checkout, or a contributor without a GMAT install. The dependency-free analytic Kepler
regression in ``test_validation_orbital.py`` always runs.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from astro_mine.sim.validation import engine_positions_at_times, validate_against_oracle

pytest.importorskip("gmat_run", reason="gmat-run not installed (Python <3.13 + a GMAT install)")

_SCRIPT = Path(__file__).parent / "data" / "two_body_moon.script"
_R_M = 1_837_400.0  # ~100 km circular lunar orbit radius (m)
# GMAT-vs-engine agree to ~4e-10 over an orbit at matched mu; 1e-6 leaves four orders of margin —
# tight enough that a wrong mu (~1e-4), frame, or unit (km↔m) regression trips it.
_BUDGET = 1e-6


@pytest.fixture(scope="module")
def gmat_run_report(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[float, list[float], list[list[float]]]:
    """Run GMAT once for the module: returns (mu_m3_s2, elapsed_s, positions_m).

    Skips (rather than fails) when no GMAT install is discoverable, so the test is required only
    where GMAT is actually provisioned (CI via ``setup-gmat``)."""
    from gmat_run import Mission
    from gmat_run.errors import GmatNotFoundError

    try:
        mission = Mission.load(_SCRIPT)
    except GmatNotFoundError as exc:  # gmat-run installed, but no GMAT install on this machine
        pytest.skip(f"GMAT install not found: {exc}")

    mu_m3_s2 = float(mission["Luna.Mu"]) * 1e9  # GMAT reports km^3/s^2
    v_circ_mps = math.sqrt(mu_m3_s2 / _R_M)
    # Pin the engine and GMAT to the identical initial state (km, km/s).
    mission["Sat.X"] = _R_M / 1000.0
    mission["Sat.Y"] = 0.0
    mission["Sat.Z"] = 0.0
    mission["Sat.VX"] = 0.0
    mission["Sat.VY"] = v_circ_mps / 1000.0
    mission["Sat.VZ"] = 0.0

    # An explicit per-run workspace: relative ReportFile output is redirected here (gmat-run does
    # not reliably surface it from its default temp dir).
    work_dir = tmp_path_factory.mktemp("gmat")
    frame = mission.run(working_dir=work_dir).reports["R"]
    elapsed_s = [float(t) for t in frame["Sat.ElapsedSecs"]]
    positions_m = [
        [float(frame[f"Sat.MoonInertial.{axis}"][i]) * 1000.0 for axis in ("X", "Y", "Z")]
        for i in range(len(frame))
    ]
    return mu_m3_s2, elapsed_s, positions_m


def test_orbital_engine_matches_gmat_over_one_orbit(
    gmat_run_report: tuple[float, list[float], list[list[float]]],
) -> None:
    mu_m3_s2, elapsed_s, gmat_positions = gmat_run_report
    assert len(elapsed_s) > 100  # GMAT reported the whole orbit at ~60 s cadence

    v_circ_mps = math.sqrt(mu_m3_s2 / _R_M)
    r0 = (_R_M, 0.0, 0.0)
    v0 = (0.0, v_circ_mps, 0.0)
    engine_positions = engine_positions_at_times(r0, v0, mu_m3_s2, elapsed_s)

    report = validate_against_oracle(
        engine_positions, gmat_positions, budget=_BUDGET, name="orbital-vs-gmat", relative=True
    )
    assert report.passed, f"engine diverged from GMAT: max rel error {report.max_error:.2e}"
