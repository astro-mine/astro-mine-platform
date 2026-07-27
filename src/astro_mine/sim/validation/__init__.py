"""Oracle-validated regression + determinism gates (RM-P0-SIM-10; sim.md §9, §10).

The validation harness: every engine result is admissible only against an **explicit error budget**,
and a run must reproduce byte-for-byte or CI fails (sim.md §2.3, §2.4, §2.9). This package gathers
the three Phase-0 gates behind one import:

- **Orbital** (:mod:`~astro_mine.sim.validation.orbital`) — the two-body RK4 engine regressed
  against the closed-form Keplerian oracle and the conserved orbital invariants. The live STK/GMAT
  external cross-check runs in the test suite.
- **Terramechanics** (:mod:`~astro_mine.sim.validation.terramechanics`) — the mobility drawbar-pull
  speed profile and the granular excavation mass/energy balance against their analytic cases.
- **Determinism** (:mod:`~astro_mine.sim.validation.determinism`) — seeded reproducibility and
  golden-trace gates.

:func:`validate_against_oracle` is the shared comparator (worst error vs budget) returning an
:class:`OracleReport`.

Backlog: RM-P0-SIM-10 -- https://github.com/astro-mine/astro-mine-sim/issues/10
"""

from __future__ import annotations

from astro_mine.sim.validation._report import OracleReport, validate_against_oracle
from astro_mine.sim.validation.determinism import (
    DeterminismError,
    assert_matches_golden,
    assert_reproducible,
    golden_hash,
)
from astro_mine.sim.validation.orbital import (
    engine_positions_at_times,
    kepler_propagate,
    specific_angular_momentum,
    specific_energy,
    validate_orbital_conservation,
    validate_orbital_engine,
)
from astro_mine.sim.validation.terramechanics import (
    drawbar_pull_speed,
    validate_dem_granular_engine,
    validate_granular_engine,
    validate_mobility_engine,
)

__all__ = [
    "DeterminismError",
    "OracleReport",
    "assert_matches_golden",
    "assert_reproducible",
    "drawbar_pull_speed",
    "engine_positions_at_times",
    "golden_hash",
    "kepler_propagate",
    "specific_angular_momentum",
    "specific_energy",
    "validate_against_oracle",
    "validate_dem_granular_engine",
    "validate_granular_engine",
    "validate_mobility_engine",
    "validate_orbital_conservation",
    "validate_orbital_engine",
]
