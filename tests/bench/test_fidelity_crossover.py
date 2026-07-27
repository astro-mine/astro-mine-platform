"""The fidelity scenario's published scaling curve is the shape it claims to be (bench#52).

`measure_fidelity_crossover.py` produces `crossover.json` beside the scenario — the DEM-vs-surrogate
speedup swept over bed size. The measurement itself needs Sim, a Hub registry and SPICE kernels, so
it runs out of tree and is committed as an artifact. What CI *can* guard, with no Sim, is that the
committed artifact is the kind of result it says it is:

- a **cost** result, never dressed up as a substitution claim (no `is_claim`);
- **monotonic in N** — the entire point. A DEM solver is O(N^2) and the served surrogate is O(N.k),
  so the speedup must *grow* with the bed. When it did not (a flat ~2x, before the served graph was
  made sparse in astro-mine-surrogate#24), that was the bug this curve exists to have caught.
"""

from __future__ import annotations

import json
from itertools import pairwise
from pathlib import Path

import pytest

_CURVE = (
    Path(__file__).resolve().parent.parent.parent
    / "src/astro_mine/bench/zoo/lunar_polar_ice_excavation_fidelity_v1/crossover.json"
)


@pytest.fixture(scope="module")
def curve() -> dict:
    if not _CURVE.exists():
        pytest.skip("crossover.json is not committed yet (produced out of tree by the maintainer)")
    return json.loads(_CURVE.read_text())


def test_the_curve_is_a_cost_result_not_a_substitution_claim(curve: dict) -> None:
    # A speedup at a bed size the tier was never validated at is a cost result. It must not carry
    # `is_claim`, and it must say what it is.
    assert curve["measures"] == "cost_only"
    assert "is_claim" not in curve
    for point in curve["points"]:
        assert "is_claim" not in point


def test_the_curve_is_anchored_to_the_scenarios_own_pinned_content(curve: dict) -> None:
    from astro_mine.bench.zoo import load_scenario

    spec = load_scenario("lunar-polar-ice-excavation-fidelity-v1")
    assert curve["scenario_id"] == spec.scenario_id
    # The sweep is off the scenario's real spec + surrogate, not a hand-authored bed.
    assert curve["spec_hash"] == spec.spec_hash
    # The lowest bed is the scenario's own, so the curve has an anchor point on the real task.
    ns = [p["n_particles"] for p in curve["points"]]
    assert ns == sorted(ns) and len(set(ns)) == len(ns)


def test_the_speedup_grows_with_the_bed(curve: dict) -> None:
    # The regression that matters. If the served graph ever reverts to O(N^2), this goes flat.
    speedups = [p["speedup"] for p in curve["points"]]
    assert all(b > a for a, b in pairwise(speedups)), (
        f"the speedup does not increase monotonically with N: {speedups} — the served tier is "
        "not out-scaling the DEM solver, which is the one thing a surrogate has to do"
    )
    assert speedups[0] > 1.0, "the surrogate is not even faster than DEM at the smallest bed"


def test_the_neighbour_count_is_a_packing_density_not_a_function_of_N(curve: dict) -> None:
    # Why the tier *can* be O(N.k): max neighbours is set by sphere packing, ~constant in N.
    max_ks = [p["max_neighbours"] for p in curve["points"]]
    assert max(max_ks) - min(max_ks) <= 3, (
        f"max neighbour count varies too much with N ({max_ks}) to be a packing density"
    )
