"""The Sim-backed Bench runners — real-physics scoring + the determinism gate (RM-P0-SIM-11).

Bench's scoring path and its determinism gate both ship a **documented stand-in**: the baseline's
``reference_episode_runner`` is "a deterministic trace fixture, *not a physics engine*", and the
harness's ``reference_runner`` is "a pure, seeded function of the scenario hash ... enough to
exercise
the reproducibility oracle *without a physics engine*". This package is the real runner behind both.

**The dependency direction is one-way, and that is the design** (conventions.md §1.1; bench.md
§2.2).
Bench never imports Sim — it stays dependency-clean (core + pydantic) and composes rather than
simulating. So the runner that satisfies Bench's Core-typed seams lives *here*, in the Sim repo, and
Bench receives it by **injection**::

    from astro_mine.bench.baseline import run
    from astro_mine.bench.zoo import load_scenario
    from astro_mine.sim.bench import SimEpisodeRunner
    from astro_mine.sim.runtime import open_bundle_store

    store = open_bundle_store("files/hub-registry")
    card = run(load_scenario("lunar-polar-ice-prospecting-v1"), policy,
               runner=SimEpisodeRunner(store=store))

and the determinism gate the same way::

    from astro_mine.bench.harness import assert_reproducible
    from astro_mine.bench.metrics import scored_metric_values
    from astro_mine.sim.bench import SIM_RUNNER_ID, SimEpisodeRunner, SimHarnessRunner

    gate = SimHarnessRunner(SimEpisodeRunner(store=store), scorer=scored_metric_values)
    result = assert_reproducible(spec, gate, runner_id=SIM_RUNNER_ID)

The ``scorer`` is Bench's, passed in: resolving a scenario's metric references is the benchmark's
job, and handing it to the runner is what keeps Sim from importing Bench (conventions.md §3.3).

Optional (``astro-mine-platform[sim-bench]``) and inert: nothing in Sim's runtime imports this
package, so the base wheel stays Bench-free.

**Which paths are still the fixture.** Injecting the runner is the *caller's* choice, and Bench's
defaults are unchanged — so anything that does not pass ``runner=`` still runs the reference
fixture.
That is deliberate (the fixture is what makes Bench's offline, no-account tier work with no Sim
installed), and it means in particular that Bench's ``astro-mine-bench`` CLI and its
``scripts/determinism_gate.py`` still use the fixture: neither exposes a ``--runner`` flag today.
A Sim-backed run is a Python-API call, as above.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astro_mine.sim.bench._runner import (
        SIM_RUNNER_ID,
        SimEpisodeRunner,
        SimHarnessRunner,
        sim_runner_provider,
    )
    from astro_mine.sim.bench._scenario import (
        ResolvedRun,
        dynamics_for_asset,
        materialize_bench_run,
        scenario_content_from_spec,
        sim_scenario_from_spec,
    )
    from astro_mine.sim.bench._scoring import (
        episode_trace_from,
        night_intervals,
        observations_from,
        scoring_context_for,
    )
    from astro_mine.sim.bench._speedup import FidelitySpeedupRunner, SpeedupReport

__all__ = [
    "SIM_RUNNER_ID",
    "FidelitySpeedupRunner",
    "ResolvedRun",
    "SimEpisodeRunner",
    "SimHarnessRunner",
    "SpeedupReport",
    "dynamics_for_asset",
    "episode_trace_from",
    "materialize_bench_run",
    "night_intervals",
    "observations_from",
    "scenario_content_from_spec",
    "scoring_context_for",
    "sim_runner_provider",
    "sim_scenario_from_spec",
]

_BENCH_HINT = (
    "the Sim-backed Bench runner requires astro-mine-bench; "
    "install it with: pip install 'astro-mine-platform[sim-bench]'"
)

#: Which private module each public name lives in — the lazy-import map that keeps Bench off the
#: base
#: wheel's import path.
_EXPORTS = {
    "SIM_RUNNER_ID": "._runner",
    "FidelitySpeedupRunner": "._speedup",
    "ResolvedRun": "._scenario",
    "SimEpisodeRunner": "._runner",
    "SimHarnessRunner": "._runner",
    "sim_runner_provider": "._runner",
    "SpeedupReport": "._speedup",
    "dynamics_for_asset": "._scenario",
    "materialize_bench_run": "._scenario",
    "episode_trace_from": "._scoring",
    "night_intervals": "._scoring",
    "observations_from": "._scoring",
    "scenario_content_from_spec": "._scenario",
    "scoring_context_for": "._scoring",
    "sim_scenario_from_spec": "._scenario",
}


def __getattr__(name: str) -> object:
    """Resolve a public name by importing its module on first access (PEP 562)."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    try:
        module = import_module(f"{__name__}{module_name}")
    except ModuleNotFoundError as exc:  # astro_mine.bench absent
        raise ModuleNotFoundError(_BENCH_HINT) from exc
    return getattr(module, name)
