#!/usr/bin/env python
"""Generate the DEM training fixture for the learned-DEM surrogate (RM-P1-SURR-02).

Drives the high-fidelity **SIM-06 DEM engine** (astro-mine-sim[dem], the ``[datagen]`` extra)
over a design of excavation configs, records the particle rollouts, and writes a compact
content-addressed ``.npz`` into the package (``astro_mine.surrogate.data``). The surrogate
trains and validates against *that frozen fixture*, so ``astro_mine.surrogate`` never imports
Sim (the narrow waist) and CI never runs the DEM engine.

The DEM engine is ``TOLERANCE`` (not bit-portable), so this is a *frozen artifact* — generate
once, commit the ``.npz``; the dataset's content hash is taken over the committed array data,
not the (timestamped) zip bytes. Regenerate:

    uv run --extra datagen python scripts/gen_dem_dataset.py

## The sampling box is a `SamplingPolicy`, not a pile of constants (surrogate#17)

The configs swept here come from :data:`SAMPLING_POLICY` — the declarative, **content-hashed**
:class:`~astro_mine.surrogate.datagen.SamplingPolicy` the package already defines for exactly this
job — and the design points from :func:`~astro_mine.surrogate.datagen.design_points`. That is not a
tidiness refactor; it is the fix for a specific class of bug.

A surrogate's trust region is *derived*: ``ExcavationTrustRegion.from_configs`` takes the tightest
box enclosing the configs the fixture actually swept. So **the sampling box silently becomes the
surrogate's contract with the rest of the platform** — the domain outside which Sim must refuse to
substitute it. When that box lives in module-level constants that no artifact records and no hash
covers, the contract is authored by accident: nothing ties it to a stated intent, nothing notices
when it stops covering the world the surrogate is deployed on, and nothing about the published tier
says which box it came from.

That is exactly how surrogate#17 happened. The friction axis shipped as ``(0.4, 0.7)`` — a
*coefficient* — while Sim queries the tier with ``tan(radians(friction_angle_deg))``, and the anchor
lunar world's regolith prior is 40 deg, i.e. a coefficient of **0.839**. Every query was
out-of-domain on the first tick, the engine escalated to DEM permanently, and the "speedup" being
measured was the reference solver against itself. Sim's own loader says so in as many words
(``engines/surrogate/_loader.py``: *"Benchmarking a surrogate outside its trust region measures
nothing"*).

Now the box is a value object with a ``content_hash()`` that rides into the published manifest, and
widening it is an edit to a declaration rather than to an incidental constant.

**The design is ``GRID`` on purpose.** A full-factorial lattice hits the box's corners exactly, so
the derived trust region (``from_configs`` min/max) comes out *equal to* the declared bounds. Under
a space-filling design (Sobol/LHS) it would land wherever the sampler happened to stop — a trust
region narrower than the policy declares, set by sampling luck. Here, declared and derived agree by
construction.

## The bed is the *published excavator's* bed

The particle bed's geometry is not a free parameter of this script: it is whatever Sim builds when
it runs the pinned Fleet excavator, which sizes the bed from the asset's ``tool`` contact element
(``runtime/content.py::dem_granular_dynamics_from_content``). Train on a 0.4 m bed of 40 particles
and deploy on a 0.8 m bed of 90 and you have an extrapolation *no trust region guards*, because bed
geometry and particle count are not axes of the box. Same for the timestep — see :data:`_DT_S`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from astro_mine.surrogate.datagen import DesignKind, SamplingPolicy, design_points
from astro_mine.surrogate.report import Bound

# --- the sampling box: the declarative, content-hashed half ------------------------

#: The excavation-parameter box the sweep samples and the surrogate declares as its trust region.
#:
#: ``friction`` is a **coefficient** (``tan(phi)``), not an angle — that unit confusion is the whole
#: of surrogate#17. The band covers the lunar regolith the anchor world actually models: its prior
#: is ``friction_angle = 40 +/- 5 deg`` (``worlds/regolith/_fields.py::DEFAULT_LUNAR_PRIOR``), so a
#: tier valid only to ``tan(35 deg) = 0.70`` is out-of-domain on the *nominal* soil, never mind the
#: uncertainty. The upper bound ``1.0`` is ``tan(45 deg)``: it covers the prior's mean, its full
#: +1-sigma excursion, and the angle of repose of the loosest plausible regolith — leaving the
#: nominal 0.839 comfortably interior rather than sitting on the boundary.
#:
#: ``restitution`` was a **degenerate point** ``[0.3, 0.3]``: the old sweep used a single value, so
#: ``from_configs`` derived a zero-width axis and the trust region was a knife-edge — any query off
#: 0.3 by a float ULP is formally outside the box. Sim never passes restitution (it is not a Worlds
#: ``RegolithParams`` field), so every query arrives at the Pydantic default of 0.3 and *happened*
#: to land on the knife-edge. Swept as a real band now, so the axis has width and 0.3 is interior:
#: the tier stays in-domain on this axis by construction rather than by coincidence.
#:
#: ``density`` (kg/m^3) brackets the anchor world's 1500. ``tool_speed`` (m/s) is the blade sweep
#: speed; Sim's benchmark runner defaults it to the midpoint of exactly this band.
SAMPLING_POLICY = SamplingPolicy(
    parameter_bounds={
        "density": Bound(low=1400.0, high=1600.0),
        "friction": Bound(low=0.4, high=1.0),
        "restitution": Bound(low=0.2, high=0.5),
        "tool_speed": Bound(low=0.05, high=0.08),
    },
    # A full-factorial lattice, so the derived trust region equals the declared box (see above).
    # `grid_design` picks `g = ceil(n_initial ** (1/P))` points per dimension (floored at 2) and
    # returns g**P rows — so `n_initial=81` over P=4 parameters means **g=3**: every axis sampled at
    # its low, midpoint and high, for 81 configs. Three levels is the least that puts an *interior*
    # sample on each axis; with two, the model would only ever have seen the box's corners and would
    # be interpolating blind across the whole of its middle — which for friction is now a 0.6-wide
    # span, not a 0.3-wide one.
    design=DesignKind.GRID,
    n_initial=81,
    # No active-learning rounds: this sweep is the initial space-filling design and the fixture is
    # frozen from it. `pool_size` is required by the schema and is inert when `n_rounds=0`.
    n_rounds=0,
    pool_size=81,
    seed=0,
)

# --- the bed: derived from the published Fleet excavator ---------------------------

#: The blade's cutting width (m) — ``astro-mine.fleet.excavator`` 0.2.0's ``tool`` contact element
#: (``mobility.contact[kind=tool].dimensions_m.x``). Sim reads it via ``asset_tool_geometry``.
_BLADE_WIDTH_M = 0.40
#: The blade's height (m) — the same element's ``dimensions_m.z``, and Sim's ``tool_height_m``.
_TOOL_HEIGHT_M = 0.15
#: The bed Sim builds for that blade: ``max(2 * width, 0.6) = 0.80`` m. Written as the formula so it
#: stays true if the blade changes (``runtime/content.py``: ``_BED_WIDTH_PER_TOOL``,
#: ``_MIN_BED_WIDTH_M``).
_BED_WIDTH_M = max(2.0 * _BLADE_WIDTH_M, 0.6)

#: Particle count and settle substeps: ``DemGranularDynamics``' own defaults, which is what Sim
#: builds with — ``dem_granular_dynamics_from_content`` overrides neither.
_N_PARTICLES = 90
_SETTLE = 1200

#: The macro timestep (s). **It must equal the tick the tier is stepped at in deployment.** The
#: served graph is a *fixed-dt map*: ``LoadedSurrogate.step(pos, vel, tool_x, config)`` takes no dt
#: and predicts exactly one dataset-timestep of evolution, whatever interval the engine believes it
#: is advancing. Train at 0.02 s and step at 0.05 s and every prediction is wrong by construction —
#: not out-of-domain (there is no dt axis on the box to catch it), just quietly wrong, until
#: re-validation against DEM breaches the error budget and escalates. 0.05 s is Sim's contact-scale
#: benchmark tick (``sim/bench/_speedup.py::_CONTACT_DT_S``) and the DEM tier's own suite's.
_DT_S = 0.05
#: Macro steps per config: 30 x 0.05 s = 1.5 s of excavation per rollout.
_STEPS = 30
_TOOL_X0_M = 0.04

_PARAM_NAMES = SAMPLING_POLICY.param_names

_OUT = (
    Path(__file__).resolve().parent.parent / "src/astro_mine/surrogate/data/dem_excavation_v1.npz"
)


def _excavate_batch():
    from astro_mine.core.messages.enums import (
        ActionKind,
        ExcavationPattern,
        ExcavationTool,
        TaskKind,
    )
    from astro_mine.core.messages.model import (
        Action,
        ActionBatch,
        ExcavateTask,
        TaskDirective,
        Vec3,
        Volume,
    )

    return ActionBatch(
        actions=[
            Action(
                agent_id="d",
                kind=ActionKind.TASK,
                task=TaskDirective(
                    task_kind=TaskKind.EXCAVATE,
                    excavate=ExcavateTask(
                        region=Volume(
                            frame="MOON_ME",
                            center_m=Vec3(x=0.0, y=0.0, z=0.0),
                            dimensions_m=Vec3(x=1.0, y=1.0, z=1.0),
                        ),
                        tool=ExcavationTool.BUCKET,
                        pattern=ExcavationPattern.TRENCH,
                        target_volume_m3=None,  # dig continuously across the window
                    ),
                ),
            )
        ]
    )


def main() -> None:
    from astro_mine.sim.engines.dem import dem_granular_engine_factory
    from astro_mine.sim.runtime import AgentSpec, DemGranularDynamics, RngStreams, Scenario

    configs = design_points(SAMPLING_POLICY)
    c_n = int(configs.shape[0])
    policy_hash = SAMPLING_POLICY.content_hash()
    box = ", ".join(
        f"{n}=[{b.low:g}, {b.high:g}]" for n, b in SAMPLING_POLICY.parameter_bounds.items()
    )
    print(
        f"sampling policy {policy_hash}\n"
        f"  box     : {box}\n"
        f"  design  : {SAMPLING_POLICY.design.value} -> {c_n} configs\n"
        f"  bed     : {_N_PARTICLES} particles, {_BED_WIDTH_M:g} m wide, "
        f"blade {_TOOL_HEIGHT_M:g} m tall, {_SETTLE} settle substeps\n"
        f"  rollout : {_STEPS} steps x {_DT_S:g} s = {_STEPS * _DT_S:g} s\n",
        flush=True,
    )

    states = np.zeros((c_n, _STEPS + 1, _N_PARTICLES, 4), dtype=np.float64)  # x, z, vx, vz
    tool_x = np.zeros((c_n, _STEPS + 1), dtype=np.float64)
    params = np.zeros((c_n, len(_PARAM_NAMES)), dtype=np.float64)
    task = _excavate_batch()

    started = time.perf_counter()
    for c, (density, friction, restitution, tool_speed) in enumerate(configs):
        dyn = DemGranularDynamics(
            n_particles=_N_PARTICLES,
            settle_substeps=_SETTLE,
            bed_width_m=_BED_WIDTH_M,
            regolith_density_kg_m3=density,
            friction_coeff=friction,
            restitution=restitution,
            tool_speed_mps=tool_speed,
            tool_x0_m=_TOOL_X0_M,
            tool_height_m=_TOOL_HEIGHT_M,
        )
        scenario = Scenario(
            name=f"dem-gen-{c}",
            horizon_steps=1,
            dt_s=_DT_S,
            agents=(AgentSpec(agent_id="d", battery_soc_j=1.0e9, dynamics=dyn),),
        )
        engine = dem_granular_engine_factory(scenario, RngStreams(c))
        engine.apply_actions(task)
        pos, vel = engine.particles("d")
        states[c, 0, :, :2], states[c, 0, :, 2:] = pos, vel
        tool_x[c, 0] = engine.bed("d").tool_x_m
        for t in range(1, _STEPS + 1):
            engine.advance(_DT_S)
            pos, vel = engine.particles("d")
            states[c, t, :, :2], states[c, t, :, 2:] = pos, vel
            tool_x[c, t] = engine.bed("d").tool_x_m
        params[c] = (density, friction, restitution, tool_speed)
        elapsed = time.perf_counter() - started
        print(
            f"  [{c + 1:3d}/{c_n}] density={density:7.1f} friction={friction:.3f} "
            f"restitution={restitution:.3f} tool_speed={tool_speed:.3f} "
            f"({elapsed:6.1f}s elapsed, {elapsed / (c + 1):.2f}s/config)",
            flush=True,
        )

    wall_clock_s = time.perf_counter() - started

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        _OUT,
        states=states,
        tool_x=tool_x,
        params=params,
        dt_s=np.array([_DT_S]),
        bed_width_m=np.array([_BED_WIDTH_M]),
        tool_height_m=np.array([_TOOL_HEIGHT_M]),
        feature_names=np.array(["pos_x", "pos_z", "vel_x", "vel_z"]),
        param_names=np.array(list(_PARAM_NAMES)),
        # The declarative box this fixture was swept under, by hash *and* by value — so a
        # surrogate's derived trust region can always be traced back to the policy that produced it,
        # and a reader of the artifact never has to guess which box it came from.
        sampling_policy_hash=np.array([policy_hash]),
        sampling_policy=np.array([json.dumps(SAMPLING_POLICY.model_dump(mode="json"))]),
    )
    print(
        f"\nwrote {_OUT} ({_OUT.stat().st_size} bytes): "
        f"{c_n} configs x {_STEPS} steps x {_N_PARTICLES} particles "
        f"in {wall_clock_s:.1f}s of DEM"
    )


if __name__ == "__main__":
    main()
