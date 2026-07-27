"""Comms-stress curves — the headline honest-eval diagnostic (RM-P1-LEARN-06).

Robust coordination under intermittent, degraded comms is the charter §8 problem; the
comms-stress curve — performance swept across ``CommsModel`` drop / delay / budget settings —
is how Learn diagnoses it (learn.md §8). Because the comms regime is baked into a
:class:`~astro_mine.learn.envs.SwarmEnv` at construction (there is no runtime setter — a
degraded channel must be identical across every algorithm), a sweep rebuilds a **fresh env per
grid point** through a :data:`CommsEnvFactory`.

- :class:`CommsStressGrid` — a declarative, JSON-Schema-emitting grid of swept points (like
  :class:`~astro_mine.learn.envs.CommsModelConfig`), enumerated to one
  :class:`~astro_mine.learn.envs.CommsModelConfig` per point via :meth:`~CommsStressGrid.points`.
- :func:`comms_stress_curve` / :func:`comms_stress_curves` — run the **identical** grid + split +
  held-out seeds for one policy, or across a mapping of named policies (ippo/mappo/qmix on the
  same axes), and emit the long-format :class:`~astro_mine.learn.eval.aggregate.CurveTable` — the
  cross-algorithm comparison the issue AC requires ("comparable across algorithms").

Each point's per-seed row records the episode return and the comms-stress denominator
(``delivery_ratio`` = aggregated delivered / offered from the env's
:meth:`~astro_mine.learn.envs.SwarmEnv.comms_report` ledger).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.core.hashing import content_hash_json
from astro_mine.learn._core import CORE_INTERFACES
from astro_mine.learn.envs import (
    BandwidthBudgetConfig,
    CommsModelConfig,
    DelayConfig,
    DropConfig,
    SwarmEnv,
)
from astro_mine.learn.eval.aggregate import CURVE_SCHEMA_VERSION, CurveRow, CurveTable, MetricSink
from astro_mine.learn.eval.split import HeldOutSplit
from astro_mine.learn.eval.sweep import DEFAULT_MIN_SEEDS, PolicyUnderTest, seed_sweep
from astro_mine.learn.train.executor import RewardFn

__all__ = [
    "CommsEnvFactory",
    "CommsStressGrid",
    "StressPoint",
    "build_curve_manifest",
    "comms_stress_curve",
    "comms_stress_curves",
]

#: A factory that builds a fresh SwarmEnv for a given comms regime — the seam a sweep uses to
#: rebuild the env per grid point (``make_swarm_env(world, assets, comms_model=CommsModel(cfg))``).
CommsEnvFactory = Callable[[CommsModelConfig], SwarmEnv]

_Prob = Annotated[float, Field(ge=0.0, le=1.0)]
_NonNegInt = Annotated[int, Field(ge=0)]
_PosFloat = Annotated[float, Field(gt=0.0)]


@dataclass(frozen=True)
class StressPoint:
    """One swept grid point: which axis varies, its value, and the concrete comms regime."""

    axis: Literal["drop", "delay", "budget"]
    value: float
    config: CommsModelConfig


class CommsStressGrid(BaseModel):
    """A declarative comms-stress sweep — one axis knob varied per point off the identity
    channel, so each axis yields a clean curve.

    Emit its JSON Schema with :meth:`model_json_schema` and round-trip it through JSON (like
    :class:`~astro_mine.learn.envs.CommsModelConfig`). Its content hash is part of the run
    manifest (the reproducibility key)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    #: Per-link Bernoulli drop probabilities to sweep (the primary comms-stress axis).
    drop_probabilities: tuple[_Prob, ...] = (0.0, 0.3, 0.6, 0.9)
    #: Fixed delivery delays in whole ticks to sweep.
    delay_ticks: tuple[_NonNegInt, ...] = ()
    #: Per-agent bandwidth budgets (bits/tick) to sweep.
    budgets: tuple[_PosFloat, ...] = ()
    #: Bumped when the meaning of the grid changes (the Bench/View curve contract).
    schema_version: Literal["0.1.0"] = "0.1.0"

    def points(self) -> list[StressPoint]:
        """Enumerate the grid into one :class:`StressPoint` per swept value.

        Each point varies a single knob off the identity channel; a ``drop`` of ``0.0`` is the
        identity regime (nothing degraded — the curve's high-water mark)."""
        points: list[StressPoint] = []
        for prob in self.drop_probabilities:
            config = CommsModelConfig(drop=DropConfig(probability=prob))
            points.append(StressPoint("drop", float(prob), config))
        for ticks in self.delay_ticks:
            points.append(
                StressPoint(
                    "delay",
                    float(ticks),
                    CommsModelConfig(delay=DelayConfig(kind="fixed", ticks=ticks)),
                )
            )
        for budget in self.budgets:
            points.append(
                StressPoint(
                    "budget",
                    float(budget),
                    CommsModelConfig(
                        bandwidth=BandwidthBudgetConfig(per_agent_bits_per_tick=budget)
                    ),
                )
            )
        return points


def build_curve_manifest(
    *,
    policy_ids: Mapping[str, str],
    grid: CommsStressGrid,
    split: HeldOutSplit,
    seeds: Sequence[int],
    steps: int,
) -> dict[str, Any]:
    """The canonical run manifest whose content hash is the curve's reproducibility key.

    Captures everything that determines the curve — the named policy ids, the grid, the
    held-out split (train + held-out seeds), the swept seeds, the horizon, and the Core
    interface versions Learn is built against — so an identical experiment hashes identically
    (conventions.md §11)."""
    return {
        "schema_version": CURVE_SCHEMA_VERSION,
        "kind": "comms_stress_curve",
        "policies": dict(sorted(policy_ids.items())),
        "grid": grid.model_dump(mode="json"),
        "split": {
            "train_seeds": sorted(split.train_seeds),
            "held_out_seeds": list(split.held_out_seeds),
        },
        "seeds": [int(s) for s in seeds],
        "steps": int(steps),
        "core_interfaces": dict(sorted(CORE_INTERFACES.items())),
    }


def comms_stress_curves(
    policies: Mapping[str, PolicyUnderTest],
    world_factory: CommsEnvFactory,
    grid: CommsStressGrid,
    split: HeldOutSplit,
    *,
    steps: int = 64,
    reward_fn: RewardFn | None = None,
    min_seeds: int = DEFAULT_MIN_SEEDS,
    policy_ids: Mapping[str, str] | None = None,
    sample_efficiencies: Mapping[str, float] | None = None,
    sink: MetricSink | None = None,
) -> CurveTable:
    """Score every named policy across the **identical** grid + held-out seeds → one table.

    Each ``(policy, point)`` runs a held-out :func:`~astro_mine.learn.eval.seed_sweep` (which
    rejects single-seed sweeps) over a fresh env built by ``world_factory`` for that point's
    comms regime, and records one long-format row per ``(algorithm, stress point, seed)`` with
    the episode return and the ``delivery_ratio`` denominator. The result is comparable across
    algorithms by construction (same axes, same seeds). If ``sink`` is given the table is
    emitted through it (default aggregation is Parquet)."""
    resolved_ids = {name: (policy_ids or {}).get(name, f"live:{name}") for name in policies}
    effs = dict(sample_efficiencies or {})
    points = grid.points()
    seeds = split.held_out_seeds

    rows: list[CurveRow] = []
    for name, policy in policies.items():
        policy_id = resolved_ids[name]
        efficiency = effs.get(name)
        for point in points:
            sweep = seed_sweep(
                policy,
                partial(world_factory, point.config),
                seeds,
                steps=steps,
                reward_fn=reward_fn,
                min_seeds=min_seeds,
            )
            comms_config_hash = content_hash_json(point.config.model_dump(mode="json"))
            report = sweep.report
            for index, seed in enumerate(report.seeds):
                ledger = report.comms_stress[index] if index < len(report.comms_stress) else {}
                offered = int(sum(agent.get("offered", 0.0) for agent in ledger.values()))
                delivered = int(sum(agent.get("delivered", 0.0) for agent in ledger.values()))
                rows.append(
                    CurveRow(
                        algorithm=name,
                        policy_id=policy_id,
                        stress_axis=point.axis,
                        stress_value=point.value,
                        seed=int(seed),
                        episode_return=report.returns[index],
                        delivery_ratio=1.0 if offered == 0 else delivered / offered,
                        offered=offered,
                        delivered=delivered,
                        eval_throughput_steps_per_s=sweep.eval_throughput_steps_per_s,
                        wall_clock_s=sweep.wall_clock_s,
                        sample_efficiency=efficiency,
                        comms_config_hash=comms_config_hash,
                    )
                )

    manifest = build_curve_manifest(
        policy_ids=resolved_ids, grid=grid, split=split, seeds=seeds, steps=steps
    )
    table = CurveTable(rows, manifest)
    if sink is not None:
        sink.write(table)
    return table


def comms_stress_curve(
    policy: PolicyUnderTest,
    world_factory: CommsEnvFactory,
    grid: CommsStressGrid,
    split: HeldOutSplit,
    *,
    algorithm: str = "policy",
    policy_id: str | None = None,
    steps: int = 64,
    reward_fn: RewardFn | None = None,
    min_seeds: int = DEFAULT_MIN_SEEDS,
    sample_efficiency: float | None = None,
    sink: MetricSink | None = None,
) -> CurveTable:
    """The single-policy comms-stress curve — a thin wrapper over :func:`comms_stress_curves`."""
    return comms_stress_curves(
        {algorithm: policy},
        world_factory,
        grid,
        split,
        steps=steps,
        reward_fn=reward_fn,
        min_seeds=min_seeds,
        policy_ids=None if policy_id is None else {algorithm: policy_id},
        sample_efficiencies=None if sample_efficiency is None else {algorithm: sample_efficiency},
        sink=sink,
    )
