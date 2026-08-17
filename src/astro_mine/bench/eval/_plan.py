# SPDX-License-Identifier: Apache-2.0
"""The evaluation-batch planner — fan seeds x submissions out onto Cloud (RM-P1-BENCH-11).

:func:`plan_batch` turns a ``(ScenarioSpec, submissions, seed set, budgets)`` request into the
[Cloud](cloud.md) specs that run it: one **``SweepSpec``** per submission — a base ``JobSpec`` whose
digest-pinned image is the [Sim](sim.md) rollout container, whose command is Bench's single-seed
worker (:mod:`astro_mine.bench.eval._worker`), fanned out over the seed set. CPU seeds compile to an
**Argo** fan-out (``compile_sweep``; ``max_parallel`` is the back-pressure cap); **GPU rollouts**
set ``distributed=True`` + a MIG ``ResourceRequest`` so Cloud routes each seed to **Ray/KubeRay**
(``select_engine``) — bench.md §3, §7, §8.

Bench owns *what* to evaluate and *its budget*; Cloud owns *how* to run it. Per the narrow waist
(bench.md §2.2) Bench never imports Sim: the rollout runner arrives as the ``JobSpec.image``, a
Cloud-scheduled container, not a package dependency. ``astro_mine.cloud`` is imported **lazily**
here (behind the ``[cloud]`` extra) so ``import astro_mine.bench.eval`` stays dependency-clean
(core + pydantic).

Backlog: RM-P1-BENCH-11 — astro-mine-bench#19
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from astro_mine.bench.leaderboard._models import SubmissionRequest
from astro_mine.bench.scenario import ScenarioSpec

if TYPE_CHECKING:
    from astro_mine.cloud.sched import CostRates
    from astro_mine.cloud.submission import SweepSpec

__all__ = [
    "DEFAULT_HOURS_PER_SEED",
    "DEFAULT_RUNNER",
    "METRICS_OUTPUT",
    "REFERENCE_ROLLOUT_IMAGE",
    "SEED_ENV",
    "TRACE_OUTPUT",
    "EvaluationPlan",
    "EvaluationTarget",
    "PlannedEvaluation",
    "plan_batch",
]

#: A digest-pinned placeholder Sim rollout image for the local/CI tier. The digest is irrelevant to
#: Cloud's local backend (which runs the worker in-process), so this stands in for the real,
#: Hub-published Sim image a deployment schedules — Bench names the container, never imports Sim.
REFERENCE_ROLLOUT_IMAGE = "ghcr.io/astro-mine/astro-mine-sim@sha256:" + "0" * 64

#: Env var carrying a variant's scenario seed. The sweep injects it per fanned-out seed and the
#: worker reads it (a dedicated key, so an arbitrary held-out seed set fans out verbatim rather than
#: as the sweep's derived ``base.seed + index``).
SEED_ENV = "ASTRO_MINE_EVAL_SEED"
#: The run-relative output name for a seed's columnar per-metric result (Parquet; bench.md §5).
METRICS_OUTPUT = "metrics.parquet"
#: The run-relative output name for a seed's raw episode trace (MCAP; bench.md §5, §6).
TRACE_OUTPUT = "trace.mcap"
#: The default runner a fan-out names — the dependency-clean trace fixture, not a physics engine.
#: A deployment scheduling the real Sim rollout image plans with ``runner="sim"``; Bench resolves
#: the name through the ``astro_mine.bench.runners`` entry-point group and never imports Sim.
DEFAULT_RUNNER = "fixture"
#: Wall-clock a single-seed rollout is costed at, for budget estimation (a knob, not an SLO).
DEFAULT_HOURS_PER_SEED = 0.05


class EvaluationTarget(BaseModel):
    """One submission to evaluate on Cloud: the display request + rollout image + routing.

    ``request`` is the submit-policy-we-run intake (reused from the leaderboard, RM-P0-BENCH-06);
    ``image`` is the **digest-pinned** Sim rollout image (``repository@sha256:…``) Cloud schedules —
    Bench never imports Sim, it names the container. ``gpu``/``mig_profile`` route a submission's
    seeds to **KubeRay** (a MIG slice shares a card; bench.md §7); otherwise CPU seeds fan out.
    ``compute_budget`` is the submission's hard compute cap (``None`` ⇒ unbounded).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    request: SubmissionRequest
    image: str = Field(min_length=1)
    tenant: str = Field(default="public", min_length=1)
    gpu: bool = False
    #: A MIG profile (e.g. ``1g.10gb``) validated against ``gpu_model`` at plan time; implies gpu.
    mig_profile: str | None = None
    gpu_model: str = "a100-80gb"
    cpu: str | None = None
    memory: str | None = None
    compute_budget: float | None = Field(default=None, gt=0.0)


@dataclass(frozen=True)
class PlannedEvaluation:
    """One submission's compiled evaluation: the Cloud sweep + Bench's budget/quota bookkeeping.

    ``sweep`` fans the seeds out (Argo for CPU, per-seed KubeRay for GPU); ``cost_per_seed`` is the
    estimated spend charged to the ``cap_key`` budget per seed; ``admission_request`` is the
    per-tenant resource reservation the fair-share queue admits against (bench.md §8).
    """

    request: SubmissionRequest
    tenant: str
    seeds: tuple[int, ...]
    sweep: SweepSpec
    distributed: bool
    cost_per_seed: float
    cap_key: str | None
    admission_request: dict[str, float]


@dataclass(frozen=True)
class EvaluationPlan:
    """A planned evaluation batch: the per-submission sweeps + the per-submission budget caps."""

    evaluations: tuple[PlannedEvaluation, ...]
    budget_caps: dict[str, float] = field(default_factory=dict)


def _cpu_cores(request: str | None) -> float:
    """Parse a Kubernetes CPU quantity (``"2"``, ``"500m"``) to fractional cores; default 1.0."""
    if request is None:
        return 1.0
    text = request.strip()
    if text.endswith("m"):
        return int(text[:-1]) / 1000.0
    return float(text)


def _budget_key(scenario_id: str, request: SubmissionRequest) -> str:
    """A stable per-submission budget key (scenario x policy), keyed before the id is scored."""
    return f"{scenario_id}::{request.policy_ref}"


def plan_batch(
    spec: ScenarioSpec,
    targets: Sequence[EvaluationTarget],
    *,
    seeds: Sequence[int],
    max_parallel: int | None = None,
    hours_per_seed: float = DEFAULT_HOURS_PER_SEED,
    cost_rates: CostRates | None = None,
    spot: bool = True,
    base_env: Mapping[str, str] | None = None,
    runner: str = DEFAULT_RUNNER,
) -> EvaluationPlan:
    """Plan a fan-out of ``targets`` over ``seeds`` on ``spec`` into Cloud sweeps + budget caps.

    Each target becomes one :class:`~astro_mine.cloud.submission.SweepSpec`: a base
    :class:`~astro_mine.cloud.submission.JobSpec` running the single-seed worker in the target's
    digest-pinned rollout image, fanned out over ``seeds`` via a ``{SEED_ENV: [...]}`` grid.
    ``max_parallel`` caps the fan-out (back-pressure; bench.md §8). GPU targets set
    ``distributed=True`` + a MIG ``ResourceRequest`` so Cloud routes each seed to KubeRay; CPU
    targets fan out across Ray/Argo workers. Requires the ``[cloud]`` extra.

    Raises ``ValueError`` for an empty seed set or a target whose ``request.scenario_id`` disagrees
    with ``spec``; propagates Cloud's ``ValueError`` for a bad MIG profile or unpinned image.
    """
    from astro_mine.cloud.gpu.mig import validate_profile
    from astro_mine.cloud.packaging import ImageRef
    from astro_mine.cloud.sched import estimate_cost
    from astro_mine.cloud.submission import JobSpec, ResourceRequest, SweepSpec

    seed_list = tuple(seeds)
    if not seed_list:
        raise ValueError("plan_batch needs at least one seed to fan out")

    core_version = spec.core_interface.get("env")
    evaluations: list[PlannedEvaluation] = []
    budget_caps: dict[str, float] = {}

    for target in targets:
        if target.request.scenario_id != spec.scenario_id:
            raise ValueError(
                f"target scenario {target.request.scenario_id!r} != spec {spec.scenario_id!r}"
            )
        image = ImageRef.parse(target.image)

        resource: ResourceRequest | None
        if target.gpu or target.mig_profile is not None:
            if target.mig_profile is not None:
                profile = validate_profile(target.gpu_model, target.mig_profile)
                resource = ResourceRequest(mig_profile=target.mig_profile)
                gpus = 1.0 / profile.instances
            else:
                resource = ResourceRequest(gpu=1)
                gpus = 1.0
            distributed = True
            admission = {"nvidia.com/gpu": gpus}
            cost = estimate_cost(hours=hours_per_seed, gpus=gpus, spot=spot, rates=cost_rates)
        else:
            cores = _cpu_cores(target.cpu)
            resource = (
                ResourceRequest(cpu=target.cpu, memory=target.memory)
                if target.cpu is not None or target.memory is not None
                else None
            )
            distributed = False
            admission = {"cpu": cores}
            cost = estimate_cost(hours=hours_per_seed, cpus=cores, spot=spot, rates=cost_rates)

        command = [
            "python",
            "-m",
            "astro_mine.bench",
            "eval-worker",
            "--scenario-id",
            spec.scenario_id,
            "--policy-ref",
            target.request.policy_ref,
            # Named in the argv, not left to the image's default: the plan is the record of what the
            # fan-out was asked to run, and the worker stamps the resolved id back (bench#64).
            "--runner",
            runner,
        ]
        base = JobSpec(
            image=image,
            command=command,
            env=dict(base_env or {}),
            outputs=[METRICS_OUTPUT, TRACE_OUTPUT],
            core_interface_version=core_version,
            resource_request=resource,
            distributed=distributed,
            tenant=target.tenant,
            budget=target.compute_budget,
        )
        sweep = SweepSpec(
            base=base,
            method="grid",
            grid={SEED_ENV: list(seed_list)},
            max_parallel=max_parallel,
        )

        cap_key: str | None = None
        if target.compute_budget is not None:
            cap_key = _budget_key(spec.scenario_id, target.request)
            budget_caps[cap_key] = target.compute_budget

        evaluations.append(
            PlannedEvaluation(
                request=target.request,
                tenant=target.tenant,
                seeds=seed_list,
                sweep=sweep,
                distributed=distributed,
                cost_per_seed=cost,
                cap_key=cap_key,
                admission_request=admission,
            )
        )

    return EvaluationPlan(evaluations=tuple(evaluations), budget_caps=budget_caps)
