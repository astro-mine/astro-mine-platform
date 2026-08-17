# SPDX-License-Identifier: Apache-2.0
"""Astro-Mine-Learn — the multi-agent reinforcement-learning toolkit.

Turns a simulatable world into trainable RL problems: Gymnasium (single-agent) and
PettingZoo (multi-agent) env wrappers over the Core Environment API with first-class
partial observability and intermittent, delayed comms; a comms-limited CTDE baseline
suite (MAPPO / IPPO / QMIX) plus the comms-learning research track (``comms_ppo``);
staged curricula and domain randomization; MLflow-backed experiment tracking with
provenance capture; Ray-based scale-out and GPU-vectorized (JAX) rollout; and portable
ONNX policy export (feed-forward, recurrent, and comms-learning policies alike).

Learn is a *consumer* of Core contracts — it defines no new asset, world, or message
schema and never widens the waist. The heavy runtime dependencies (Ray RLlib + Torch,
ONNX/ONNX-Runtime, pyarrow, JAX, MLflow) all live behind optional extras, so the base
wheel stays a lightweight env-adapter library.
See ``docs/architecture/learn.md``.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("astro-mine-platform")
except PackageNotFoundError:  # pragma: no cover - source tree without installed metadata
    __version__ = "0.0.0"

# The Core interface versions Learn is built against — advertised here so consumers and
# the contract test cite one source of truth (defined in :mod:`astro_mine.learn._core`).
from astro_mine.learn._core import CORE_INTERFACES

# The MARL baselines (RM-P1-LEARN-03): the Learn-internal algorithm registry, the produced-
# policy Core manifest emission, and the typed export intermediate. Torch-free surface; the
# concrete Torch trainers are lazy-loaded through the registry only when a trainer is built.
from astro_mine.learn.algos import (
    COMMS_PPO_SPEC,
    IPPO_SPEC,
    MAPPO_SPEC,
    QMIX_SPEC,
    AlgorithmRegistry,
    LearnedPolicy,
    PolicyExport,
    TrainConfig,
    comms_learning_specs,
    default_registry,
    make_reference_policy,
    manifest_from_export,
    policy_manifest,
)
from astro_mine.learn.bench import ReferenceReport, evaluate, reference_score

# Curricula (learn.md §3 ``curriculum/``): hand-authored staged difficulty + domain
# randomization, with the plugin/registry seam an automatic curriculum registers through.
# Torch-free (a curriculum produces configs); ``run_curriculum`` drives a baseline through one.
from astro_mine.learn.curriculum import (
    Curriculum,
    CurriculumRegistry,
    CurriculumSpec,
    StagedCurriculum,
    comms_ladder,
    default_curriculum_registry,
    randomized_comms,
    run_curriculum,
)

# The SwarmEnv adapter (RM-P1-LEARN-01): a Core world as PettingZoo/Gymnasium RL envs.
from astro_mine.learn.envs import SwarmEnv, make_swarm_env

# The GPU-vectorized rollout tier (RM-P1-LEARN-04): the batched-world seam Sim's Brax/MJX GPU
# tier plugs into, plus Learn's JAX reference kernel. Falls back to a sequential CPU loop.
from astro_mine.learn.envs.vector import BatchedWorld, VectorExecutor

# The honest-evaluation harness (RM-P1-LEARN-06): held-out seed splits, seed sweeps with
# variance + single-seed rejection, comms-stress curves comparable across algorithms, and
# content-addressed curve aggregation (Parquet default; optional MLflow sink). Torch-free
# surface layered on the bench.evaluate rollout seam.
from astro_mine.learn.eval import (
    CommsStressGrid,
    CurveTable,
    HeldOutSplit,
    MetricSink,
    MlflowSink,
    ParquetSink,
    SweepReport,
    comms_stress_curve,
    comms_stress_curves,
    onnx_policy_id,
    onnx_policy_under_test,
    partition,
    sample_efficiency,
    seed_sweep,
)
from astro_mine.learn.track import InMemoryBackend, MlflowBackend, TrackedRun, run_provenance
from astro_mine.learn.train import LocalExecutor

__all__ = [
    "COMMS_PPO_SPEC",
    "CORE_INTERFACES",
    "IPPO_SPEC",
    "MAPPO_SPEC",
    "QMIX_SPEC",
    "AlgorithmRegistry",
    "BatchedWorld",
    "CommsStressGrid",
    "Curriculum",
    "CurriculumRegistry",
    "CurriculumSpec",
    "CurveTable",
    "HeldOutSplit",
    "InMemoryBackend",
    "LearnedPolicy",
    "LocalExecutor",
    "MetricSink",
    "MlflowBackend",
    "MlflowSink",
    "ParquetSink",
    "PolicyExport",
    "ReferenceReport",
    "StagedCurriculum",
    "SwarmEnv",
    "SweepReport",
    "TrackedRun",
    "TrainConfig",
    "VectorExecutor",
    "__version__",
    "comms_ladder",
    "comms_learning_specs",
    "comms_stress_curve",
    "comms_stress_curves",
    "default_curriculum_registry",
    "default_registry",
    "evaluate",
    "make_reference_policy",
    "make_swarm_env",
    "manifest_from_export",
    "onnx_policy_id",
    "onnx_policy_under_test",
    "partition",
    "policy_manifest",
    "randomized_comms",
    "reference_score",
    "run_curriculum",
    "run_provenance",
    "sample_efficiency",
    "seed_sweep",
]
