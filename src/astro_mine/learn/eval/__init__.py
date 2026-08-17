# SPDX-License-Identifier: Apache-2.0
"""Honest-evaluation harness — held-out envs, seed sweeps, comms-stress curves (LEARN-06).

The harness that makes a reported score statistically honest (learn.md §10; charter §8):

- :mod:`~astro_mine.learn.eval.split` — held-out evaluation seeds **separated from training by
  construction** (:class:`HeldOutSplit`, :func:`partition`).
- :mod:`~astro_mine.learn.eval.sweep` — seed sweeps with variance and **single-seed rejection**
  (:func:`seed_sweep`), reporting sample-efficiency and wall-clock alongside reward.
- :mod:`~astro_mine.learn.eval.aggregate` — the versioned long-format curve schema, its
  content-addressed :class:`CurveTable`, and the :class:`MetricSink` seam (default
  :class:`ParquetSink`; optional :class:`MlflowSink`).

Layered strictly on top of the :func:`~astro_mine.learn.bench.evaluate` rollout seam — it reuses
the tier-1 :class:`~astro_mine.learn.train.executor.LocalExecutor` path, consumes only
``astro_mine.core`` + Learn-internal modules, and keeps ONNX/pyarrow/MLflow imports lazy.
"""

from __future__ import annotations

from astro_mine.learn.eval.aggregate import (
    CURVE_SCHEMA_VERSION,
    CurveRow,
    CurveTable,
    MetricSink,
    MlflowSink,
    ParquetSink,
)
from astro_mine.learn.eval.comms_stress import (
    CommsEnvFactory,
    CommsStressGrid,
    StressPoint,
    build_curve_manifest,
    comms_stress_curve,
    comms_stress_curves,
)
from astro_mine.learn.eval.onnx import OnnxGraph, onnx_policy_id, onnx_policy_under_test
from astro_mine.learn.eval.split import HeldOutSplit, partition
from astro_mine.learn.eval.sweep import (
    PolicyUnderTest,
    SweepReport,
    sample_efficiency,
    seed_sweep,
)

__all__ = [
    "CURVE_SCHEMA_VERSION",
    "CommsEnvFactory",
    "CommsStressGrid",
    "CurveRow",
    "CurveTable",
    "HeldOutSplit",
    "MetricSink",
    "MlflowSink",
    "OnnxGraph",
    "ParquetSink",
    "PolicyUnderTest",
    "StressPoint",
    "SweepReport",
    "build_curve_manifest",
    "comms_stress_curve",
    "comms_stress_curves",
    "onnx_policy_id",
    "onnx_policy_under_test",
    "partition",
    "sample_efficiency",
    "seed_sweep",
]
