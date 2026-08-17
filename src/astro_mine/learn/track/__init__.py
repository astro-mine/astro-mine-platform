# SPDX-License-Identifier: Apache-2.0
"""Experiment tracking & provenance capture (learn.md §3 ``track/``; §4, §11).

learn.md §3's module tree lists ``track/`` — *"experiment tracking (MLflow default / W&B
option), provenance capture"* — and §11 recommends **MLflow as the OSS default**. This is that
module.

Tracking here serves **reproducibility**, not dashboards. learn.md §2.4: *"Every run records its
inputs, code version, lockfile, and seeds; results are content-addressed so Bench can re-derive
them."* So:

- :class:`TrackedRun` captures the :class:`TrainConfig`, the :class:`CommsModelConfig`, the
  :class:`CurriculumSpec`, the seeds, the toolchain, and the lockfile hash, and content-addresses
  the lot into :attr:`TrackedRun.run_hash` — **the** reproducibility key. It streams the
  per-iteration learning curve and the honest-eval :class:`CurveTable` into that same run, and
  links the produced policy's ONNX digests back to it (learn.md §4: "Runs link to Bench results
  and Hub artifacts by content hash").
- :class:`TrackingBackend` is the seam: :class:`MlflowBackend` (the default, behind the optional
  ``[mlflow]`` extra, lazily imported) or :class:`InMemoryBackend` (dependency-free — a tier-1
  workstation run gets its full provenance record with no server and no network).

The pre-existing :class:`~astro_mine.learn.eval.MlflowSink` — the narrow "mirror a CurveTable
into MLflow" seam — is now a thin adapter **over** :class:`MlflowBackend`, so there is exactly
one MLflow implementation in the package.
"""

from __future__ import annotations

from astro_mine.learn.track.backends import InMemoryBackend, MlflowBackend, TrackingBackend
from astro_mine.learn.track.run import (
    RUN_RECORD_VERSION,
    TrackedRun,
    run_provenance,
    tracked_run,
)

__all__ = [
    "RUN_RECORD_VERSION",
    "InMemoryBackend",
    "MlflowBackend",
    "TrackedRun",
    "TrackingBackend",
    "run_provenance",
    "tracked_run",
]
