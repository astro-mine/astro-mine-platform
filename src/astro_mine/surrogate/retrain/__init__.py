# SPDX-License-Identifier: Apache-2.0
"""``retrain`` — offline retrain + gated promotion (RM-P1-SURR-03; surrogate.md §5, §11).

The build loop's terminal step: :func:`retrain_surrogate` trains a new SemVer surrogate on a
(possibly resampled) dataset, gates it on calibration/coverage + error budget, and — only on a
pass — exports the served bundle with full reproduction provenance. The prior version is never
overwritten.

Importing this subpackage pulls the ``[datasets]`` extra (via the dataset store) and, on a gate
pass, the ``[serve]`` extra (ONNX export, imported lazily); it never imports ``astro_mine.sim``.
"""

from __future__ import annotations

from astro_mine.surrogate.retrain.harness import BumpKind, RetrainResult, retrain_surrogate

__all__ = ["BumpKind", "RetrainResult", "retrain_surrogate"]
