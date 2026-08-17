# SPDX-License-Identifier: Apache-2.0
"""Learned control policies — ONNX Runtime hosting (RM-P1-MIND-03).

Mind *hosts* inference, it does not train (mind.md §4): a learned policy Learn exported as an
ONNX ``PolicyPackage`` (RM-P1-LEARN-05) is loaded through Core's
:class:`~astro_mine.core.policy.onnx.OnnxPolicy` contract and run with an ONNX-Runtime session,
so it satisfies the Core Policy interface unchanged — it drops into the control tier, is
Guard-wrapped, and is scored by Bench like any controller. See
:mod:`~astro_mine.mind.control.policy.onnx_tier`. The ``onnxruntime`` runtime is the optional
``[onnx]`` extra; Mind imports no sibling package to consume the artifact.
"""

from __future__ import annotations
