"""Serve: ONNX export + ONNX-Runtime inference for the learned fidelity tier (RM-P1-SURR-04).

The inline **use** loop of surrogate.md §3: a trained
:class:`~astro_mine.surrogate.models.excavation.ExcavationSurrogate` is exported to a self-contained
ONNX graph (:func:`export_excavation_surrogate`), packaged as a content-addressed
:class:`OnnxBundle`, published + signed to Hub (:func:`publish_served_surrogate`), and loaded back
fail-closed (:func:`load_served_surrogate`) as an :class:`OnnxServedSurrogate` that Sim's scheduler
drives via a :class:`ServedTier`. Requires the ``[serve]`` extra (``onnx``/``onnxruntime``); the
base package import never pulls the ONNX stack.
"""

from __future__ import annotations

from astro_mine.surrogate.serve.bundle import (
    ONNX_BUNDLE_MEDIA_TYPE,
    SERVE_META_FORMAT_VERSION,
    OnnxBundle,
)
from astro_mine.surrogate.serve.export import OnnxExportError, export_excavation_surrogate
from astro_mine.surrogate.serve.load import (
    ServedIntegrityError,
    load_served_surrogate,
    resolve_and_load,
)
from astro_mine.surrogate.serve.publish import PublishedSurrogate, publish_served_surrogate
from astro_mine.surrogate.serve.runtime import OnnxServedSurrogate
from astro_mine.surrogate.serve.tier import ServedTier

__all__ = [
    "ONNX_BUNDLE_MEDIA_TYPE",
    "SERVE_META_FORMAT_VERSION",
    "OnnxBundle",
    "OnnxExportError",
    "OnnxServedSurrogate",
    "PublishedSurrogate",
    "ServedIntegrityError",
    "ServedTier",
    "export_excavation_surrogate",
    "load_served_surrogate",
    "publish_served_surrogate",
    "resolve_and_load",
]
