# SPDX-License-Identifier: Apache-2.0
"""``OnnxServedSurrogate`` — the ONNX-Runtime served fidelity tier (RM-P1-SURR-04).

Runs a raw-state :class:`~astro_mine.surrogate.serve.bundle.OnnxBundle` through ONNX Runtime and
presents it as a :class:`~astro_mine.surrogate.model.SurrogateModel` — the served counterpart of the
torch :class:`~astro_mine.surrogate.models.excavation.ExcavationSurrogate`, byte-for-byte
reproducible on the served path (surrogate.md §10 determinism gate). Because featurization now lives
**inside** the graph, the runtime just forwards the raw particle state (``position``/``velocity``/
``tool_x``/``config``) to ORT and maps the outputs back onto a
:class:`~astro_mine.surrogate.model.Prediction` — **no torch, no numpy featurization**: the ONNX
stack is the only inference dependency.

The session is single-threaded and deterministic — seeded/fixed inputs reproduce identical outputs,
the property Sim's golden gate pins.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from astro_mine.surrogate.model import Prediction, SurrogateState
from astro_mine.surrogate.models.excavation import _parse_state
from astro_mine.surrogate.report import ErrorReport
from astro_mine.surrogate.serve.bundle import OnnxBundle

__all__ = ["OnnxServedSurrogate"]


def _deterministic_session(model: bytes) -> ort.InferenceSession:
    """A single-threaded CPU session — deterministic inference for the golden gate."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(model, options, providers=["CPUExecutionProvider"])


class OnnxServedSurrogate:
    """A served :class:`~astro_mine.surrogate.model.SurrogateModel` backed by ONNX Runtime.

    Construct from an :class:`OnnxBundle` (or its bytes via :meth:`from_bytes`); ``predict`` returns
    the same :class:`~astro_mine.surrogate.model.Prediction` shape as the torch surrogate — per-
    particle ``position``/``velocity`` fields with conformal ``field_uncertainty``, the
    ``in_domain`` flag, and the signed ``ood_margin`` — computed entirely inside the ONNX graph.
    """

    def __init__(self, bundle: OnnxBundle) -> None:
        self._bundle = bundle
        self._session = _deterministic_session(bundle.onnx_model)

    @classmethod
    def from_bytes(cls, data: bytes) -> OnnxServedSurrogate:
        """Build a served surrogate from serialized :class:`OnnxBundle` bytes."""
        return cls(OnnxBundle.parse(data))

    @property
    def error_report(self) -> ErrorReport:
        """The static, calibrated :class:`ErrorReport` this served tier carries (its bound)."""
        return self._bundle.error_report

    def predict(self, state: SurrogateState, action: SurrogateState | None = None) -> Prediction:
        """Predict the next particle state with calibrated uncertainty via ONNX Runtime.

        Forwards the raw particle state to the self-contained graph, which builds the radius graph
        and node/edge features, runs the ensemble, and returns the mean, the conformal half-width
        (OOD-inflated outside the trust region), the ``in_domain`` flag, and the signed margin.
        """
        particle_state, tool_x, config = _parse_state(state)
        mean, half, in_domain, margin = self._session.run(
            None,
            {
                "position": particle_state[:, :2].astype(np.float32),
                "velocity": particle_state[:, 2:].astype(np.float32),
                "tool_x": np.array([tool_x], dtype=np.float32),
                "config": config.astype(np.float32),
            },
        )
        mean = mean.astype(np.float64)
        half = half.astype(np.float64)
        return Prediction(
            channels={},
            uncertainty={},
            in_domain=bool(int(in_domain)),
            ood_margin=float(margin),
            fields={"position": mean[:, :2], "velocity": mean[:, 2:]},
            field_uncertainty={"position": half[:, :2], "velocity": half[:, 2:]},
        )
