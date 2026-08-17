# SPDX-License-Identifier: Apache-2.0
"""Load a served surrogate tier fail-closed, Core-side (RM-P1-SIM-03; sim.md §5, §6).

Sim consumes the [Surrogate](surrogate.md) tier as a **content-addressed ONNX artifact** through
Core contracts — never by importing ``astro_mine.surrogate``. The artifact is the raw-state
``OnnxBundle`` (a zip of ``model.onnx`` + ``serve_meta.json`` + ``error_report.json``) plus its
signed Core :class:`~astro_mine.core.registry.PluginManifest`. This loader verifies the signature
and the content hashes **before** building the ONNX Runtime session (mirroring the surrogate's own
load gate), reads the admission budget straight from the manifest attributes, and exposes a raw-
state ``step`` that emits next-state + calibrated uncertainty + the trust-region flag.

onnxruntime is the ``[surrogate]`` extra; hub's verifier is the ``[hub]`` extra — both imported
lazily so the base engine set stays free of them.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from astro_mine.core.hashing import content_hash
from astro_mine.core.registry import PluginManifest, PluginRegistry

if TYPE_CHECKING:
    import numpy.typing as npt

    from astro_mine.core.registry import Verifier

    FloatArray = npt.NDArray[np.float64]

__all__ = ["LoadedSurrogate", "SurrogateIntegrityError", "load_surrogate_tier"]

_MODEL_ENTRY = "model.onnx"
_META_ENTRY = "serve_meta.json"


class SurrogateIntegrityError(Exception):
    """The signed manifest does not match the delivered bundle — a swapped or corrupt artifact."""


@dataclass(frozen=True)
class SurrogateStep:
    """One surrogate step: next per-particle state, calibrated uncertainty, and the trust flag."""

    next_pos: FloatArray  # (N, 2)
    next_vel: FloatArray  # (N, 2)
    pos_uncertainty: FloatArray  # (N, 2)
    vel_uncertainty: FloatArray  # (N, 2)
    in_domain: bool
    ood_margin: float


class LoadedSurrogate:
    """A verified, ONNX-Runtime-backed surrogate tier ready to step a particle bed.

    ``recommended_error_budget`` is the per-output-channel admission budget (from the manifest, read
    by the scheduler); ``step`` runs the raw-state graph on a bed. Built only via
    :func:`load_surrogate_tier`, which fails closed on a bad signature or hash.

    ``trust_region`` is the input domain the tier was *trained* on, as its manifest declares it
    (``{channel: {"low": .., "high": ..}}``). The graph already enforces it per query — that is what
    :attr:`SurrogateStep.in_domain` reports — but the band is also readable here, because a caller
    that is *choosing* how to exercise the tier needs to know where it is valid **before** stepping
    it. Benchmarking a surrogate outside its trust region measures nothing: every query is
    out-of-domain, the engine escalates to the reference solver, and the "speedup" is the reference
    solver against itself."""

    def __init__(
        self,
        session: Any,
        *,
        name: str,
        recommended_error_budget: dict[str, float],
        input_config_order: list[str],
        trust_region: dict[str, dict[str, float]] | None = None,
        budget_horizon_steps: int = 1,
    ) -> None:
        self._session = session
        self.name = name
        self.recommended_error_budget = recommended_error_budget
        self.input_config_order = input_config_order
        self.trust_region = trust_region or {}
        #: The autoregressive rollout horizon (steps) the ``recommended_error_budget`` was
        #: calibrated to hold at (astro-mine-surrogate#23). A caller that re-validates *less* often
        #: than this grades the tier over a longer rollout than its budget covers — the adaptive
        #: engine refuses that, rather than silently checking a bound the producer never made.
        self.budget_horizon_steps = budget_horizon_steps

    def step(
        self, pos: FloatArray, vel: FloatArray, tool_x_m: float, config: FloatArray
    ) -> SurrogateStep:
        """Run the raw-state ONNX graph on a bed → next state + uncertainty + trust flag."""
        mean, half, in_domain, margin = self._session.run(
            None,
            {
                "position": pos.astype(np.float32),
                "velocity": vel.astype(np.float32),
                "tool_x": np.asarray([tool_x_m], dtype=np.float32),
                "config": config.astype(np.float32),
            },
        )
        mean = mean.astype(np.float64)
        half = half.astype(np.float64)
        return SurrogateStep(
            next_pos=mean[:, :2],
            next_vel=mean[:, 2:],
            pos_uncertainty=half[:, :2],
            vel_uncertainty=half[:, 2:],
            in_domain=bool(int(in_domain)),
            ood_margin=float(margin),
        )


def load_surrogate_tier(
    bundle_bytes: bytes,
    manifest: PluginManifest,
    *,
    verifier: Verifier | None = None,
) -> LoadedSurrogate:
    """Verify the signed manifest against ``bundle_bytes`` and build the ONNX-Runtime tier.

    Fail-closed in order: (1) the Core registry signature gate rejects an unsigned/tampered/
    untrusted-key manifest; (2) ``content_hash(bundle_bytes)`` must equal the signed
    ``provenance.digest``. Only then is the ORT session built from the bundle's ``model.onnx``.
    ``verifier`` is a trusted-key verifier (``astro_mine.hub.supply_chain.make_verifier``); without
    one the registry only checks a signature is present."""
    import onnxruntime as ort

    PluginRegistry(require_signature=True, verifier=verifier).register(manifest)
    if manifest.provenance is None or content_hash(bundle_bytes) != manifest.provenance.digest:
        raise SurrogateIntegrityError(
            "bundle content hash does not match the signed manifest provenance.digest"
        )
    with zipfile.ZipFile(io.BytesIO(bundle_bytes)) as archive:
        onnx_model = archive.read(_MODEL_ENTRY)
        meta = json.loads(archive.read(_META_ENTRY))

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(onnx_model, options, providers=["CPUExecutionProvider"])

    budget = dict(manifest.attributes.get("recommended_error_budget", {}))
    if not budget:
        raise SurrogateIntegrityError("manifest carries no recommended_error_budget for admission")
    trust_region = dict(manifest.attributes.get("trust_region", {}).get("bounds", {}))
    return LoadedSurrogate(
        session,
        name=manifest.name,
        recommended_error_budget=budget,
        input_config_order=list(meta.get("param_names", [])),
        trust_region=trust_region,
        budget_horizon_steps=int(manifest.attributes.get("budget_horizon_steps", 1)),
    )
