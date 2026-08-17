# SPDX-License-Identifier: Apache-2.0
"""ONNX-controller hosting through the Core OnnxPolicy contract (RM-P1-MIND-03).

Binds a Learn-exported ONNX ``PolicyPackage`` to Core's
:class:`~astro_mine.core.policy.onnx.OnnxPolicy`: Core owns the adapter shape but never runs the
graph, so Mind (the host) supplies the ``infer`` callable that drives an ONNX-Runtime session.
:func:`onnx_controller` maps each agent's position error toward its GOTO target through the
graph to a clamped velocity setpoint — the learned counterpart of the classical
:class:`~astro_mine.mind.control.reference.PidController`, behind the identical Core Controller
contract, so it drops into the control tier with no framework change and is Guard-wrapped like
any other. ``onnxruntime`` is imported lazily (the optional ``[onnx]`` extra); no sibling
package is imported to consume the artifact.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.model import ActionBatch, Observation
from astro_mine.core.policy.model import DecisionContext, ModelRef, PolicyPackage
from astro_mine.core.policy.onnx import InferenceFn, OnnxPolicy
from astro_mine.core.policy.protocol import Policy
from astro_mine.core.registry.loader import load_manifest
from astro_mine.core.registry.model import PluginManifest
from astro_mine.core.registry.tier import TierPlugin
from astro_mine.mind.control.reference import _clamp, _target_for, _velocity_action

__all__ = ["content_digest", "onnx_control_plugin", "onnx_controller"]


def content_digest(onnx_bytes: bytes) -> str:
    """The content hash identifying an ONNX graph (a PolicyPackage is content-addressed)."""
    return f"sha256:{hashlib.sha256(onnx_bytes).hexdigest()}"


def onnx_controller(
    package: PolicyPackage,
    onnx_bytes: bytes,
    *,
    max_speed_mps: float = 2.0,
    provided: Mapping[str, str] | None = None,
) -> OnnxPolicy:
    """Load ``onnx_bytes`` as a control-tier :class:`OnnxPolicy` for the toy prospecting IO.

    The host inference maps each agent's ``[error_x, error_y]`` (target minus position) through the
    graph to a ``[vx, vy]`` velocity, clamped to ``max_speed_mps``. Construction negotiates the
    package's ``core_interfaces`` against this Core, so an incompatible policy fails at load."""
    import numpy as np  # onnxruntime pulls numpy; both arrive with the [onnx] extra
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    def infer(observations: Mapping[AgentId, Observation], context: DecisionContext) -> ActionBatch:
        upstream = context.upstream if context.upstream is not None else ActionBatch()
        by_agent = {action.agent_id: action for action in upstream.actions}
        actions = []
        for agent_id in sorted(observations):
            position = observations[agent_id].self_state.pose.translation_m
            target = _target_for(by_agent.get(agent_id), position)
            features = np.asarray(
                [[target.x - position.x, target.y - position.y]], dtype=np.float32
            )
            output = session.run([output_name], {input_name: features})[0]
            vx = _clamp(float(output[0][0]), max_speed_mps)
            vy = _clamp(float(output[0][1]), max_speed_mps)
            actions.append(_velocity_action(agent_id, vx, vy))
        return ActionBatch(actions=actions)

    infer_fn: InferenceFn = infer
    return OnnxPolicy(package, infer_fn, provided=provided)


def _manifest(filename: str) -> PluginManifest:
    text = (
        resources.files("astro_mine.mind.reference")
        .joinpath("manifests", filename)
        .read_text(encoding="utf-8")
    )
    return load_manifest(text).manifest


def _package_from_params(params: Mapping[str, Any], onnx_bytes: bytes) -> PolicyPackage:
    return PolicyPackage(
        name=str(params.get("name", "mind.control.onnx")),
        version=str(params.get("version", "0.1.0")),
        onnx_model=ModelRef(digest=content_digest(onnx_bytes)),
        core_interfaces={"policy": "0.1.0", "messages": "0.1.0"},
    )


def _onnx_factory(params: Mapping[str, Any]) -> Policy:
    model_path = params.get("model_path")
    if not model_path:
        raise ValueError(
            "mind.control.onnx requires a 'model_path' param pointing to the exported ONNX "
            "PolicyPackage graph (Learn RM-P1-LEARN-05); it is a template plugin"
        )
    onnx_bytes = Path(str(model_path)).read_bytes()
    return onnx_controller(
        _package_from_params(params, onnx_bytes),
        onnx_bytes,
        max_speed_mps=float(params.get("max_speed_mps", 2.0)),
    )


def onnx_control_plugin() -> TierPlugin:
    """Provider for the ONNX control-tier plugin (entry point). The factory binds a graph named
    by a ``model_path`` param — a stack points it at a Learn-exported artifact."""
    return TierPlugin(manifest=_manifest("onnx_control.yaml"), factory=_onnx_factory)
