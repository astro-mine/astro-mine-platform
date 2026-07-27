"""Score an exported ONNX ``PolicyPackage`` through the same rollout as a live policy (LEARN-06).

The honest-eval harness scores **live** policies and **exported ONNX packages** identically —
the same held-out seed sweep and comms-stress curve run over both (issue AC; learn.md §10). A
live :class:`~astro_mine.learn.algos.policy.LearnedPolicy` already exposes the
:class:`~astro_mine.learn.eval.PolicyUnderTest` ``act`` contract; this module adapts one or more
exported graphs into the *same* shape by wiring an onnxruntime session per agent behind a flat-obs
:data:`~astro_mine.learn.algos.policy.InferFn` (reusing
:func:`~astro_mine.learn.export.host.onnx_action_sample` to decode the graph outputs), so a
scored package flows through the executor rollout with no ONNX-specific eval path.

``onnxruntime`` (the ``[export]`` extra) and the host decoder are imported **lazily**, so this
module — and the whole ``eval`` surface — imports without the extra installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from astro_mine.core.env.model import AgentId
from astro_mine.core.hashing import content_hash_json
from astro_mine.core.policy.model import PolicyPackage
from astro_mine.learn.algos.policy import InferFn, LearnedPolicy, action_heads
from astro_mine.learn.envs.adapter.spaces import AgentSpaceSpec
from astro_mine.learn.export.tensors import HIDDEN_INPUT, HIDDEN_OUTPUT, MESSAGE_INPUT, OBS_INPUT

__all__ = ["OnnxGraph", "onnx_policy_id", "onnx_policy_under_test"]


@dataclass(frozen=True)
class OnnxGraph:
    """One agent's exported artifact: the Core :class:`PolicyPackage` sidecar + its ONNX bytes.

    Heterogeneous agents each carry their own graph (one ``obs_dim`` per agent), so a scoreable
    package set is a per-agent mapping of these."""

    package: PolicyPackage
    onnx_bytes: bytes


def _onnx_infer(onnx_bytes: bytes, spec: AgentSpaceSpec) -> InferFn:
    """Build a flat-obs inference callable backed by a cached onnxruntime session (lazy import).

    Reuses :func:`~astro_mine.learn.export.host.onnx_action_sample` (the same argmax/box decode
    the Core ``OnnxPolicy`` host binding uses), so a package scores exactly as it will run.

    Binds whatever tensors the graph *declares*, so a **recurrent** package scores honestly: the
    closure carries ``hidden_out`` back in as the next call's ``hidden_in`` (a fresh eval rollout
    starts from the zero belief), exactly as ``export/host.py``'s
    :class:`~astro_mine.learn.export.host.StatefulOnnxPolicy` does — otherwise an RNN policy
    would be scored as if it had amnesia. A **comms-learning** package is fed the zero ``msg``
    (the isolated-agent aggregate) unless the swarm's messages are supplied."""
    import onnxruntime  # lazy: the [export] extra, never required to import this module

    from astro_mine.learn.export.host import onnx_action_sample

    session = onnxruntime.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    output_names = [output.name for output in session.get_outputs()]
    input_dims = {inp.name: int(inp.shape[-1]) for inp in session.get_inputs()}
    discrete_heads = set(action_heads(spec.action_space).discrete)
    hidden_dim = input_dims.get(HIDDEN_INPUT, 0)
    message_dim = input_dims.get(MESSAGE_INPUT, 0)
    carried: dict[str, NDArray[np.float32]] = {}

    def infer(flat_obs: NDArray[np.float32]) -> Mapping[str, object]:
        inputs: dict[str, NDArray[np.float32]] = {
            OBS_INPUT: flat_obs.reshape(1, -1).astype(np.float32)
        }
        if message_dim:
            inputs[MESSAGE_INPUT] = np.zeros((1, message_dim), dtype=np.float32)
        if hidden_dim:
            inputs[HIDDEN_INPUT] = carried.setdefault(
                HIDDEN_INPUT, np.zeros((1, hidden_dim), dtype=np.float32)
            )
        outputs = session.run(output_names, inputs)
        if hidden_dim:
            carried[HIDDEN_INPUT] = np.asarray(
                outputs[output_names.index(HIDDEN_OUTPUT)], dtype=np.float32
            )
        return onnx_action_sample(output_names, outputs, discrete_heads)

    return infer


def onnx_policy_under_test(
    graphs: Mapping[AgentId, OnnxGraph], specs: Mapping[AgentId, AgentSpaceSpec]
) -> LearnedPolicy:
    """Adapt a per-agent set of exported ONNX graphs into a :class:`PolicyUnderTest`.

    Returns a :class:`~astro_mine.learn.algos.policy.LearnedPolicy` whose per-agent ``infer``
    runs the agent's onnxruntime session — so the scored package drives the *same*
    :func:`~astro_mine.learn.bench.evaluate` rollout as a live policy. A missing spec for an
    agent in ``graphs`` raises ``KeyError`` (loud, not a silent skip)."""
    infer: dict[AgentId, InferFn] = {
        agent: _onnx_infer(graph.onnx_bytes, specs[agent]) for agent, graph in graphs.items()
    }
    return LearnedPolicy({agent: specs[agent] for agent in graphs}, infer)


def onnx_policy_id(graphs: Mapping[AgentId, OnnxGraph]) -> str:
    """A deterministic content-addressed id for a scored package set (its ``policy_id``).

    Hashes the per-agent graph digests (``ModelRef.digest``) so the id is stable and unique to
    the exact artifacts scored — the ``policy_id`` that lands on every curve row."""
    return content_hash_json(
        {agent: graph.package.onnx_model.digest for agent, graph in sorted(graphs.items())}
    )
