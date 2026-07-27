"""Host-side ONNX inference adapter — the Mind/Bench consumption seam (RM-P1-LEARN-05; §10).

Core defines :class:`~astro_mine.core.policy.OnnxPolicy` as *package + a host-supplied
``infer`` callable* — the ONNX runtime lives at the edge, not in Core (core.md §2). This is a
reference host binding: it opens an ONNX-Runtime session over an exported graph and wires it to
one agent's capability-keyed spaces (encode the Core observation → run the graph → argmax the
discrete heads / take the box means → decode back to a Core Action). The result satisfies the
Core Policy contract (``check_policy``), proving an exported PolicyPackage is consumable as a
controller by Mind, wrappable by Guard, and scoreable by Bench — the M1.2 flywheel.

**Stateful and comms-learning graphs.** The binding reads the graph's *declared* inputs rather
than assuming a lone ``obs``:

- ``hidden_in``/``hidden_out`` (a recurrent policy) — the host owns the hidden state, keyed
  **per agent**, initialized to zeros and carried across :meth:`decide` calls. This is what
  makes the exported RNN policy behave like the trained one: the belief it accumulates across
  comms-denied / masked ticks lives in the state the host feeds back. :meth:`reset_state`
  starts a fresh episode.
- ``msg`` (a comms-learning policy) — the aggregated peer-message context. A decentralized host
  running one agent's graph has no peers to aggregate, so it feeds the **zero** message: exactly
  the :class:`~astro_mine.learn.models.comms.MessageModule` semantics for an agent nothing
  reached this tick. A host that *does* have the swarm's messages passes them via
  ``message_fn``.

Needs the ``[export]`` extra (``onnxruntime``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import onnxruntime
from numpy.typing import NDArray

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.model import ActionBatch, Observation
from astro_mine.core.policy import DecisionContext, OnnxPolicy
from astro_mine.core.policy.model import PolicyPackage
from astro_mine.learn.algos.policy import action_heads, flatten_obs
from astro_mine.learn.envs.adapter.encode import decode_action, encode_observation, zero_observation
from astro_mine.learn.envs.adapter.spaces import AgentSpaceSpec
from astro_mine.learn.export.tensors import (
    HIDDEN_INPUT,
    HIDDEN_OUTPUT,
    MESSAGE_INPUT,
    OBS_INPUT,
    STATE_TENSORS,
)

__all__ = ["MessageFn", "StatefulOnnxPolicy", "onnx_action_sample", "onnx_policy"]

#: A host-supplied peer-message provider for a comms-learning graph: agent → its aggregated
#: peer-message context ``(msg_dim,)``. ``None`` (the default) feeds the zero message — the
#: honest "isolated agent" case for a decentralized single-agent host.
MessageFn = Callable[[AgentId], NDArray[np.float32]]


def onnx_action_sample(
    output_names: list[str],
    outputs: list[np.ndarray],
    discrete_heads: set[str],
) -> dict[str, Any]:
    """Decode ONNX graph outputs into a single action-sample dict.

    Discrete heads are ``argmax``'d host-side (the graph emits logits/Q-values; matching
    :meth:`DictActorCritic.forward_export`), box heads take their squashed means verbatim — the
    ``kind``/``mode``/``goto`` sample :func:`~astro_mine.learn.envs.adapter.encode.decode_action`
    consumes. Policy-**state** tensors (``hidden_out``) are not action components and are
    skipped here; the host carries them forward instead (:class:`StatefulOnnxPolicy`)."""
    sample: dict[str, Any] = {}
    for name, tensor in zip(output_names, outputs, strict=True):
        if name in STATE_TENSORS:
            continue
        row = np.asarray(tensor)[0]
        if name in discrete_heads:
            sample[name] = int(np.argmax(row))
        else:
            sample[name] = row.astype(np.float32)
    return sample


class StatefulOnnxPolicy(OnnxPolicy):
    """An :class:`OnnxPolicy` that owns the exported graph's **recurrent state**.

    The exported recurrent graph is a pure step function over an explicit
    ``hidden_in`` → ``hidden_out`` pair (export/onnx.py), so *somebody* must carry the state.
    That somebody is the host: this policy keeps one hidden vector per agent, seeds it to zeros
    (the trained :meth:`GRUCore.initial_state`), feeds it in on every :meth:`decide`, and stores
    what comes back. :meth:`reset_state` drops the belief at an episode boundary — Bench and
    Mind call it between episodes so one episode's state cannot leak into the next."""

    def __init__(
        self,
        package: PolicyPackage,
        infer: Any,
        state: dict[AgentId, NDArray[np.float32]],
        *,
        provided: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__(package, infer, provided=provided)
        self._state = state

    @property
    def hidden_state(self) -> dict[AgentId, NDArray[np.float32]]:
        """The per-agent hidden state the host is carrying (empty before the first decide)."""
        return self._state

    def reset_state(self) -> None:
        """Drop the carried recurrent state — a fresh episode starts from the zero belief."""
        self._state.clear()


def onnx_policy(
    package: PolicyPackage,
    onnx_bytes: bytes,
    spec: AgentSpaceSpec,
    *,
    provided: Mapping[str, str] | None = None,
    message_fn: MessageFn | None = None,
) -> OnnxPolicy:
    """Bind an exported graph + one agent's spec into a Core :class:`OnnxPolicy`.

    Construction negotiates the package's declared Core interface versions (loud failure on an
    incompatible policy). The returned policy decides for the agent(s) in the observation map
    using this graph; observations for a masked/unobservable agent decode from the neutral zero
    observation (no sensor leak), mirroring ``LearnedPolicy``.

    A **recurrent** graph yields a :class:`StatefulOnnxPolicy` that carries ``hidden_out`` back
    in per agent across calls; a **comms-learning** graph is fed its ``msg`` context from
    ``message_fn`` (zeros — the isolated-agent case — when none is supplied)."""
    session = onnxruntime.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    output_names = [output.name for output in session.get_outputs()]
    input_dims = {inp.name: int(inp.shape[-1]) for inp in session.get_inputs()}
    discrete_heads = set(action_heads(spec.action_space).discrete)
    recurrent = HIDDEN_INPUT in input_dims
    hidden_dim = input_dims.get(HIDDEN_INPUT, 0)
    message_dim = input_dims.get(MESSAGE_INPUT, 0)
    state: dict[AgentId, NDArray[np.float32]] = {}

    def infer(
        observations: Mapping[AgentId, Observation], _context: DecisionContext
    ) -> ActionBatch:
        actions = []
        for agent_id, obs in observations.items():
            sample = encode_observation(obs, spec) if obs.observable else zero_observation(spec)
            flat = flatten_obs(sample, spec.observation_space).reshape(1, -1).astype(np.float32)
            feed: dict[str, NDArray[np.float32]] = {OBS_INPUT: flat}
            if message_dim:
                message = (
                    message_fn(agent_id)
                    if message_fn is not None
                    else np.zeros(message_dim, dtype=np.float32)
                )
                feed[MESSAGE_INPUT] = np.asarray(message, dtype=np.float32).reshape(1, -1)
            if recurrent:
                feed[HIDDEN_INPUT] = state.setdefault(
                    agent_id, np.zeros((1, hidden_dim), dtype=np.float32)
                )
            outputs = session.run(output_names, feed)
            if recurrent:
                state[agent_id] = np.asarray(
                    outputs[output_names.index(HIDDEN_OUTPUT)], dtype=np.float32
                )
            action_sample = onnx_action_sample(output_names, outputs, discrete_heads)
            actions.append(decode_action(action_sample, spec))
        return ActionBatch(actions=actions)

    if recurrent:
        return StatefulOnnxPolicy(package, infer, state, provided=provided)
    return OnnxPolicy(package, infer, provided=provided)
