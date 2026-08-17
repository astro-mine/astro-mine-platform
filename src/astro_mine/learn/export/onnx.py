# SPDX-License-Identifier: Apache-2.0
"""ONNX-graph construction for the PolicyPackage export (RM-P1-LEARN-05; learn.md §5, §10).

The ONNX graph is the **only** cross-component policy artifact (learn.md §5): a Learn-trained
actor is traced into a portable, framework-neutral graph that Mind hosts (ONNX Runtime), Guard
wraps, and Bench scores through the Core Policy contract. This module traces one net's
:meth:`forward_export` — the tensor-only, sampling-free actor forward — into serialized ONNX
bytes with a **pinned opset**, so the same weights yield the same graph (its content hash is
the artifact identity; RM-P1-LEARN-05 §5). The critic never crosses the waist; only the
decentralized actor is exported (one graph per heterogeneous agent).

**Stateful and comms-learning actors.** The graph's IO is whatever the net *declares* through
``export_input_specs`` / ``export_output_specs``, not a fixed ``obs`` → heads shape:

- a **recurrent** (GRU) actor adds an explicit ``hidden_in`` input and ``hidden_out`` output —
  the GRU state is lifted out of the module and into the tensor contract, which is what keeps
  the graph expressible at the pinned opset with a *single* dynamic (batch) axis: no ONNX
  ``Loop``, no sequence-length axis, and a host that carries the state across calls (the
  multi-step equivalence check in :mod:`~astro_mine.learn.export.equivalence` exercises
  exactly that);
- a **comms-learning** actor adds an explicit ``msg`` input — the aggregated peer-message
  context — so the declared IoSignature is honest about what the policy consumes rather than
  smuggling the message inside a widened ``obs``.

Every axis except the batch is static, so the graph stays byte-stable for fixed weights.

Needs the ``[export]`` extra (``onnx``); imported only from the export path.
"""

from __future__ import annotations

import io
import warnings
from typing import Any, cast

import torch
from torch import nn

from astro_mine.learn.export.tensors import HIDDEN_INPUT, OBS_INPUT

__all__ = [
    "DEFAULT_OPSET",
    "dummy_inputs",
    "input_specs_for",
    "is_recurrent_graph",
    "output_names_for",
    "output_specs_for",
    "to_onnx_bytes",
]

#: The pinned ONNX opset the exported graph targets. Fixed so the serialized graph — hence its
#: content-hash identity (RM-P1-LEARN-05) — is stable across runs for the same weights. Opset 17
#: expresses the GRU cell's Gemm/Sigmoid/Tanh decomposition, so a recurrent actor needs no opset
#: bump — only the explicit hidden-state tensors :mod:`~astro_mine.learn.export.tensors` names.
DEFAULT_OPSET = 17


def _spec_net(net: nn.Module) -> Any:
    """The net's IO-declaration surface. ``export_input_specs``/``export_output_specs`` are
    duck-typed on the actor nets; ``nn.Module.__getattr__`` is typed ``Tensor | Module``, so
    reach them through ``Any`` (as ``forward_export`` already is)."""
    if not hasattr(net, "export_input_specs") or not hasattr(net, "export_output_specs"):
        raise TypeError(f"cannot derive an ONNX IO signature for {type(net).__name__}")
    return cast(Any, net)


def input_specs_for(net: nn.Module) -> list[tuple[str, int]]:
    """The ordered ``(name, width)`` ONNX graph inputs the net declares."""
    specs: list[tuple[str, int]] = _spec_net(net).export_input_specs()
    return specs


def output_specs_for(net: nn.Module) -> list[tuple[str, int]]:
    """The ordered ``(name, width)`` ONNX graph outputs the net declares."""
    specs: list[tuple[str, int]] = _spec_net(net).export_output_specs()
    return specs


def output_names_for(net: nn.Module) -> list[str]:
    """The ONNX output tensor names, in :meth:`forward_export` order.

    ``DictActorCritic`` emits one logits tensor per discrete head then one squashed-mean tensor
    per box head (sorted keys), plus ``hidden_out`` when recurrent; ``AgentQNet`` emits the
    per-``kind`` Q-values as ``kind``. These are the head names the host maps back to Core
    action components (``decode_action``) — minus the state tensors, which it carries forward."""
    return [name for name, _dim in output_specs_for(net)]


def is_recurrent_graph(net: nn.Module) -> bool:
    """Whether the exported graph carries an explicit hidden state (a recurrent actor)."""
    return HIDDEN_INPUT in {name for name, _dim in input_specs_for(net)}


def dummy_inputs(net: nn.Module, obs_dim: int) -> tuple[torch.Tensor, ...]:
    """Zero tracing inputs matching the net's declared input specs (batch of 1).

    ``obs_dim`` overrides the declared ``obs`` width so the caller stays the single source of
    truth for the flattened observation size (it comes from the env's spaces, not the net)."""
    dims = [obs_dim if name == OBS_INPUT else dim for name, dim in input_specs_for(net)]
    return tuple(torch.zeros(1, dim, dtype=torch.float32) for dim in dims)


class _ExportModule(nn.Module):
    """Adapts a policy net's :meth:`forward_export` to a ``forward`` for ``torch.onnx.export``
    (the actor nets deliberately have no ``forward``— sampling lives in ``act``).

    The forward takes the net's declared inputs positionally. The optional slots are resolved
    at *trace* time from the example tuple's arity — the traced graph therefore has exactly as
    many inputs as the net declares, and no Python-level branch survives into the graph."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net

    def forward(
        self,
        obs: torch.Tensor,
        extra_a: torch.Tensor | None = None,
        extra_b: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, ...]:
        # forward_export is duck-typed on our actor nets; nn.Module.__getattr__ is typed as
        # Tensor|Module, so reach it through Any.
        net = cast(Any, self.net)
        inputs = [obs]
        if extra_a is not None:
            inputs.append(extra_a)
        if extra_b is not None:
            inputs.append(extra_b)
        result: tuple[torch.Tensor, ...] = net.forward_export(*inputs)
        return result


def to_onnx_bytes(net: nn.Module, obs_dim: int, *, opset: int = DEFAULT_OPSET) -> bytes:
    """Trace ``net``'s actor forward into serialized ONNX bytes with a pinned opset.

    The graph's inputs and outputs are exactly what the net declares (:func:`input_specs_for` /
    :func:`output_specs_for`): always the float ``obs`` (shape ``[batch, obs_dim]``), plus the
    ``msg`` and ``hidden_in`` tensors for a comms-learning / recurrent actor, mapping to the
    per-head outputs and (when recurrent) ``hidden_out``.

    **Dynamic axes:** the batch is the *only* dynamic axis — on every input **and** every
    output, including the hidden state. Everything else (obs width, message width, hidden size,
    head cardinality) is static, which is what makes the recurrent graph a plain feed-forward
    step function over an explicit state rather than an opset-fragile ``Loop``/``Scan``. The net
    is put in eval mode and traced with zero dummy inputs; the graph is float-only, so
    ONNX-Runtime output matches the Torch forward to floating-point tolerance (checked by
    :mod:`~astro_mine.learn.export.equivalence`, single-shot *and* over a multi-step stateful
    rollout)."""
    input_names = [name for name, _dim in input_specs_for(net)]
    output_names = output_names_for(net)
    module = _ExportModule(net).eval()
    dummies = dummy_inputs(net, obs_dim)
    dynamic_axes = {name: {0: "batch"} for name in (*input_names, *output_names)}
    buffer = io.BytesIO()
    # torch.onnx.export accepts a file-like object at runtime; its stub only types a path, so
    # pass it through Any.
    sink: Any = buffer
    # Use the legacy TorchScript exporter (dynamo=False): under the pinned Torch it produces a
    # byte-stable, opset-pinned graph with no extra dependency (the dynamo exporter needs
    # onnxscript). Migrating to the dynamo exporter — once its byte-stability is validated — is
    # a follow-up; suppress its deprecation notice here to keep the export path quiet.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        torch.onnx.export(
            module,
            dummies,
            sink,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            dynamo=False,
        )
    return buffer.getvalue()
