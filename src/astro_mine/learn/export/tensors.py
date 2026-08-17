# SPDX-License-Identifier: Apache-2.0
"""The exported graph's reserved tensor names — one source of truth (RM-P1-LEARN-05).

A PolicyPackage's ONNX graph declares more than ``obs`` → action heads: a **recurrent** actor
lifts its GRU state into an explicit ``hidden_in``/``hidden_out`` tensor pair, and a
**comms-learning** actor takes the aggregated peer-message context as an explicit ``msg``
input (learn.md §5). Four parties must agree on those names — the tracer
(:mod:`~astro_mine.learn.export.onnx`), the equivalence gate
(:mod:`~astro_mine.learn.export.equivalence`), the host binding
(:mod:`~astro_mine.learn.export.host`), and the eval scorer
(:mod:`~astro_mine.learn.eval.onnx`) — so they live here.

Deliberately **dependency-free** (no torch, no onnxruntime): the eval scorer imports these
names without pulling the training toolchain into the honest-eval surface.
"""

from __future__ import annotations

__all__ = ["HIDDEN_INPUT", "HIDDEN_OUTPUT", "MESSAGE_INPUT", "OBS_INPUT", "STATE_TENSORS"]

#: The flattened observation — every exported actor graph's first input.
OBS_INPUT = "obs"

#: A recurrent actor's explicit hidden-state tensors. The graph is a pure *step function* over
#: this state (no ONNX ``Loop``/``Scan``, one dynamic batch axis), so the **host** carries it
#: across calls — which is exactly what the multi-step stateful equivalence gate verifies.
HIDDEN_INPUT = "hidden_in"
HIDDEN_OUTPUT = "hidden_out"

#: A comms-learning actor's aggregated peer-message context. Declared as its own input rather
#: than smuggled inside a widened ``obs``, so the IoSignature is honest about what the policy
#: consumes; the zero vector is the isolated-agent aggregate (MessageModule semantics).
MESSAGE_INPUT = "msg"

#: Graph outputs that are **policy state**, not action components — excluded from the action
#: space descriptor and the declared action bounds Guard enforces, and carried forward by the
#: host instead of decoded into an Action.
STATE_TENSORS: frozenset[str] = frozenset({HIDDEN_OUTPUT})
