"""PolicyPackage export — the one cross-component artifact Learn emits (RM-P1-LEARN-05).

Renders a trained policy's typed :class:`~astro_mine.learn.algos.PolicyExport` intermediate
into a Core :class:`~astro_mine.core.policy.PolicyPackage`: an ONNX graph plus the typed
metadata sidecar Mind hosts, Guard wraps, and Bench scores (learn.md §5, §10). The ONNX graph
is the *only* artifact that crosses the waist — framework-native checkpoints stay internal.

- :func:`export_policy_package` / :func:`export_policy_packages` — render an actor into a Core
  PolicyPackage document, gated by the ONNX-Runtime equivalence check; the graph digest is the
  content-addressed identity. **Feed-forward, recurrent (GRU) and comms-learning actors all
  export**: a recurrent policy's hidden state and a comms-learning policy's aggregated
  peer-message context are declared as explicit graph tensors
  (:mod:`~astro_mine.learn.export.tensors`), so the IoSignature states the full contract a host
  must bind — and a recurrent export additionally passes a **multi-step stateful** equivalence
  gate that carries the state across steps.
- :func:`publish` — write the graph + sidecar to a content-addressed store (Bench resolves it
  by hash), with an optional Hub handoff (no hard Hub dependency).
- :func:`onnx_policy` — bind an exported graph to the Core Policy contract (:class:`OnnxPolicy`)
  so Mind/Guard/Bench consume it — the M1.2 flywheel. A recurrent graph yields a
  :class:`StatefulOnnxPolicy` that owns the per-agent hidden state across decisions.

Everything above needs the ``[export]`` extra (``onnx``/``onnxruntime``) and ``[rllib]``
(``torch``) — **and is therefore resolved lazily**.

**Why this module is lazy, and what it fixes.** Importing *any* submodule runs this file first.
So while :mod:`~astro_mine.learn.export.tensors` imports nothing at all — it is four tensor-name
constants, the vocabulary a host binds — reaching it used to execute the eager re-exports below
and pull in ``onnxruntime`` and ``torch`` with them. That made
``from astro_mine.learn.export.tensors import OBS_INPUT`` cost the whole export stack, which in
turn broke :mod:`astro_mine.learn.eval` (whose ``onnx`` module wants only those constants and
carefully defers the runtime), which broke ``astro_mine.learn`` itself. A base install could not
import the package at all, and ``[export]`` — declared precisely so *"a policy consumer needs
neither"* — bought nothing (astro-mine-learn#30; ``conventions.md §7``, the tier that MUST
always work).

The constants stay eager because they cost nothing. The machinery resolves on first attribute
access, so a name that genuinely needs the extras fails **when it is used**, naming the extra —
not when an unrelated import happens to traverse this package.

This is the same rule the rest of Learn already follows for its optional trees, applied at the
level where the problem actually is: :mod:`astro_mine.learn.track` defers ``mlflow`` into a
constructor, :mod:`astro_mine.learn.envs.vector.jax_world` defers ``jax`` the same way. Those
are leaf dependencies used inside functions, so a function-level import suffices. Here the
barrier is a package ``__init__`` re-export, and PEP 562 is the tool that matches it — the leaf
modules keep their honest module-level ``import onnxruntime``/``import torch``, because a module
that cannot do its job without them should say so.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

# Dependency-free by construction (`tensors` imports nothing but `__future__`): the graph-tensor
# names are the contract vocabulary a host must bind, not export machinery. Keeping them eager is
# what lets `eval.onnx` — and anything else that only needs to *name* a tensor — import this
# package without the extras. A dependency must never be added to `tensors`.
from astro_mine.learn.export.tensors import (
    HIDDEN_INPUT,
    HIDDEN_OUTPUT,
    MESSAGE_INPUT,
    OBS_INPUT,
    STATE_TENSORS,
)

if TYPE_CHECKING:
    # Re-imported for type checkers only, so `from astro_mine.learn.export import onnx_policy`
    # keeps its real signature under mypy even though it resolves through `__getattr__` at
    # runtime. A lazy surface must not become an untyped one.
    from astro_mine.learn.export.equivalence import (
        EquivalenceError,
        assert_onnx_equivalence,
        assert_onnx_stateful_equivalence,
        fixed_input_batch,
        fixed_obs_batch,
    )
    from astro_mine.learn.export.host import (
        MessageFn,
        StatefulOnnxPolicy,
        onnx_action_sample,
        onnx_policy,
    )
    from astro_mine.learn.export.onnx import (
        DEFAULT_OPSET,
        input_specs_for,
        is_recurrent_graph,
        output_specs_for,
        to_onnx_bytes,
    )
    from astro_mine.learn.export.package import (
        ExportedPolicy,
        content_id,
        export_policy_package,
        export_policy_packages,
    )
    from astro_mine.learn.export.publish import PublishedPolicy, publish

#: Lazily-resolved name → the submodule that defines it. Every one of these needs ``[export]``
#: and/or ``[rllib]``; none of them is needed to *reach* this package.
_LAZY: dict[str, str] = {
    "EquivalenceError": "equivalence",
    "assert_onnx_equivalence": "equivalence",
    "assert_onnx_stateful_equivalence": "equivalence",
    "fixed_input_batch": "equivalence",
    "fixed_obs_batch": "equivalence",
    "MessageFn": "host",
    "StatefulOnnxPolicy": "host",
    "onnx_action_sample": "host",
    "onnx_policy": "host",
    "DEFAULT_OPSET": "onnx",
    "input_specs_for": "onnx",
    "is_recurrent_graph": "onnx",
    "output_specs_for": "onnx",
    "to_onnx_bytes": "onnx",
    "ExportedPolicy": "package",
    "content_id": "package",
    "export_policy_package": "package",
    "export_policy_packages": "package",
    "PublishedPolicy": "publish",
    "publish": "publish",
}

#: The optional distributions the lazy submodules import, and the extra that supplies each —
#: so a missing one is reported as an install line rather than as a bare import error three
#: frames into somebody else's module.
_EXTRA_FOR: dict[str, str] = {
    "onnx": "export",
    "onnxruntime": "export",
    "torch": "rllib",
}


def __getattr__(name: str) -> Any:
    """Resolve an export symbol on first use (PEP 562), and blame the right extra if it fails."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        module = importlib.import_module(f"astro_mine.learn.export.{module_name}")
    except ModuleNotFoundError as exc:
        missing = (exc.name or "").split(".")[0]
        extra = _EXTRA_FOR.get(missing)
        if extra is None:
            raise
        raise ModuleNotFoundError(
            f"`astro_mine.learn.export.{name}` needs {missing!r}, which is in the optional "
            f"[{extra}] extra — install it with `pip install 'astro-mine-learn[{extra}]'` "
            f"(or `uv add 'astro-mine-learn[{extra}]'`). The base wheel deliberately omits it: "
            f"a policy *consumer* needs neither ONNX Runtime nor Torch.",
            name=exc.name,
            path=exc.path,
        ) from exc
    value = getattr(module, name)
    # Cache on the module, so only the first access pays the lookup and every later one is a
    # plain global — `__getattr__` is consulted only for names not already in globals().
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Keep introspection and tab-completion complete despite the lazy surface."""
    return sorted(__all__)


__all__ = [
    "DEFAULT_OPSET",
    "HIDDEN_INPUT",
    "HIDDEN_OUTPUT",
    "MESSAGE_INPUT",
    "OBS_INPUT",
    "STATE_TENSORS",
    "EquivalenceError",
    "ExportedPolicy",
    "MessageFn",
    "PublishedPolicy",
    "StatefulOnnxPolicy",
    "assert_onnx_equivalence",
    "assert_onnx_stateful_equivalence",
    "content_id",
    "export_policy_package",
    "export_policy_packages",
    "fixed_input_batch",
    "fixed_obs_batch",
    "input_specs_for",
    "is_recurrent_graph",
    "onnx_action_sample",
    "onnx_policy",
    "output_specs_for",
    "publish",
    "to_onnx_bytes",
]
