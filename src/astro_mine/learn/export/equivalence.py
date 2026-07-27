"""ONNX-Runtime equivalence check that GATES publish (RM-P1-LEARN-05 AC; learn.md §10).

"Every export passes the ONNX-Runtime equivalence check vs the source policy on a fixed
observation batch before publish; CI fails otherwise" (issue #5 AC). This module runs the
exported graph in ONNX Runtime on a deterministic, fixed observation batch and asserts it is
numerically equivalent to the Torch source :meth:`forward_export`, raising
:class:`EquivalenceError` on any mismatch. ``export_policy_package`` (export/package.py) calls
it *before* returning a package, so a graph that diverges from its source policy can never be
published.

**Stateful (recurrent) policies.** A single-shot check is not enough for a policy whose graph
carries a hidden state: a graph can agree with Torch on one step and still drift once the state
is fed back. :func:`assert_onnx_stateful_equivalence` therefore drives a **multi-step rollout**
through both runtimes — feeding each step's ``hidden_out`` back in as the next step's
``hidden_in``, in ONNX Runtime and in Torch independently — and compares every step's outputs.
Divergence in the state recursion (the failure mode a one-shot check misses) is what this
catches. ``export_policy_package`` runs *both* gates for a recurrent actor.

Needs the ``[export]`` extra (``onnxruntime``); imported only from the export path.
"""

from __future__ import annotations

import numpy as np
import onnxruntime
import torch
from numpy.typing import NDArray

from astro_mine.learn.export.onnx import (
    input_specs_for,
    is_recurrent_graph,
    output_names_for,
)
from astro_mine.learn.export.tensors import HIDDEN_INPUT, HIDDEN_OUTPUT, OBS_INPUT

__all__ = [
    "DEFAULT_ATOL",
    "DEFAULT_ROLLOUT_STEPS",
    "DEFAULT_RTOL",
    "EquivalenceError",
    "assert_onnx_equivalence",
    "assert_onnx_stateful_equivalence",
    "fixed_input_batch",
    "fixed_obs_batch",
]

#: Float32 allclose tolerances. ONNX-Runtime and Torch run the same float32 Linear/LayerNorm/
#: tanh ops on CPU, so agreement is tight; the small atol absorbs the LayerNorm/matmul
#: reduction-order jitter between the two runtimes without masking a real graph divergence.
DEFAULT_RTOL = 1.0e-4
DEFAULT_ATOL = 1.0e-5

#: Steps the stateful (recurrent) rollout gate carries the hidden state across. Enough for a
#: state recursion to visibly drift if the two runtimes disagree, cheap enough to gate publish.
DEFAULT_ROLLOUT_STEPS = 8


class EquivalenceError(AssertionError):
    """Raised when an exported ONNX graph diverges from its Torch source policy."""


def fixed_obs_batch(obs_dim: int, *, batch: int = 8, seed: int = 0) -> NDArray[np.float32]:
    """A deterministic observation batch derived from the IoSignature dim (learn.md §10).

    Fixed under ``seed`` + ``obs_dim`` so the equivalence check — and any reproduction of it —
    runs on identical inputs."""
    rng = np.random.default_rng(seed)
    return rng.standard_normal((batch, obs_dim)).astype(np.float32)


def fixed_input_batch(
    net: torch.nn.Module,
    obs_dim: int,
    *,
    batch: int = 8,
    seed: int = 0,
    obs_batch: NDArray[np.float32] | None = None,
) -> dict[str, NDArray[np.float32]]:
    """A deterministic feed for **every** tensor the graph declares, not just ``obs``.

    A comms-learning actor also consumes ``msg`` (the aggregated peer-message context) and a
    recurrent one ``hidden_in``; both are drawn from the same seeded stream so the gate runs on
    identical inputs across runs. Non-``obs`` tensors get their own decorrelated draw (seed +
    position), and a caller-supplied ``obs_batch`` overrides only ``obs``."""
    feed: dict[str, NDArray[np.float32]] = {}
    for position, (name, dim) in enumerate(input_specs_for(net)):
        if name == OBS_INPUT:
            feed[name] = (
                obs_batch
                if obs_batch is not None
                else fixed_obs_batch(obs_dim, batch=batch, seed=seed)
            )
            continue
        rng = np.random.default_rng([seed, position])
        rows = int(feed[OBS_INPUT].shape[0])
        feed[name] = rng.standard_normal((rows, dim)).astype(np.float32)
    return feed


def _torch_outputs(
    net: torch.nn.Module, feed: dict[str, NDArray[np.float32]]
) -> list[NDArray[np.float32]]:
    args = [torch.from_numpy(feed[name]) for name, _dim in input_specs_for(net)]
    net.eval()
    with torch.no_grad():
        outputs = net.forward_export(*args)  # type: ignore[operator]
    return [np.asarray(tensor.detach().numpy(), dtype=np.float32) for tensor in outputs]


def _assert_allclose(
    names: list[str],
    expected: list[NDArray[np.float32]],
    actual: list[NDArray[np.float32]],
    *,
    rtol: float,
    atol: float,
    context: str = "",
) -> None:
    for name, want, got in zip(names, expected, actual, strict=True):
        if not np.allclose(want, got, rtol=rtol, atol=atol):
            max_abs = float(np.max(np.abs(want - np.asarray(got))))
            where = f" {context}" if context else ""
            raise EquivalenceError(
                f"ONNX output {name!r}{where} diverges from the Torch source policy "
                f"(max |diff| = {max_abs:.3e}, rtol={rtol}, atol={atol})"
            )


def assert_onnx_equivalence(
    net: torch.nn.Module,
    onnx_bytes: bytes,
    obs_dim: int,
    *,
    obs_batch: NDArray[np.float32] | None = None,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    seed: int = 0,
) -> None:
    """Assert the exported ``onnx_bytes`` reproduce ``net.forward_export`` on a fixed batch.

    Feeds every tensor the graph declares (``obs``, and ``msg``/``hidden_in`` where the actor
    is comms-learning / recurrent). Raises :class:`EquivalenceError` on the first output whose
    ONNX-Runtime value is not ``allclose`` to Torch's — the gate that runs before publish. For a
    recurrent actor this is the *single-step* half of the gate; see
    :func:`assert_onnx_stateful_equivalence` for the state recursion."""
    feed = fixed_input_batch(net, obs_dim, seed=seed, obs_batch=obs_batch)
    output_names = output_names_for(net)
    expected = _torch_outputs(net, feed)
    session = onnxruntime.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])
    actual = session.run(output_names, feed)
    _assert_allclose(output_names, expected, actual, rtol=rtol, atol=atol)


def assert_onnx_stateful_equivalence(
    net: torch.nn.Module,
    onnx_bytes: bytes,
    obs_dim: int,
    *,
    steps: int = DEFAULT_ROLLOUT_STEPS,
    batch: int = 4,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
    seed: int = 0,
) -> None:
    """Assert a **recurrent** graph tracks its Torch source across a multi-step stateful rollout.

    Drives ``steps`` sequential decisions through both runtimes on the same deterministic
    observation sequence, each carrying its *own* ``hidden_out`` back in as the next step's
    ``hidden_in`` — exactly how a host runs the policy (``export/host.py``). Every step's every
    output (including the hidden state itself) must stay ``allclose``, so a graph whose state
    recursion drifts from Torch's fails the gate even though its first step agreed.

    The rollout starts from the zero hidden state (:meth:`GRUCore.initial_state`), the same
    initial condition the host uses. A no-op on a non-recurrent graph."""
    if not is_recurrent_graph(net):
        return
    hidden_dim = dict(input_specs_for(net))[HIDDEN_INPUT]
    other = [
        (name, dim) for name, dim in input_specs_for(net) if name not in {OBS_INPUT, HIDDEN_INPUT}
    ]
    output_names = output_names_for(net)
    session = onnxruntime.InferenceSession(onnx_bytes, providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(seed)
    # Both runtimes start from the same zero state and advance it independently — the point of
    # the gate is that the two recursions do not drift apart.
    torch_hidden = np.zeros((batch, hidden_dim), dtype=np.float32)
    onnx_hidden = np.zeros((batch, hidden_dim), dtype=np.float32)
    for step in range(steps):
        obs = rng.standard_normal((batch, obs_dim)).astype(np.float32)
        extras = {name: rng.standard_normal((batch, dim)).astype(np.float32) for name, dim in other}
        expected = _torch_outputs(net, {OBS_INPUT: obs, **extras, HIDDEN_INPUT: torch_hidden})
        actual = session.run(output_names, {OBS_INPUT: obs, **extras, HIDDEN_INPUT: onnx_hidden})
        _assert_allclose(
            output_names,
            expected,
            actual,
            rtol=rtol,
            atol=atol,
            context=f"at stateful rollout step {step}",
        )
        state_at = output_names.index(HIDDEN_OUTPUT)
        torch_hidden = expected[state_at]
        onnx_hidden = np.asarray(actual[state_at], dtype=np.float32)
