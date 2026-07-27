"""ONNX-Runtime equivalence gate for the exported policy graph (RM-P1-LEARN-05) — [export].

The AC: "every export passes the ONNX-Runtime equivalence check vs the source policy on a
fixed observation batch before publish; CI fails otherwise" (issue #5). These prove the
traced graph reproduces the Torch source ``forward_export`` — over a Hypothesis sweep of
observation batches, not just the one fixed batch — and that the gate actually bites when the
graph and the source diverge. The graph is pinned (opset + weights ⇒ byte-stable bytes), the
foundation of the content-addressed identity checked in ``test_policy_package.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from astro_mine.learn.export.equivalence import (
    EquivalenceError,
    assert_onnx_equivalence,
    fixed_obs_batch,
)
from astro_mine.learn.export.onnx import to_onnx_bytes
from astro_mine.learn.models.mlp import AgentQNet, DictActorCritic

_OBS_DIM = 12


def _actor() -> DictActorCritic:
    torch.manual_seed(0)
    return DictActorCritic(_OBS_DIM, {"kind": 3, "mode": 2}, {"goto": 3}, (16, 16))


@given(
    obs=hnp.arrays(
        dtype=np.float32,
        shape=(4, _OBS_DIM),
        elements=st.floats(-5.0, 5.0, width=32, allow_nan=False, allow_infinity=False),
    )
)
@settings(max_examples=25, deadline=None)
def test_onnx_matches_torch_over_a_hypothesis_obs_sweep(obs: np.ndarray) -> None:
    net = _actor()
    onnx_bytes = to_onnx_bytes(net, _OBS_DIM)
    # Raises EquivalenceError on any divergence — so a passing sweep is the gate.
    assert_onnx_equivalence(net, onnx_bytes, _OBS_DIM, obs_batch=obs)


def test_qnet_graph_is_equivalent() -> None:
    torch.manual_seed(1)
    qnet = AgentQNet(10, 4, (16, 16))
    assert_onnx_equivalence(qnet, to_onnx_bytes(qnet, 10), 10)


def test_graph_is_byte_stable_for_fixed_weights() -> None:
    net = _actor()
    assert to_onnx_bytes(net, _OBS_DIM) == to_onnx_bytes(net, _OBS_DIM)


def test_equivalence_gate_bites_on_divergence() -> None:
    net = _actor()
    onnx_bytes = to_onnx_bytes(net, _OBS_DIM)
    # Perturb the source net after export: the graph no longer matches, and the gate must raise.
    with torch.no_grad():
        for param in net.parameters():
            param.add_(1.0)
    with pytest.raises(EquivalenceError, match="diverges"):
        assert_onnx_equivalence(net, onnx_bytes, _OBS_DIM, obs_batch=fixed_obs_batch(_OBS_DIM))
