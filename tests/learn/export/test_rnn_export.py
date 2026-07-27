"""ONNX export for recurrent (GRU) policies (RM-P1-LEARN-05; learn.md §3, §5, §10) — [export].

``TrainConfig.use_rnn=True`` is *the* natural architecture for the charter §8 problem — a
recurrent core carries belief across the ticks an agent cannot see or hear — and until now such
a policy could not be exported at all (``_rebuild_export_net`` raised ``NotImplementedError``).

The export represents the GRU state as an **explicit** ``hidden_in`` → ``hidden_out`` tensor
pair, which is what resolves the "opset/dynamic-axis care" the old comment deferred: the graph
becomes a plain *step function over an explicit state* — a single dynamic (batch) axis, no ONNX
``Loop``/``Scan``, no sequence-length axis — and the **host** carries the state across calls.

The load-bearing test is the **stateful** one: a graph can match Torch on a single step and still
drift once its own output is fed back in. So the gate drives a multi-step rollout through both
runtimes, carrying each one's own hidden state, and compares every step.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import onnxruntime
import torch

from astro_mine.core.policy import (
    DecisionContext,
    check_policy,
    validate_policy_package,
)
from astro_mine.learn import TrainConfig, default_registry, make_swarm_env
from astro_mine.learn.export import (
    HIDDEN_INPUT,
    HIDDEN_OUTPUT,
    EquivalenceError,
    StatefulOnnxPolicy,
    assert_onnx_equivalence,
    assert_onnx_stateful_equivalence,
    export_policy_package,
    export_policy_packages,
    is_recurrent_graph,
    onnx_policy,
    to_onnx_bytes,
)
from astro_mine.learn.models.mlp import DictActorCritic
from tests.learn.fakes import FakeSwarmWorld, build_assets

_OBS_DIM = 12
_RNN_CFG = TrainConfig(seed=2, iterations=1, rollout_steps=8, hidden_sizes=(16, 16), use_rnn=True)


def _rnn_actor() -> DictActorCritic:
    torch.manual_seed(0)
    net = DictActorCritic(
        _OBS_DIM, {"kind": 3, "mode": 2}, {"goto": 3}, (16, 16), use_rnn=True
    ).eval()
    return net


def _train_rnn(tag: str = "ippo"):
    env = make_swarm_env(FakeSwarmWorld(), build_assets())
    trainer = default_registry().get(tag).make_trainer(env, _RNN_CFG)
    trainer.train_iteration()
    return trainer, env


# --- the graph: explicit state, one dynamic axis ------------------------------------


def test_recurrent_graph_declares_hidden_state_tensors() -> None:
    net = _rnn_actor()
    assert is_recurrent_graph(net)
    session = onnxruntime.InferenceSession(
        to_onnx_bytes(net, _OBS_DIM), providers=["CPUExecutionProvider"]
    )
    inputs = {i.name: i.shape for i in session.get_inputs()}
    outputs = {o.name: o.shape for o in session.get_outputs()}
    assert inputs[HIDDEN_INPUT] == ["batch", net.feat_dim]
    assert outputs[HIDDEN_OUTPUT] == ["batch", net.feat_dim]
    # The opset/dynamic-axis point: batch is the ONLY dynamic axis, on every tensor including
    # the state — no sequence-length axis, so the graph is a step function, not a Loop.
    for shape in (*inputs.values(), *outputs.values()):
        assert shape[0] == "batch"
        assert all(isinstance(dim, int) for dim in shape[1:])


def test_recurrent_graph_is_byte_stable_for_fixed_weights() -> None:
    net = _rnn_actor()
    assert to_onnx_bytes(net, _OBS_DIM) == to_onnx_bytes(net, _OBS_DIM)


# --- the gate: multi-step stateful equivalence ---------------------------------------


def test_stateful_rollout_equivalence_over_multiple_steps() -> None:
    net = _rnn_actor()
    onnx_bytes = to_onnx_bytes(net, _OBS_DIM)
    assert_onnx_equivalence(net, onnx_bytes, _OBS_DIM)  # single-shot half
    # The real gate: hidden state carried across steps in BOTH runtimes, independently.
    assert_onnx_stateful_equivalence(net, onnx_bytes, _OBS_DIM, steps=8, batch=4)


def test_the_stateful_gate_bites_when_the_state_recursion_diverges() -> None:
    # Perturb ONLY the GRU cell after export. A graph that ignored the state would still agree
    # on outputs; the stateful gate must catch the drift the recursion produces.
    net = _rnn_actor()
    onnx_bytes = to_onnx_bytes(net, _OBS_DIM)
    with torch.no_grad():
        for param in net.rnn.parameters():
            param.add_(0.5)
    with pytest.raises(EquivalenceError, match="stateful rollout step"):
        assert_onnx_stateful_equivalence(net, onnx_bytes, _OBS_DIM, steps=6, batch=2)


def test_the_hidden_state_actually_propagates() -> None:
    # A graph that silently dropped the state would trivially "pass" a stateful check that never
    # varied the state. Prove the state changes the answer: same obs, different hidden ⇒
    # different logits, and hidden_out is a real function of hidden_in.
    net = _rnn_actor()
    session = onnxruntime.InferenceSession(
        to_onnx_bytes(net, _OBS_DIM), providers=["CPUExecutionProvider"]
    )
    obs = np.zeros((1, _OBS_DIM), dtype=np.float32)
    zero_state = np.zeros((1, net.feat_dim), dtype=np.float32)
    other_state = np.full((1, net.feat_dim), 0.7, dtype=np.float32)
    names = ["kind", HIDDEN_OUTPUT]
    kind_a, next_a = session.run(names, {"obs": obs, HIDDEN_INPUT: zero_state})
    kind_b, next_b = session.run(names, {"obs": obs, HIDDEN_INPUT: other_state})
    assert not np.allclose(kind_a, kind_b)
    assert not np.allclose(next_a, next_b)


# --- the AC: a trained RNN policy exports and reloads --------------------------------


def test_trained_rnn_policy_exports_and_reloads() -> None:
    # The regression AC: use_rnn=True used to raise NotImplementedError here.
    trainer, _env = _train_rnn()
    exported = export_policy_package(trainer.export(), "rover", version="0.1.0")
    validate_policy_package(exported.document)  # Core's shipped JSON Schema
    package = exported.document.policy_package
    package.assert_core_compatible()

    hidden_dim = _RNN_CFG.hidden_sizes[-1]
    inputs = {t.name: t.shape for t in package.io_signature.inputs}
    outputs = {t.name: t.shape for t in package.io_signature.outputs}
    assert inputs[HIDDEN_INPUT] == [-1, hidden_dim]
    assert outputs[HIDDEN_OUTPUT] == [-1, hidden_dim]
    assert package.io_signature.observation_space["recurrent"] is True
    assert package.io_signature.observation_space["hidden_dim"] == hidden_dim

    # The hidden state is policy STATE, not an action: it must not be declared as an action
    # component or carry action bounds Guard would try to enforce.
    kinds = {o["name"]: o["kind"] for o in package.io_signature.action_space["outputs"]}
    assert kinds[HIDDEN_OUTPUT] == "state"
    assert kinds["goto"] == "box"
    assert HIDDEN_OUTPUT not in package.assumptions.action_bounds

    # Reload: the serialized bytes open as a session with the same declared IO.
    session = onnxruntime.InferenceSession(exported.onnx_bytes, providers=["CPUExecutionProvider"])
    assert {i.name for i in session.get_inputs()} == set(inputs)
    assert {o.name for o in session.get_outputs()} == set(outputs)


def test_every_heterogeneous_rnn_agent_exports() -> None:
    trainer, _env = _train_rnn("mappo")
    packages = export_policy_packages(trainer.export(), version="0.1.0")
    assert set(packages) == {"rover", "digger", "relay"}
    assert len({p.digest for p in packages.values()}) == 3


def test_same_weights_yield_the_same_recurrent_digest() -> None:
    trainer, _env = _train_rnn()
    export = trainer.export()
    first = export_policy_package(export, "rover", version="0.1.0")
    second = export_policy_package(export, "rover", version="0.1.0")
    assert first.digest == second.digest  # the graph hash IS the artifact identity


# --- the host carries the state ------------------------------------------------------


def test_hosted_rnn_policy_carries_hidden_state_across_decisions() -> None:
    trainer, env = _train_rnn()
    exported = export_policy_package(trainer.export(), "rover", version="0.1.0")
    policy = onnx_policy(
        exported.document.policy_package, exported.onnx_bytes, env.agent_specs["rover"]
    )
    # A recurrent package yields a policy that OWNS the state — otherwise the exported policy
    # would run with amnesia and behave nothing like the one that was trained.
    assert isinstance(policy, StatefulOnnxPolicy)
    assert policy.hidden_state == {}

    world = FakeSwarmWorld()
    observations = {"rover": world.reset(seed=0).observations["rover"]}
    check_policy(policy, observations, DecisionContext())
    after_one = policy.hidden_state["rover"].copy()
    assert not np.allclose(after_one, 0.0)  # the zero belief has moved

    check_policy(policy, observations, DecisionContext())
    after_two = policy.hidden_state["rover"]
    assert not np.allclose(after_one, after_two)  # ... and keeps moving: the state is carried

    # An episode boundary drops the belief — one episode must not leak into the next.
    policy.reset_state()
    assert policy.hidden_state == {}


def test_recurrent_package_is_scored_by_the_eval_harness_with_its_state() -> None:
    # The honest-eval harness must score a recurrent package the way it will actually run
    # (carrying state), not as a memoryless policy.
    from astro_mine.learn.algos.policy import flat_obs_dim
    from astro_mine.learn.eval import OnnxGraph, onnx_policy_under_test

    trainer, env = _train_rnn()
    packages = export_policy_packages(trainer.export(), version="0.1.0")
    graphs = {
        agent: OnnxGraph(package=p.document.policy_package, onnx_bytes=p.onnx_bytes)
        for agent, p in packages.items()
    }
    specs = env.agent_specs
    policy = onnx_policy_under_test(graphs, specs)
    flat = {
        agent: np.zeros(flat_obs_dim(spec.observation_space), dtype=np.float32)
        for agent, spec in specs.items()
    }
    first = policy.act(flat)
    second = policy.act(flat)
    assert set(first) == {"rover", "digger", "relay"}
    # Identical observations, different answers ⇒ the scorer really is carrying the state.
    assert any(
        not np.array_equal(np.asarray(first[a]["goto"]), np.asarray(second[a]["goto"]))
        for a in first
    )
