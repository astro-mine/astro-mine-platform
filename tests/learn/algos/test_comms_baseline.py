"""The comms-learning baseline ``comms_ppo`` (RM-P1-LEARN-03; learn.md §11) — needs [rllib].

learn.md §11 makes comms-learning a **first-class research track**, "because comms-limited
cooperation *is* the charter §8 problem". The claim this file has to substantiate is not just
"it trains" (``test_baselines_smoke.py`` parametrizes it with the other baselines for that) —
it is that **the learned messages ride the CommsModel's real channel**:

- the message aggregate is built from the peers the channel actually **delivered** to
  (post gate → budget → drop → delay), read off the executor's recorded reachability;
- an agent nobody reaches receives the **zero** message (it cannot cheat by peeking);
- an agent whose messages are never delivered contributes **no gradient** to the encoder — the
  channel's verdict is what decides whether a message was worth learning;
- degrading the channel changes what the policy learns (the comms constraint actually binds).

Plus the issue's remaining ACs: it exports to a PolicyPackage that round-trips the ONNX-Runtime
equivalence check, and it is seed-reproducible.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

pytestmark = pytest.mark.ray

import numpy as np  # noqa: E402
import torch  # noqa: E402

from astro_mine.learn import TrainConfig, default_registry, make_swarm_env  # noqa: E402
from astro_mine.learn.algos._torch_common import MESSAGE_DIM  # noqa: E402
from astro_mine.learn.envs import (  # noqa: E402
    CommsModel,
    CommsModelConfig,
    DelayConfig,
    DropConfig,
)
from astro_mine.learn.models.comms import CommsEncoder, reach_matrix  # noqa: E402
from astro_mine.learn.train.executor import CommsAwareStep  # noqa: E402
from tests.learn.fakes import AGENTS, FakeSwarmWorld, build_assets  # noqa: E402

_CFG = TrainConfig(seed=5, iterations=1, rollout_steps=8, hidden_sizes=(16, 16))


def _env(comms: CommsModelConfig | None = None):
    model = CommsModel(comms) if comms is not None else None
    return make_swarm_env(FakeSwarmWorld(), build_assets(), comms_model=model)


def _trainer(comms: CommsModelConfig | None = None, config: TrainConfig = _CFG):
    return default_registry().get("comms_ppo").make_trainer(_env(comms), config)


# --- the channel is the channel ----------------------------------------------------


def test_the_step_is_comms_aware_and_the_executor_hands_it_the_verdict() -> None:
    # The seam: the trainer's rollout step declares CommsAwareStep, so LocalExecutor hands it
    # the per-tick reachability *before* the decision — the same verdict it then records on the
    # RolloutStep. A comms-blind baseline does not implement it and is never called.
    assert isinstance(_trainer().rollout_step, CommsAwareStep)
    assert not isinstance(default_registry().get("ippo").make_trainer(_env(), _CFG), CommsAwareStep)


def test_messages_aggregate_only_over_peers_the_channel_delivered() -> None:
    # reach_matrix is the bridge from the executor's recorded verdict to the message mask:
    # entry (i, j) is 1 iff j's message REACHED i. Nothing else may set it.
    reach = {"rover": ("relay",), "relay": ("rover",), "digger": ()}
    mask = reach_matrix(AGENTS, reach)
    rover, digger, relay = (AGENTS.index(a) for a in ("rover", "digger", "relay"))
    assert mask[rover, relay] == 1.0  # relay's message got through to rover
    assert mask[relay, rover] == 1.0
    assert mask[rover, digger] == 0.0  # digger's did not
    assert mask[digger].sum() == 0.0  # digger heard from nobody
    assert np.trace(mask) == 0.0  # no agent messages itself


def test_an_isolated_agent_receives_the_zero_message() -> None:
    # "A message that the channel dropped simply does not arrive; the aggregate over an isolated
    # agent is zero" — the no-cheating invariant. An agent cut off must not see peer information.
    encoder = CommsEncoder({a: 6 for a in AGENTS}, feat_dim=8, msg_dim=MESSAGE_DIM)
    obs = {a: torch.ones(1, 6) for a in AGENTS}
    isolated = torch.from_numpy(reach_matrix(AGENTS, {a: () for a in AGENTS})[np.newaxis, ...])
    for message in encoder(obs, isolated).values():
        assert torch.allclose(message, torch.zeros(1, MESSAGE_DIM))

    # ... and with a live link, the receiver's message is non-zero (the channel carries signal).
    connected = torch.from_numpy(
        reach_matrix(AGENTS, {"rover": ("relay",), "digger": (), "relay": ()})[np.newaxis, ...]
    )
    delivered = encoder(obs, connected)
    assert not torch.allclose(delivered["rover"], torch.zeros(1, MESSAGE_DIM))
    assert torch.allclose(delivered["digger"], torch.zeros(1, MESSAGE_DIM))


def test_only_delivered_messages_earn_gradients() -> None:
    # The sharpest statement of "respects the CommsModel semantics": the FakeSwarmWorld gives
    # `digger` no comms mask at all, so its messages are never delivered to anyone. The team
    # objective therefore CANNOT depend on what digger says — and its message-encoder projection
    # must receive exactly zero gradient, while a delivering agent's does not.
    trainer = _trainer(CommsModelConfig(drop=DropConfig(probability=0.3)))
    trainer.train_iteration()
    encoder = trainer.rollout_step._comms
    assert encoder is not None
    grads = {agent: encoder.project[i].weight.grad for i, agent in enumerate(encoder.agents)}
    assert grads["digger"] is not None
    assert float(grads["digger"].abs().sum()) == 0.0  # never delivered ⇒ never learned from
    assert float(grads["rover"].abs().sum()) > 0.0  # rover's message reaches relay ⇒ it learns
    # The shared MessageModule encoder is trained by whatever DID get through.
    assert float(encoder.messages.encode.weight.grad.abs().sum()) > 0.0


def test_the_channel_binds_the_learned_behaviour() -> None:
    # Same seed, same everything — except the channel. If the comms regime did not actually
    # reach the policy, these curves would be identical. They must not be.
    clean = _trainer(CommsModelConfig()).train_iteration()
    denied = _trainer(
        CommsModelConfig(
            drop=DropConfig(probability=0.9),
            delay=DelayConfig(kind="geometric", mean_ticks=2.0, max_ticks=8),
        )
    ).train_iteration()
    assert clean["message_delivery_rate"] > denied["message_delivery_rate"]
    assert clean["policy_loss"] != denied["policy_loss"]


def test_delivery_rate_is_reported_for_the_learned_channel() -> None:
    # A comms-learning score is only interpretable next to the delivery rate it was learned
    # under (learn.md §10), so the baseline reports it; comms-blind baselines do not.
    metrics = _trainer(CommsModelConfig(drop=DropConfig(probability=0.5))).train_iteration()
    assert 0.0 <= metrics["message_delivery_rate"] <= 1.0
    ippo = default_registry().get("ippo").make_trainer(_env(), _CFG).train_iteration()
    assert "message_delivery_rate" not in ippo


# --- it is a baseline: reproducible, exportable, comparable -------------------------


def test_seeded_run_is_reproducible() -> None:
    comms = CommsModelConfig(drop=DropConfig(probability=0.4))
    first = [_trainer(comms).train_iteration() for _ in range(1)]
    second = [_trainer(comms).train_iteration() for _ in range(1)]
    assert first == second  # byte-identical under the fixed seed (the CX-REPRO gate)


def test_comparison_run_against_the_comms_blind_baselines() -> None:
    # The point of registering it as a baseline: it is scored on the SAME env, the SAME seed and
    # the SAME comms regime as IPPO/MAPPO/QMIX, so the leaderboard comparison is honest
    # (learn.md §2.3 "results are comparable across algorithms").
    comms = CommsModelConfig(drop=DropConfig(probability=0.3))
    reg = default_registry()
    curves = {}
    for tag in ("ippo", "mappo", "qmix", "comms_ppo"):
        trainer = reg.get(tag).make_trainer(_env(comms), _CFG)
        trainer.train_iteration()
        curves[tag] = trainer.learning_curve()
    assert set(curves) == {"ippo", "mappo", "qmix", "comms_ppo"}
    assert all(len(curve) == 1 for curve in curves.values())
    # Every baseline saw the same declared comms envelope — that is what makes them comparable.
    exports = {tag: reg.get(tag).make_trainer(_env(comms), _CFG).export() for tag in curves}
    regimes = {
        tag: export.assumptions.comms_observability["config"] for tag, export in exports.items()
    }
    assert len({str(sorted(r.items())) for r in regimes.values()}) == 1


def test_export_declares_the_message_input_and_passes_the_equivalence_gate() -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from astro_mine.learn.export import export_policy_package

    trainer = _trainer(CommsModelConfig(drop=DropConfig(probability=0.2)))
    trainer.train_iteration()
    export = trainer.export()
    assert export.net_arch["rover"]["comms_dim"] == MESSAGE_DIM
    # The shared message encoder stays internal to Learn (like the CTDE critic).
    assert "__comms__" in export.weights

    # export_policy_package runs the ONNX-Runtime equivalence gate BEFORE returning, so simply
    # getting a package back is the round-trip passing.
    exported = export_policy_package(export, "rover", version="0.1.0")
    package = exported.document.policy_package
    # The aggregated peer-message context is an EXPLICIT graph input — honest metadata for the
    # host, not a message smuggled inside a widened `obs` (learn.md §9).
    inputs = {t.name: t.shape for t in package.io_signature.inputs}
    assert inputs["msg"] == [-1, MESSAGE_DIM]
    assert package.io_signature.observation_space["comms_dim"] == MESSAGE_DIM
    assert "msg" in package.io_signature.observation_space["stateful_inputs"]
    # ... and it is not mistaken for an action.
    assert "msg" not in package.assumptions.action_bounds
    assert exported.digest.startswith("sha256:")


def test_exported_comms_policy_is_hosted_as_a_core_policy() -> None:
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from astro_mine.core.policy import DecisionContext, check_policy
    from astro_mine.learn.export import export_policy_package, onnx_policy

    env = _env(CommsModelConfig(drop=DropConfig(probability=0.2)))
    trainer = default_registry().get("comms_ppo").make_trainer(env, _CFG)
    trainer.train_iteration()
    exported = export_policy_package(trainer.export(), "rover", version="0.1.0")

    # A decentralized host running one agent's graph has no peers to aggregate, so it feeds the
    # ZERO message — the isolated-agent case, exactly the MessageModule semantics.
    policy = onnx_policy(
        exported.document.policy_package, exported.onnx_bytes, env.agent_specs["rover"]
    )
    world = FakeSwarmWorld()
    observations = {"rover": world.reset(seed=0).observations["rover"]}
    batch = check_policy(policy, observations, DecisionContext())
    assert len(batch.actions) == 1
