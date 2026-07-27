"""Model building-block unit tests (RM-P1-LEARN-03) — needs the [rllib] extra (Torch)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.ray

from astro_mine.learn.algos._contract import CentralizedCriticSpec  # noqa: E402
from astro_mine.learn.algos._torch_common import make_generator  # noqa: E402
from astro_mine.learn.models import (  # noqa: E402
    MLP,
    CentralizedCritic,
    DictActorCritic,
    GRUCore,
    MessageModule,
    QMixer,
    VDNMixer,
)


def test_mlp_shapes() -> None:
    net = MLP(6, [8, 8], 3)
    assert net(torch.zeros(4, 6)).shape == (4, 3)


def test_dict_actor_critic_act_evaluate_greedy() -> None:
    net = DictActorCritic(6, {"kind": 3, "mode": 2}, {"goto": 3}, [8, 8])
    gen = make_generator(0)
    out, _ = net.act(torch.zeros(1, 6), gen)
    assert set(out.action) == {"kind", "mode", "goto"}
    assert isinstance(out.action["kind"], int) and out.action["goto"].shape == (3,)
    # evaluate re-scores a batch of the same heads
    actions = {
        "kind": torch.tensor([0, 1]),
        "mode": torch.tensor([0, 1]),
        "goto": torch.zeros(2, 3),
    }
    log_prob, _entropy, value = net.evaluate(torch.zeros(2, 6), actions)
    assert log_prob.shape == (2,) and value.shape == (2,)
    greedy = net.greedy(torch.zeros(1, 6))
    assert set(greedy) == {"kind", "mode", "goto"}


def test_recurrent_actor_critic_and_gru_core() -> None:
    core = GRUCore(4, 5)
    h = core(torch.zeros(1, 4))
    assert h.shape == (1, 5)
    net = DictActorCritic(4, {"kind": 2}, {}, [5], use_rnn=True)
    out, next_h = net.act(torch.zeros(1, 4), make_generator(1))
    assert next_h is not None and out.action["kind"] in (0, 1)


def test_comms_widened_actor_critic_trunk() -> None:
    net = DictActorCritic(6, {"kind": 2}, {}, [8], comms_dim=4)
    feat, _ = net.features(torch.zeros(1, 10))  # obs_dim(6) + comms_dim(4)
    assert feat.shape == (1, 8)


def test_centralized_critic_over_state() -> None:
    spec = CentralizedCriticSpec(global_state_dim=12, per_agent_obs_dims={"a": 6})
    critic = CentralizedCritic(spec, [8])
    assert critic(torch.zeros(3, 12)).shape == (3,)


def test_mixers_combine_agent_qs() -> None:
    agent_qs = torch.tensor([[1.0, 2.0, 3.0]])
    assert torch.allclose(VDNMixer()(agent_qs, torch.zeros(1, 5)), torch.tensor([6.0]))
    mixed = QMixer(3, 5)(agent_qs, torch.zeros(1, 5))
    assert mixed.shape == (1,)


def test_message_module_aggregates_only_reachable_peers() -> None:
    module = MessageModule(feat_dim=4, msg_dim=3)
    features = torch.ones(3, 4)
    # agent 0 reaches nobody (isolated) → zero aggregate; agent 1 reaches agent 2.
    reach = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    agg = module(features, reach)
    assert agg.shape == (3, 3)
    assert torch.allclose(agg[0], torch.zeros(3))  # isolated agent receives nothing
    assert not torch.allclose(agg[1], torch.zeros(3))  # agent 1 gets agent 2's message
