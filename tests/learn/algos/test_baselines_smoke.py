"""Tiny CPU smoke-train for IPPO/MAPPO/QMIX/comms_ppo (RM-P1-LEARN-03) — needs [rllib].

Trains each baseline for 1-2 iterations on ``SwarmEnv(FakeSwarmWorld())`` (the loop that
scales out unchanged through the RM-P1-LEARN-04 executor seam), then asserts: the loop runs
and reports metrics, the produced policy passes ``check_policy``, ``export()`` yields a valid
typed :class:`PolicyExport`, and that export registers as a Core ``POLICY`` plugin. CTDE
baselines expose their centralized-critic input spec. This is the CI smoke reference; the
real anchor lunar-polar-prospecting reference score needs Sim and is the out-of-CI ``slow``
artifact (see ``tests/bench``).

``comms_ppo`` — the comms-learning research track — is parametrized in here with the others on
purpose: a comms-learning baseline is a *baseline*, so it must clear every bar the comms-blind
ones do (trainable, ``check_policy``, exportable, seed-reproducible). Its channel-specific
behaviour is tested in ``test_comms_baseline.py``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

pytestmark = pytest.mark.ray

from astro_mine.core.policy import DecisionContext, check_policy  # noqa: E402
from astro_mine.core.registry import PluginKind, PluginRegistry  # noqa: E402
from astro_mine.learn.algos import (  # noqa: E402
    LearnedPolicy,
    PolicyExport,
    default_registry,
    manifest_from_export,
)
from tests.learn.fakes import FakeSwarmWorld  # noqa: E402

BASELINES = ["ippo", "mappo", "qmix", "comms_ppo"]


@pytest.mark.parametrize("tag", BASELINES)
def test_smoke_train_loop_runs_and_reports(tag, env_factory, tiny_config) -> None:
    trainer = default_registry().get(tag).make_trainer(env_factory(), tiny_config)
    m1 = trainer.train_iteration()
    trainer.train_iteration()  # a second iteration to exercise the loop across ticks
    assert "mean_reward" in m1 and "env_steps" in m1
    assert m1["env_steps"] > 0
    assert len(trainer.learning_curve()) == 2
    assert trainer.spec.name == tag


@pytest.mark.parametrize("tag", BASELINES)
def test_produced_policy_passes_check_policy(tag, env_factory, tiny_config) -> None:
    trainer = default_registry().get(tag).make_trainer(env_factory(), tiny_config)
    trainer.train_iteration()
    policy = trainer.policy()
    assert isinstance(policy, LearnedPolicy)
    world = FakeSwarmWorld()
    batch = check_policy(policy, world.reset(seed=1).observations, DecisionContext())
    assert len(batch.actions) == len(world.possible_agents)


@pytest.mark.parametrize("tag", BASELINES)
def test_export_is_a_valid_typed_intermediate(tag, env_factory, tiny_config) -> None:
    trainer = default_registry().get(tag).make_trainer(env_factory(), tiny_config)
    trainer.train_iteration()
    export = trainer.export()
    assert isinstance(export, PolicyExport)
    assert export.algorithm == tag and export.backend == "torch"
    assert export.io_signature.agent_ids == ("rover", "digger", "relay")
    assert export.provenance.seed == tiny_config.seed
    assert export.assumptions.partial_observability is True
    assert export.policy_kind() == PluginKind.POLICY
    # The produced policy registers as a Core POLICY plugin.
    manifest = manifest_from_export(export, name=f"{tag}-policy", version="0.1.0")
    PluginRegistry(require_signature=False).register(manifest)


def test_ctde_baselines_declare_a_centralized_critic(env_factory, tiny_config) -> None:
    reg = default_registry()
    ippo = reg.get("ippo").make_trainer(env_factory(), tiny_config)
    assert ippo.centralized_critic is None  # the simple control shares nothing
    for tag in ("mappo", "qmix", "comms_ppo"):
        trainer = reg.get(tag).make_trainer(env_factory(), tiny_config)
        critic = trainer.centralized_critic
        assert critic is not None
        assert critic.source == "SwarmEnv.state()"
        assert critic.global_state_dim > 0
        assert set(critic.per_agent_obs_dims) == {"rover", "digger", "relay"}


def test_qmix_supports_both_mixers(env_factory) -> None:
    from astro_mine.learn.algos import TrainConfig

    reg = default_registry()
    for mixer in ("vdn", "qmix"):
        cfg = TrainConfig(iterations=1, rollout_steps=8, hidden_sizes=(16, 16), mixer=mixer)
        trainer = reg.get("qmix").make_trainer(env_factory(), cfg)
        metrics = trainer.train_iteration()
        assert "td_loss" in metrics


def test_comms_model_provenance_flows_into_the_export(comms_env_factory, tiny_config) -> None:
    trainer = default_registry().get("mappo").make_trainer(comms_env_factory(), tiny_config)
    trainer.train_iteration()
    export = trainer.export()
    assert export.assumptions.comms_observability is not None
    assert export.assumptions.comms_observability["kind"] == "comms_model"


def test_recurrent_policy_trains(env_factory) -> None:
    from astro_mine.learn.algos import TrainConfig

    cfg = TrainConfig(iterations=1, rollout_steps=8, hidden_sizes=(16, 16), use_rnn=True)
    trainer = default_registry().get("ippo").make_trainer(env_factory(), cfg)
    assert "mean_reward" in trainer.train_iteration()


def test_surrogate_fidelity_run_is_flagged_for_high_fidelity_validation(env_factory) -> None:
    # Multi-fidelity (RM-P1-LEARN-04 AC): a policy trained mostly on surrogate fidelity is
    # flagged for a high-fidelity validation pass, and the caveat rides into the honest
    # metadata Guard reads off the produced-policy manifest (learn.md §9).
    from astro_mine.learn.algos import TrainConfig

    sim = TrainConfig(iterations=1, rollout_steps=8, hidden_sizes=(16, 16), fidelity="sim_high")
    surrogate = sim.model_copy(update={"fidelity": "surrogate"})

    sim_trainer = default_registry().get("ippo").make_trainer(env_factory(), sim)
    sim_trainer.train_iteration()
    assert sim_trainer.export().assumptions.surrogate_fidelity_caveats == ()

    sur_trainer = default_registry().get("ippo").make_trainer(env_factory(), surrogate)
    sur_trainer.train_iteration()
    export = sur_trainer.export()
    caveats = export.assumptions.surrogate_fidelity_caveats
    assert caveats and "high-fidelity" in caveats[0]
    manifest = manifest_from_export(export, name="ippo-surrogate", version="0.1.0")
    assert manifest.attributes["surrogate_fidelity_caveats"] == list(caveats)
