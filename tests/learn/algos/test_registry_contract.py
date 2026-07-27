"""Registry + Core-contract tests — no Torch, runs in every CI job (RM-P1-LEARN-03).

The waist-facing surface: the Learn-internal algorithm registry lists/resolves the three
baselines by capability tag; a produced policy registers in the **Core** plugin registry as
a ``POLICY`` manifest and passes ``check_policy`` against the conformant FakeSwarmWorld.
This is deliberately Torch-free (the ``[rllib]`` extra is not needed) — the narrow-waist
guarantee holds even without the training toolchain installed.
"""

from __future__ import annotations

import pytest

from astro_mine.core.policy import DecisionContext, check_policy
from astro_mine.core.registry import (
    PluginKind,
    PluginRegistry,
    Provenance,
    UnsignedManifest,
)
from astro_mine.learn import make_reference_policy, make_swarm_env
from astro_mine.learn.algos import (
    AlgorithmRegistry,
    IoSignature,
    PolicyAssumptions,
    PolicyExport,
    action_heads,
    agent_io_signature,
    comms_learning_specs,
    default_registry,
    flat_obs_dim,
    flatten_obs,
    manifest_from_export,
    policy_manifest,
)
from tests.learn.fakes import FakeSwarmWorld, build_assets

#: The registered baselines: three comms-blind (IPPO control + the CTDE defaults) and the
#: comms-learning research track (RM-P1-LEARN-03; learn.md §11).
BASELINES = {"ippo", "mappo", "qmix", "comms_ppo"}


def test_registry_lists_the_baselines() -> None:
    assert {s.name for s in default_registry().specs()} == BASELINES


def test_paradigm_declarations() -> None:
    reg = default_registry()
    assert reg.spec("ippo").paradigm == "independent"
    assert not reg.spec("ippo").is_ctde
    assert reg.spec("mappo").paradigm == "ctde"
    assert reg.spec("qmix").is_ctde
    assert reg.spec("comms_ppo").is_ctde


def test_comms_learning_track_is_discoverable_by_its_declaration() -> None:
    # learn.md §11 makes comms-learning a first-class research track: Bench sorts a leaderboard
    # into comms-blind vs comms-learning by this flag, so the flag must be the ONLY thing that
    # decides membership (a third-party plugin appears here the moment it declares it).
    assert [spec.name for spec in comms_learning_specs()] == ["comms_ppo"]
    reg = default_registry()
    assert reg.spec("comms_ppo").comms_learning is True
    for comms_blind in ("ippo", "mappo", "qmix"):
        assert reg.spec(comms_blind).comms_learning is False


def test_resolve_by_tag_or_name_and_membership() -> None:
    reg = AlgorithmRegistry()
    assert "ippo" in reg
    assert "marl.ctde.mappo" in reg
    assert "marl.ctde.comms_ppo" in reg
    assert "nope" not in reg
    assert len(reg) == len(BASELINES)
    assert {s.name for s in reg} == BASELINES
    with pytest.raises(KeyError):
        reg.spec("nonexistent")


def test_register_a_custom_algorithm_double() -> None:
    reg = AlgorithmRegistry(builtins=False)
    assert len(reg) == 0
    ippo = default_registry().get("ippo")
    reg.register(ippo)
    assert reg.get("ippo") is ippo


def test_discover_entry_points_is_empty_without_plugins() -> None:
    # No third-party algorithm is installed under the entry-point group in the test env.
    assert default_registry().discover_entry_points() == []


def test_lazy_algorithm_instances_expose_their_spec() -> None:
    reg = default_registry()
    for tag in sorted(BASELINES):
        assert reg.get(tag).spec.name == tag


def test_discover_entry_points_registers_a_third_party_plugin(monkeypatch) -> None:
    from astro_mine.learn.algos import registry as registry_mod

    ippo = default_registry().get("ippo")

    class _EP:
        name = "thirdparty"

        def load(self):  # returns an Algorithm factory
            return lambda: ippo

    monkeypatch.setattr(registry_mod, "entry_points", lambda *, group: [_EP()])
    reg = AlgorithmRegistry(builtins=False)
    added = reg.discover_entry_points()
    assert added == [ippo.spec.capability_tag]
    assert reg.get(ippo.spec.capability_tag) is ippo


def test_policy_manifest_is_a_core_policy_plugin() -> None:
    manifest = policy_manifest("baseline-ippo", "0.1.0")
    assert manifest.kind == PluginKind.POLICY
    assert manifest.core_interfaces == {
        "env": "0.1.0",
        "messages": "0.1.0",
        "sadf": "0.1.0",
        "policy": "0.1.0",
    }
    assert manifest.inputs == ["Observation"]
    assert manifest.outputs == ["ActionBatch"]
    # Registers (and negotiates) against a Core registry with signatures off (local/dev).
    registry = PluginRegistry(require_signature=False)
    registry.register(manifest)
    assert "baseline-ippo" in registry
    assert registry.by_kind(PluginKind.POLICY)[0].name == "baseline-ippo"


def test_signed_registry_rejects_the_unsigned_policy_manifest() -> None:
    # The manifest carries no signature; a signature-requiring registry must refuse it.
    registry = PluginRegistry()  # require_signature=True by default
    with pytest.raises(UnsignedManifest):
        registry.register(policy_manifest("baseline-ippo", "0.1.0"))


def test_manifest_from_export_carries_honest_metadata() -> None:
    export = PolicyExport(
        algorithm="mappo",
        backend="torch",
        io_signature=IoSignature(agent_ids=("rover",), per_agent={}, global_state_dim=25),
        assumptions=PolicyAssumptions(
            comms_observability={"kind": "comms_model"}, partial_observability=True
        ),
        provenance=Provenance(seed=7, code_version="0.0.0"),
        metrics={"mean_reward": -0.1},
    )
    assert export.policy_kind() == PluginKind.POLICY
    manifest = manifest_from_export(export, name="mappo-policy", version="0.1.0")
    assert manifest.attributes["algorithm"] == "mappo"
    assert manifest.attributes["comms_observability"] == {"kind": "comms_model"}
    assert manifest.attributes["metrics"] == {"mean_reward": -0.1}
    assert manifest.provenance is not None and manifest.provenance.seed == 7
    PluginRegistry(require_signature=False).register(manifest)


def test_reference_policy_passes_check_policy() -> None:
    # A produced policy consumes Core Observations and produces a Sim-consumable ActionBatch —
    # the Core Policy/Planner contract, with NO astro_mine.sim import (waist-pure).
    env = make_swarm_env(FakeSwarmWorld(), build_assets())
    policy = make_reference_policy(env.agent_specs, seed=0)
    world = FakeSwarmWorld()
    result = world.reset(seed=0)
    batch = check_policy(policy, result.observations, DecisionContext())
    assert len(batch.actions) == len(result.observations)


def test_io_flattening_helpers_match_the_spaces() -> None:
    env = make_swarm_env(FakeSwarmWorld(), build_assets())
    spec = env.agent_specs["rover"]
    dim = flat_obs_dim(spec.observation_space)
    sig = agent_io_signature(spec)
    assert sig.obs_dim == dim
    assert "kind" in sig.discrete_heads and "mode" in sig.discrete_heads
    heads = action_heads(spec.action_space)
    assert heads.discrete["kind"] == len(spec.modalities)
    # A sampled observation flattens to exactly the declared width.
    sample = spec.observation_space.sample()
    assert flatten_obs(sample, spec.observation_space).shape == (dim,)


def test_reference_policy_is_seed_reproducible() -> None:
    env = make_swarm_env(FakeSwarmWorld(), build_assets())
    world = FakeSwarmWorld()
    obs = world.reset(seed=0).observations
    a = make_reference_policy(env.agent_specs, seed=3).decide(obs, DecisionContext())
    b = make_reference_policy(env.agent_specs, seed=3).decide(obs, DecisionContext())
    assert [x.model_dump() for x in a.actions] == [x.model_dump() for x in b.actions]
