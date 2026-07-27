"""Consumer-driven contract: an exported PolicyPackage is a Core Policy (RM-P1-LEARN-05; M1.2).

The flywheel AC: an exported PolicyPackage is consumed as a controller by Mind, wrapped by
Guard, and scored by Bench. This proves it through the Core seam — wrap the exported graph in
Core :class:`~astro_mine.core.policy.OnnxPolicy` and assert Core ``check_policy`` passes
(returns a Sim-consumable ActionBatch) — plus the content-addressed publish path Bench resolves
by hash. Needs the [export] extra + Torch; not ray/gpu.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from astro_mine.core.compat import IncompatibleCoreInterface
from astro_mine.core.policy import DecisionContext, check_policy
from astro_mine.core.policy.loader import load_policy_package
from astro_mine.learn import TrainConfig, default_registry, make_swarm_env
from astro_mine.learn.export import export_policy_package, onnx_policy, publish
from tests.learn.fakes import FakeSwarmWorld, build_assets

_CFG = TrainConfig(seed=1, iterations=1, rollout_steps=8, hidden_sizes=(16, 16))


def _exported(tag: str, agent: str):
    env = make_swarm_env(FakeSwarmWorld(), build_assets())
    trainer = default_registry().get(tag).make_trainer(env, _CFG)
    trainer.train_iteration()
    return export_policy_package(trainer.export(), agent, version="0.1.0"), env.agent_specs[agent]


@pytest.mark.parametrize(("tag", "agent"), [("ippo", "rover"), ("qmix", "digger")])
def test_exported_policy_satisfies_the_core_policy_contract(tag: str, agent: str) -> None:
    exported, spec = _exported(tag, agent)
    policy = onnx_policy(exported.document.policy_package, exported.onnx_bytes, spec)
    core_obs = FakeSwarmWorld().reset(seed=1).observations
    # Core's consumer-driven contract test: the ONNX-hosted policy is a valid controller and
    # emits a Sim-consumable ActionBatch (the same check Mind/Guard/Bench run).
    batch = check_policy(policy, {agent: core_obs[agent]}, DecisionContext())
    assert len(batch.actions) == 1
    assert batch.actions[0].agent_id == agent


def test_incompatible_core_interface_fails_loud_at_load() -> None:
    exported, spec = _exported("ippo", "rover")
    with pytest.raises(IncompatibleCoreInterface):
        # A future-major Core requirement the current package cannot satisfy.
        onnx_policy(
            exported.document.policy_package,
            exported.onnx_bytes,
            spec,
            provided={"policy": "1.0.0"},
        )


def test_publish_is_content_addressed_and_reloadable(tmp_path) -> None:
    exported, _spec = _exported("ippo", "rover")
    seen: list[str] = []
    published = publish(exported, tmp_path, publisher=lambda p: seen.append(p.digest))
    # Content-addressed layout: <store>/<hex>/{model.onnx, policy_package.json}.
    assert published.onnx_path.exists() and published.sidecar_path.exists()
    assert published.digest.split(":", 1)[-1] in str(published.onnx_path)
    assert seen == [exported.digest]  # the optional Hub handoff fired after the local write
    assert published.onnx_path.read_bytes() == exported.onnx_bytes
    # Bench resolves the sidecar by hash and it re-validates against Core's schema.
    reloaded = load_policy_package(published.sidecar_path.read_text())
    assert reloaded.policy_package.onnx_model.digest == exported.digest
    assert reloaded.policy_package.onnx_model.uri == published.onnx_path.as_uri()
    # The stored sidecar is self-consistent JSON.
    assert json.loads(published.sidecar_path.read_text())["policy_package_version"] == "0.1"
