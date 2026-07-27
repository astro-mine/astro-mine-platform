"""PolicyPackage validity, content-addressed identity, honest provenance (RM-P1-LEARN-05).

The AC (issue #5): every export passes the equivalence gate before it becomes a package; the
sidecar carries honest provenance for Guard (comms/observability, action bounds,
surrogate-fidelity caveats); and the artifact is content-addressed so Bench re-derives it by
hash. Needs the [export] extra (onnx/onnxruntime) + Torch; not ray/gpu.
"""

from __future__ import annotations

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from astro_mine.core.policy import validate_policy_package
from astro_mine.core.policy.model import PolicyPackageDocument
from astro_mine.learn import TrainConfig, default_registry, make_swarm_env
from astro_mine.learn.envs import CommsModel, CommsModelConfig, DropConfig
from astro_mine.learn.export import (
    content_id,
    export_policy_package,
    export_policy_packages,
)
from tests.learn.fakes import FakeSwarmWorld, build_assets


def _train(tag: str, config: TrainConfig, *, comms: bool = False):
    comms_model = CommsModel(CommsModelConfig(drop=DropConfig(probability=0.5))) if comms else None
    env = make_swarm_env(FakeSwarmWorld(), build_assets(), comms_model=comms_model)
    trainer = default_registry().get(tag).make_trainer(env, config)
    trainer.train_iteration()
    return trainer


_CFG = TrainConfig(seed=1, iterations=1, rollout_steps=8, hidden_sizes=(16, 16))


def test_exported_document_is_valid_and_core_compatible() -> None:
    exported = export_policy_package(_train("ippo", _CFG).export(), "rover", version="0.1.0")
    validate_policy_package(exported.document)  # structural, against Core's shipped JSON Schema
    package = exported.document.policy_package
    package.assert_core_compatible()  # negotiates core_interfaces against this Core (0.1.0)
    assert package.name == "ippo.rover"
    assert package.onnx_model.digest == exported.digest
    assert package.onnx_model.opset == 17
    # IoSignature: one float input, one output tensor per actor head.
    assert [t.name for t in package.io_signature.inputs] == ["obs"]
    assert {t.name for t in package.io_signature.outputs} == {"kind", "mode", "goto"}


def test_same_weights_yield_the_same_content_addressed_identity() -> None:
    trainer = _train("ippo", _CFG)
    export = trainer.export()
    first = export_policy_package(export, "rover", version="0.1.0")
    second = export_policy_package(export, "rover", version="0.1.0")
    # The ONNX graph hash IS the artifact identity: same weights ⇒ same digest + same sidecar id.
    assert first.digest == second.digest
    assert first.digest.startswith("sha256:")
    assert content_id(first.document) == content_id(second.document)


def test_provenance_carries_seed_lockfile_and_comms() -> None:
    exported = export_policy_package(
        _train("mappo", _CFG, comms=True).export(),
        "rover",
        version="0.1.0",
        env_lockfile="sha256:lockfile",
    )
    package = exported.document.policy_package
    assert package.provenance is not None
    assert package.provenance.seed == 1
    assert package.provenance.env_lockfile == "sha256:lockfile"
    # comms regime serialized into Core's str field, round-trippable JSON for Guard.
    assert package.assumptions is not None
    comms = package.assumptions.comms_observability
    assert comms is not None
    assert json.loads(comms)["kind"] == "comms_model"


def test_surrogate_caveat_and_action_bounds_reach_the_sidecar() -> None:
    surrogate = _CFG.model_copy(update={"fidelity": "surrogate"})
    package = export_policy_package(
        _train("ippo", surrogate).export(), "rover", version="0.1.0"
    ).document.policy_package
    assert package.assumptions is not None
    assert any("high-fidelity" in c for c in package.assumptions.surrogate_fidelity_caveats)
    # Box heads declare their [-1, 1] bounds; discrete heads do not.
    assert "goto" in package.assumptions.action_bounds
    assert "kind" not in package.assumptions.action_bounds
    assert package.assumptions.action_bounds["goto"] == {"low": -1.0, "high": 1.0, "dim": 3}


def test_qmix_exports_a_kind_only_graph() -> None:
    package = export_policy_package(
        _train("qmix", _CFG).export(), "digger", version="0.1.0"
    ).document.policy_package
    # The value-based baseline decides only the discrete task selector.
    assert [t.name for t in package.io_signature.outputs] == ["kind"]
    assert package.assumptions is not None
    assert package.assumptions.action_bounds == {}


def test_export_all_agents_yields_one_package_per_heterogeneous_agent() -> None:
    packages = export_policy_packages(_train("ippo", _CFG).export(), version="0.1.0")
    assert set(packages) == {"rover", "digger", "relay"}
    # Distinct agents ⇒ distinct graphs (different capability-keyed spaces) ⇒ distinct digests.
    assert len({p.digest for p in packages.values()}) == 3
    for exported in packages.values():
        assert isinstance(exported.document, PolicyPackageDocument)
