"""Tests for ``astro_mine.core.policy`` — the uniform decision contract, the tier
sub-interfaces, composition (allocator+controller under one object), Guard-style
wrapping, active perception, and the conformance utility (RM-P0-CORE-03)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from astro_mine.core import compat
from astro_mine.core.messages.enums import ActionKind, TaskKind
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    ModeCommand,
    Observation,
    ProspectTask,
    Quat,
    StateSample,
    TaskDirective,
    Transform,
    Vec3,
    Volume,
)
from astro_mine.core.objective.enums import MetricDirection
from astro_mine.core.objective.model import MetricBinding, ObjectiveSpec, SuccessCriterion
from astro_mine.core.policy import (
    ComposedPolicy,
    Controller,
    DecisionContext,
    ModelRef,
    OnnxPolicy,
    Policy,
    PolicyContractError,
    PolicyPackage,
    PolicyPackageDocument,
    PolicyPackageValidationError,
    TensorDType,
    check_composition,
    check_policy,
    load_policy_package,
    validate_policy_package,
)
from astro_mine.core.units import MOON_BODY_FIXED

AgentId = str


def _obs(agent: str) -> Observation:
    return Observation(
        tick=0,
        sim_time_s=0.0,
        agent_id=agent,
        self_state=StateSample(
            agent_id=agent,
            frame=MOON_BODY_FIXED,
            pose=Transform(
                translation_m=Vec3(x=0.0, y=0.0, z=0.0),
                rotation_quat_xyzw=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        ),
    )


OBS: Mapping[str, Observation] = {"rover_a": _obs("rover_a"), "rover_b": _obs("rover_b")}
CTX = DecisionContext()


# --- reference policies (controller, allocator, active-perception, recording, shield) ---


class StandbyController:
    """Reactive controller: a MODE 'idle' setpoint per agent."""

    def decide(
        self, observations: Mapping[str, Observation], context: DecisionContext
    ) -> ActionBatch:
        return ActionBatch(
            actions=[
                Action(agent_id=a, kind=ActionKind.MODE, mode=ModeCommand(mode="idle"))
                for a in observations
            ]
        )


class StandbyAllocator:
    """Allocator: a STANDBY task assignment per agent."""

    def decide(
        self, observations: Mapping[str, Observation], context: DecisionContext
    ) -> ActionBatch:
        return ActionBatch(
            actions=[
                Action(
                    agent_id=a, kind=ActionKind.TASK, task=TaskDirective(task_kind=TaskKind.STANDBY)
                )
                for a in observations
            ]
        )


class ProspectingPolicy:
    """Active-perception policy: emits a PROSPECT task with an information-gain target —
    acting to reduce belief uncertainty, never reading a point ground-truth value."""

    def decide(
        self, observations: Mapping[str, Observation], context: DecisionContext
    ) -> ActionBatch:
        region = Volume(
            frame="body",
            center_m=Vec3(x=0.0, y=0.0, z=0.0),
            dimensions_m=Vec3(x=10.0, y=10.0, z=1.0),
        )
        return ActionBatch(
            actions=[
                Action(
                    agent_id=a,
                    kind=ActionKind.TASK,
                    task=TaskDirective(
                        task_kind=TaskKind.PROSPECT,
                        prospect=ProspectTask(region=region, info_gain_target=0.5),
                    ),
                )
                for a in observations
            ]
        )


class RecordingController:
    """Controller that records the upstream batch it was handed (to prove composition
    threads the prior tier's assignments through ``context.upstream``)."""

    def __init__(self) -> None:
        self.seen_upstream: ActionBatch | None = None

    def decide(
        self, observations: Mapping[str, Observation], context: DecisionContext
    ) -> ActionBatch:
        self.seen_upstream = context.upstream
        return ActionBatch(
            actions=[
                Action(agent_id=a, kind=ActionKind.MODE, mode=ModeCommand(mode="idle"))
                for a in observations
            ]
        )


class SafetyShield:
    """A Guard-style wrapper: any policy is wrappable because a shield is itself a Policy.
    This one forbids high-level TASK actions, passing only control actions through."""

    def __init__(self, inner: Policy) -> None:
        self._inner = inner

    def decide(
        self, observations: Mapping[str, Observation], context: DecisionContext
    ) -> ActionBatch:
        inner = self._inner.decide(observations, context)
        return ActionBatch(actions=[a for a in inner.actions if a.kind != ActionKind.TASK])


# --- the contract ----------------------------------------------------------------


def test_reference_policies_satisfy_protocol() -> None:
    assert isinstance(StandbyController(), Policy)
    assert isinstance(StandbyAllocator(), Policy)
    # A controller annotated against the Controller sub-interface is structurally a Policy.
    controller: Controller = StandbyController()
    assert isinstance(controller, Policy)


def test_controller_conforms_and_drives_sim() -> None:
    out = check_policy(StandbyController(), OBS, CTX)
    assert {a.agent_id for a in out.actions} == set(OBS)
    assert all(a.kind == ActionKind.MODE for a in out.actions)


def test_allocator_conforms() -> None:
    out = check_policy(StandbyAllocator(), OBS, CTX)
    assert all(a.kind == ActionKind.TASK and a.task is not None for a in out.actions)


def test_active_perception_is_expressible() -> None:
    out = check_policy(ProspectingPolicy(), OBS, CTX)
    for action in out.actions:
        assert action.task is not None
        assert action.task.task_kind == TaskKind.PROSPECT
        assert action.task.prospect is not None
        assert action.task.prospect.info_gain_target == 0.5


def test_controller_is_deterministic() -> None:
    policy = StandbyController()
    assert policy.decide(OBS, CTX) == policy.decide(OBS, CTX)


# --- composition (the acceptance criterion) --------------------------------------


def test_allocator_and_controller_compose_under_one_object() -> None:
    allocator = StandbyAllocator()
    recording = RecordingController()
    composed = ComposedPolicy(allocator, recording)
    assert isinstance(composed, Policy)
    assert composed.stages == (allocator, recording)

    out = check_policy(composed, OBS, CTX)
    # The controller saw the allocator's TASK assignments as its upstream.
    assert recording.seen_upstream is not None
    assert all(a.kind == ActionKind.TASK for a in recording.seen_upstream.actions)
    # The composed object's output is the final (controller) stage's batch.
    assert all(a.kind == ActionKind.MODE for a in out.actions)


def test_compose_threads_incoming_upstream() -> None:
    recording = RecordingController()
    seed_batch = StandbyAllocator().decide(OBS, CTX)
    ComposedPolicy(recording).decide(OBS, DecisionContext(upstream=seed_batch))
    assert recording.seen_upstream == seed_batch


def test_compose_requires_at_least_one_stage() -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        ComposedPolicy()


# --- Guard-style wrapping (any policy is wrappable) ------------------------------


def test_guard_can_wrap_any_policy() -> None:
    # Wrapping an allocator (TASK output) → the shield strips the high-level tasks.
    shielded_alloc = SafetyShield(StandbyAllocator())
    assert isinstance(shielded_alloc, Policy)
    assert check_policy(shielded_alloc, OBS, CTX).actions == []
    # Wrapping a controller (MODE output) → control actions pass through.
    shielded_ctrl = SafetyShield(StandbyController())
    assert all(a.kind == ActionKind.MODE for a in check_policy(shielded_ctrl, OBS, CTX).actions)


# --- DecisionContext --------------------------------------------------------------


def test_decision_context_defaults() -> None:
    ctx = DecisionContext()
    assert ctx.sim_time_s == 0.0
    assert ctx.objective is None and ctx.upstream is None and ctx.seed is None
    assert ctx.extras == {}


def test_decision_context_carries_objective_and_seed() -> None:
    objective = ObjectiveSpec(
        id="o1",
        name="Characterize the PSR",
        success_criteria=[
            SuccessCriterion(
                id="c1",
                binding=MetricBinding(
                    metric="information_gain",
                    unit="dimensionless",
                    direction=MetricDirection.HIGHER_BETTER,
                    target=1.0,
                    tolerance=0.1,
                ),
            )
        ],
    )
    ctx = DecisionContext(
        sim_time_s=5.0, objective=objective, seed=42, extras={"belief": "ref://b"}
    )
    assert ctx.objective is not None and ctx.objective.id == "o1"
    assert ctx.seed == 42 and ctx.extras["belief"] == "ref://b"


# --- conformance failure modes ----------------------------------------------------


def test_check_policy_rejects_non_policy() -> None:
    with pytest.raises(PolicyContractError):
        check_policy(object(), OBS, CTX)  # type: ignore[arg-type]


class _BadOutput:
    def decide(self, observations: Mapping[str, Observation], context: DecisionContext) -> Any:
        return {}


class _InvalidBatch:
    def decide(
        self, observations: Mapping[str, Observation], context: DecisionContext
    ) -> ActionBatch:
        # A MODE action missing its 'mode' payload — fails the tagged-union check.
        return ActionBatch(actions=[Action(agent_id="rover_a", kind=ActionKind.MODE)])


def test_check_policy_rejects_non_action_batch() -> None:
    with pytest.raises(PolicyContractError, match="must return an ActionBatch"):
        check_policy(_BadOutput(), OBS, CTX)


def test_check_policy_rejects_invalid_action_batch() -> None:
    with pytest.raises(PolicyContractError, match="invalid ActionBatch"):
        check_policy(_InvalidBatch(), OBS, CTX)


# --- PolicyPackage: the exported-ONNX-policy sidecar (RM-P1-CORE-01) --------------


def _package_doc(**overrides: Any) -> dict[str, Any]:
    """Smallest valid PolicyPackage document (plus any field overrides)."""
    package: dict[str, Any] = {
        "name": "greedy-onnx",
        "version": "0.1.0",
        "onnx_model": {"digest": "sha256:aa"},
        "core_interfaces": {"policy": "0.1.0", "messages": "0.1.0"},
    }
    package.update(overrides)
    return {"policy_package_version": "0.1", "policy_package": package}


PKG = PolicyPackage(
    name="greedy-onnx",
    version="0.1.0",
    onnx_model=ModelRef(digest="sha256:aa"),
    core_interfaces={"policy": "0.1.0"},
)


def _fake_infer(observations: Mapping[str, Observation], context: DecisionContext) -> ActionBatch:
    """A stand-in for a host ONNX-Runtime session: emits a MODE 'idle' control action per
    agent (Core ships no runtime; the host supplies inference)."""
    return ActionBatch(
        actions=[
            Action(agent_id=a, kind=ActionKind.MODE, mode=ModeCommand(mode="idle"))
            for a in observations
        ]
    )


EXAMPLES = sorted(
    (Path(__file__).resolve().parents[2] / "examples" / "policy").glob("*.policy-package.yaml")
)


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.name)
def test_examples_corpus_loads(path: Path) -> None:
    # The checked-in corpus is the shared parity fixture: the Rust fast path validates the same
    # files in `cargo test` (lib.rs::policy_package_corpus_is_structurally_valid).
    doc = load_policy_package(path.read_text(encoding="utf-8"))
    assert doc.policy_package_version == "0.1"
    assert doc.policy_package.onnx_model.digest.startswith("sha256:")


def test_examples_corpus_is_not_empty() -> None:
    assert EXAMPLES, "the policy example corpus is the Rust parity fixture; it must not be empty"


def test_policy_package_loads_and_validates() -> None:
    doc = load_policy_package(json.dumps(_package_doc()).encode())  # bytes path
    assert isinstance(doc, PolicyPackageDocument)
    assert doc.policy_package.onnx_model.digest == "sha256:aa"
    validate_policy_package(doc)
    validate_policy_package(_package_doc())  # dict path
    validate_policy_package(json.dumps(_package_doc()))  # text path


def test_policy_package_full_document_round_trips() -> None:
    doc = _package_doc(
        io_signature={
            "inputs": [{"name": "obs", "dtype": "float32", "shape": [-1, 8]}],
            "outputs": [{"name": "act", "dtype": "float32", "shape": [-1, 2]}],
            "observation_space": {"rover": {"kind": "box"}},
            "action_space": {"rover": {"kind": "box"}},
        },
        assumptions={
            "comms_observability": "intermittent",
            "surrogate_fidelity_caveats": ["terramechanics"],
            "deterministic": True,
        },
        provenance={"code_version": "0.1.0", "seed": 7},
    )
    pkg = load_policy_package(json.dumps(doc)).policy_package
    assert pkg.io_signature is not None
    assert pkg.io_signature.inputs[0].dtype is TensorDType.FLOAT32
    assert pkg.assumptions is not None and pkg.assumptions.deterministic is True


def test_policy_package_rejects_unknown_field() -> None:
    doc = _package_doc()
    doc["policy_package"]["bogus"] = 1
    with pytest.raises(PolicyPackageValidationError):
        validate_policy_package(doc)


def test_policy_package_rejects_missing_onnx_model() -> None:
    doc = _package_doc()
    del doc["policy_package"]["onnx_model"]
    with pytest.raises(PolicyPackageValidationError):
        load_policy_package(json.dumps(doc))


def test_policy_package_rejects_bad_tensor_dtype() -> None:
    doc = _package_doc(
        io_signature={"inputs": [{"name": "obs", "dtype": "float16", "shape": [-1, 8]}]}
    )
    with pytest.raises(PolicyPackageValidationError):
        validate_policy_package(doc)


def test_policy_package_rejects_non_mapping() -> None:
    with pytest.raises(PolicyPackageValidationError, match="must be a YAML/JSON mapping"):
        load_policy_package("[1, 2, 3]")


def test_validate_policy_package_rejects_unknown_type() -> None:
    with pytest.raises(PolicyPackageValidationError, match="cannot validate object of type"):
        validate_policy_package(object())  # type: ignore[arg-type]


def test_onnx_policy_satisfies_policy_and_delegates() -> None:
    policy = OnnxPolicy(PKG, _fake_infer)
    assert isinstance(policy, Policy)
    controller: Controller = policy  # an ONNX policy type-checks as a Controller tier
    out = check_policy(controller, OBS, CTX)
    assert {a.agent_id for a in out.actions} == set(OBS)
    assert all(a.kind == ActionKind.MODE for a in out.actions)
    assert policy.package.name == "greedy-onnx"


def test_onnx_policy_refuses_incompatible_core_version() -> None:
    pkg = PolicyPackage(
        name="p",
        version="0.1.0",
        onnx_model=ModelRef(digest="sha256:bb"),
        core_interfaces={"policy": "9.9.9"},
    )
    with pytest.raises(compat.IncompatibleCoreInterface, match="policy"):
        OnnxPolicy(pkg, _fake_infer)
    # an explicit `provided` override negotiates against a hypothetical future Core
    OnnxPolicy(pkg, _fake_infer, provided={"policy": "9.9.0"})


def test_shielded_allocator_delegating_to_onnx_controller_composes() -> None:
    """Acceptance: a Guard-wrapping -> Allocate delegation -> ONNX controller stack
    type-checks and passes the Core contract tests."""
    onnx_controller = OnnxPolicy(PKG, _fake_infer)
    out = check_composition(
        SafetyShield(StandbyAllocator()), onnx_controller, observations=OBS, context=CTX
    )
    assert {a.agent_id for a in out.actions} == set(OBS)
    assert all(a.kind == ActionKind.MODE for a in out.actions)


def test_check_composition_requires_a_stage() -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        check_composition(observations=OBS, context=CTX)
