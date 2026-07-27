"""Behavior-tree scaffold — XML, engine, and executive integration (RM-P1-MIND-02)."""

from __future__ import annotations

import pytest

from astro_mine.core.messages.model import ActionBatch
from astro_mine.core.policy.model import DecisionContext
from astro_mine.mind.bt.engine import BehaviorTreeEngine, PrimitiveError
from astro_mine.mind.bt.model import (
    ActionNode,
    BehaviorTree,
    ConditionKind,
    ConditionNode,
    ControlKind,
    ControlNode,
    DecoratorKind,
    DecoratorNode,
    InvokeKind,
)
from astro_mine.mind.bt.xml import BehaviorTreeXMLError, parse_behavior_tree, to_xml
from astro_mine.mind.compose import ComposeError, compose
from astro_mine.mind.exec import Executive
from astro_mine.mind.reference import load_reference_bt, load_stack_resource
from tests.mind.support.harness import (
    RaisingPolicy,
    assert_deterministic_trace,
    assert_shielded_egress,
    compose_reference_bt,
    policy_plugin,
    reference_doc_with_shield,
    reference_registry,
)
from tests.mind.support.toy_env import ToyProspectingEnv


class _EmptyPolicy:
    def decide(self, observations, context):  # type: ignore[no-untyped-def]
        return ActionBatch()


# --- XML parse / validate / round-trip -------------------------------------------------


def test_reference_bt_round_trips() -> None:
    tree = load_reference_bt()
    assert tree.tree_id == "LunarProspecting"
    assert tree.tier_refs() == {"mission", "tamp", "control"}
    assert parse_behavior_tree(to_xml(tree)) == tree


def test_parse_bare_behavior_tree_element() -> None:
    tree = parse_behavior_tree(
        '<BehaviorTree ID="T"><Action ID="A" kind="primitive" action="standby"/></BehaviorTree>'
    )
    assert tree.tree_id == "T"
    assert isinstance(tree.root, ActionNode)


@pytest.mark.parametrize(
    "xml",
    [
        "<not-a-tree/>",
        "<root></root>",
        '<root main_tree_to_execute="X"><BehaviorTree ID="Y"/></root>',
        "<root><BehaviorTree ID='A'/><BehaviorTree ID='B'/></root>",
        '<BehaviorTree ID="T"><Sequence/></BehaviorTree>',
        '<BehaviorTree><Action ID="a" kind="primitive" action="standby"/></BehaviorTree>',
        '<BehaviorTree ID="T"><Action ID="a" kind="planner"/></BehaviorTree>',
        '<BehaviorTree ID="T"><Action ID="a" kind="nope" tier="mission"/></BehaviorTree>',
        '<BehaviorTree ID="T"><Condition ID="c" kind="nope"/></BehaviorTree>',
        '<BehaviorTree ID="T"><Inverter/></BehaviorTree>',
        '<BehaviorTree ID="T"><Bogus/></BehaviorTree>',
        '<BehaviorTree ID="T"><Action ID="a" kind="primitive"/></BehaviorTree>',
        "<root>&bad;</root>",
    ],
)
def test_malformed_bt_xml_is_rejected(xml: str) -> None:
    with pytest.raises(BehaviorTreeXMLError):
        parse_behavior_tree(xml)


def test_parse_accepts_bytes_and_reactive_aliases() -> None:
    tree = parse_behavior_tree(
        b'<BehaviorTree ID="T"><ReactiveFallback><Condition ID="c" kind="fresh_upstream"/>'
        b'<Action ID="a" kind="primitive" action="standby"/></ReactiveFallback></BehaviorTree>'
    )
    assert isinstance(tree.root, ControlNode)
    assert tree.root.kind is ControlKind.FALLBACK


# --- engine --------------------------------------------------------------------------


def _control_map():  # type: ignore[no-untyped-def]
    registry = reference_registry()
    return {
        "mission": registry.instantiate("mind.reference.mission"),
        "tamp": registry.instantiate("mind.reference.tamp"),
        "control": registry.instantiate("mind.reference.control"),
    }


def _observations():  # type: ignore[no-untyped-def]
    return ToyProspectingEnv(horizon=4).reset().observations


def test_engine_threads_tiers_on_happy_path() -> None:
    engine = BehaviorTreeEngine(load_reference_bt(), _control_map())
    batch, records = engine.tick(_observations(), DecisionContext(seed=7))
    assert [a.actuator is not None for a in batch.actions] == [True, True]  # control setpoints
    assert [r.role for r in records] == ["mission", "tamp", "control"]
    assert not any(r.fallback_used for r in records)


def test_selector_fallback_activates_on_stale_input() -> None:
    # A tree whose control branch is gated on fresh upstream; with an empty TAMP the condition
    # fails and the Fallback degrades to the safe-idle standby primitive (RM-P1-MIND-02 accept).
    tree = load_reference_bt()
    tiers = {**_control_map(), "tamp": _EmptyPolicy()}
    batch, records = BehaviorTreeEngine(tree, tiers).tick(_observations(), DecisionContext(seed=7))
    assert all(a.task is not None and a.task.task_kind.value == "standby" for a in batch.actions)
    assert any(r.role == "primitive.standby" and r.fallback_used for r in records)


def test_engine_catches_tier_exception_as_failure() -> None:
    tiers = {**_control_map(), "control": RaisingPolicy()}
    batch, records = BehaviorTreeEngine(load_reference_bt(), tiers).tick(
        _observations(), DecisionContext(seed=7)
    )
    # control raised -> its branch failed -> Fallback took the safe-idle standby.
    assert all(a.task is not None and a.task.task_kind.value == "standby" for a in batch.actions)
    assert any(r.role == "control" and r.fallback_used for r in records)


def test_unresolved_tier_ref_fails_the_branch() -> None:
    tree = BehaviorTree(
        tree_id="T",
        root=ControlNode(
            kind=ControlKind.FALLBACK,
            children=(
                ActionNode(invoke=InvokeKind.POLICY, ref="missing", node_id="x"),
                ActionNode(invoke=InvokeKind.PRIMITIVE, ref="standby", node_id="idle"),
            ),
        ),
    )
    batch, _ = BehaviorTreeEngine(tree, {}).tick(_observations(), DecisionContext())
    assert all(a.task is not None and a.task.task_kind.value == "standby" for a in batch.actions)


def _standby_only(batch: ActionBatch) -> bool:
    return bool(batch.actions) and all(
        a.task is not None and a.task.task_kind.value == "standby" for a in batch.actions
    )


@pytest.mark.parametrize(
    ("kind", "fresh", "standby_runs"),
    [
        (DecoratorKind.INVERTER, False, True),  # fresh fails -> inverter succeeds -> reach standby
        (DecoratorKind.INVERTER, True, False),  # fresh succeeds -> inverter fails -> stop
        (DecoratorKind.FORCE_SUCCESS, True, True),  # forced success -> reach standby
        (DecoratorKind.FORCE_FAILURE, False, False),  # forced failure -> stop
    ],
)
def test_decorators_transform_status(kind, fresh, standby_runs) -> None:  # type: ignore[no-untyped-def]
    # A Sequence[Decorator(fresh_upstream), standby]: the standby leaf is reached iff the
    # decorated condition succeeds, so the emitted batch reveals the decorator's transform.
    tree = BehaviorTree(
        tree_id="T",
        root=ControlNode(
            kind=ControlKind.SEQUENCE,
            children=(
                DecoratorNode(
                    kind=kind, child=ConditionNode(check=ConditionKind.FRESH_UPSTREAM, node_id="c")
                ),
                ActionNode(invoke=InvokeKind.PRIMITIVE, ref="standby", node_id="idle"),
            ),
        ),
    )
    upstream = (
        _control_map()["mission"].decide(_observations(), DecisionContext())
        if fresh
        else ActionBatch()
    )
    batch, _ = BehaviorTreeEngine(tree, {}).tick(
        _observations(), DecisionContext(upstream=upstream)
    )
    assert _standby_only(batch) is standby_runs


def test_unknown_primitive_raises() -> None:
    tree = BehaviorTree(
        tree_id="T", root=ActionNode(invoke=InvokeKind.PRIMITIVE, ref="teleport", node_id="p")
    )
    with pytest.raises(PrimitiveError):
        BehaviorTreeEngine(tree, {}).tick(_observations(), DecisionContext())


# --- executive integration -----------------------------------------------------------


def test_bt_stack_runs_and_moves_agents() -> None:
    graph = compose_reference_bt()
    result = Executive(graph).run(ToyProspectingEnv(horizon=6), max_ticks=6, seed=7)
    assert result.ticks_run == 6
    for obs in result.final_observations.values():
        assert obs.self_state.pose.translation_m.x > 0.0


def test_bt_stack_is_deterministic() -> None:
    assert_deterministic_trace(
        lambda: Executive(compose_reference_bt()).run(
            ToyProspectingEnv(horizon=6), max_ticks=6, seed=7
        )
    )


def test_bt_stack_guard_is_the_only_output() -> None:
    registry = reference_registry()
    registry.register(policy_plugin("test.tagging-shield", lambda params: _TaggingShield()))
    doc = reference_doc_with_shield("test.tagging-shield")
    spec = doc.stack_spec.model_copy(
        update={"execution": load_stack_resource("lunar_prospecting_bt.yaml").stack_spec.execution}
    )
    from astro_mine.mind.spec.model import StackSpecDocument

    graph = compose(
        StackSpecDocument(stack_spec_version="0.1", stack_spec=spec),
        registry,
        seed=7,
        behavior_tree=load_reference_bt(),
    )
    result = Executive(graph).run(ToyProspectingEnv(horizon=4), max_ticks=4, seed=7)
    assert_shielded_egress(result)


def test_compose_behavior_tree_requires_a_tree() -> None:
    with pytest.raises(ComposeError, match="requires the parsed BehaviorTree"):
        compose(load_stack_resource("lunar_prospecting_bt.yaml"), reference_registry(), seed=7)


def test_compose_behavior_tree_rejects_unresolved_ref() -> None:
    tree = BehaviorTree(
        tree_id="T", root=ActionNode(invoke=InvokeKind.PLANNER, ref="nonexistent", node_id="x")
    )
    with pytest.raises(ComposeError, match="invokes tier"):
        compose(
            load_stack_resource("lunar_prospecting_bt.yaml"),
            reference_registry(),
            seed=7,
            behavior_tree=tree,
        )


class _TaggingShield:
    def decide(self, observations, context):  # type: ignore[no-untyped-def]
        from tests.mind.support.harness import TAGGING_SENTINEL

        base = context.upstream if context.upstream is not None else ActionBatch()
        return ActionBatch(
            actions=[a.model_copy(update={"sim_time_s": TAGGING_SENTINEL}) for a in base.actions]
        )
