"""Behavior-tree node model (RM-P1-MIND-02).

The in-memory AST the executive ticks under the ``behavior_tree`` execution kind — the
reactive glue that sequences and guards the tier backends. Groot-compatible (the
BehaviorTree.CPP XML dialect; :mod:`astro_mine.mind.bt.xml` is the concrete syntax); this
module is the abstract syntax. Frozen dataclasses, closed node vocabularies (append-only,
like the stack-spec enums) — a tree is a value, so the same tree ticks identically given the
same inputs (principle 9).

Three leaf kinds realise the "uniform reactive scaffold across tiers" (mind.md §4): an
:class:`ActionNode` is **planner-invoking** (calls a mission/TAMP backend), **policy-invoking**
(runs a controller / ONNX policy), or **primitive** (emits an SADF-declared action);
:class:`ConditionNode` gates a branch on the belief (e.g. whether fresh input arrived from the
tier above). The composites are the graceful-degradation substrate: a :class:`ControlNode`
``fallback`` (selector) tries its children in order and takes the first that succeeds, so an
explicit degradation branch is reachable when the primary branch fails on stale/absent input —
the branch RM-P1-MIND-06 validates under injected comms loss.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
    "ActionNode",
    "BTNode",
    "BehaviorTree",
    "ConditionKind",
    "ConditionNode",
    "ControlKind",
    "ControlNode",
    "DecoratorKind",
    "DecoratorNode",
    "InvokeKind",
    "NodeStatus",
]


class NodeStatus(StrEnum):
    """A node's tick result (BehaviorTree.CPP semantics). ``running`` is reserved for
    multi-tick leaves; the reference leaves settle to ``success``/``failure`` each tick, and
    the composites propagate ``running`` faithfully so a native BehaviorTree.CPP binding
    drops in unchanged."""

    SUCCESS = "success"
    FAILURE = "failure"
    RUNNING = "running"


class ControlKind(StrEnum):
    """A composite (internal) node. ``sequence`` succeeds iff every child succeeds (stops at
    the first non-success); ``fallback`` (selector) succeeds at the first child that
    succeeds (the explicit graceful-degradation branch, principle 4). Both are evaluated
    reactively — re-ticked from the first child each executive tick — which is the natural
    posture for the per-tick reactive executive (no cross-tick node memory, so determinism
    holds without hidden state)."""

    SEQUENCE = "sequence"
    FALLBACK = "fallback"


class DecoratorKind(StrEnum):
    """A single-child decorator. ``inverter`` swaps success/failure; ``force_success`` /
    ``force_failure`` coerce the child's result — the stateless decorators, so the tree
    stays a pure function of its inputs. Stateful decorators (retry/repeat with counters)
    are appended when a multi-tick leaf lands."""

    INVERTER = "inverter"
    FORCE_SUCCESS = "force_success"
    FORCE_FAILURE = "force_failure"


class InvokeKind(StrEnum):
    """What an :class:`ActionNode` leaf does when ticked. ``planner`` invokes a mission/TAMP
    tier backend, ``policy`` runs a controller / ONNX policy, ``primitive`` emits an
    SADF-declared action directly (e.g. a safe-idle standby). The uniform reactive scaffold
    across tiers (mind.md §4)."""

    PLANNER = "planner"
    POLICY = "policy"
    PRIMITIVE = "primitive"


class ConditionKind(StrEnum):
    """A :class:`ConditionNode` predicate over the belief/context. ``fresh_upstream`` is
    ``success`` iff the tier above produced fresh (non-empty) input this tick — the guard on
    a degradation branch that fires when the tier above yields nothing (RM-P1-MIND-02
    acceptance). Comms-regime conditions (``earth_contact`` etc.) append in RM-P1-MIND-06."""

    FRESH_UPSTREAM = "fresh_upstream"


@dataclass(frozen=True, slots=True)
class ControlNode:
    """A composite node: a ``kind`` (sequence / fallback) over an ordered tuple of
    ``children``. ``node_id`` is the optional Groot name (for inspection/tracing)."""

    kind: ControlKind
    children: tuple[BTNode, ...]
    node_id: str | None = None


@dataclass(frozen=True, slots=True)
class DecoratorNode:
    """A single-child decorator transforming its ``child``'s status."""

    kind: DecoratorKind
    child: BTNode
    node_id: str | None = None


@dataclass(frozen=True, slots=True)
class ActionNode:
    """A leaf that produces (part of) the tick's proposed action. ``invoke`` selects the
    kind; ``ref`` names the tier role (``planner``/``policy``) or the SADF action kind
    (``primitive``); ``params`` carry leaf configuration (e.g. a primitive's directive).
    ``node_id`` is the Groot ``ID`` and identifies the node in the trace."""

    invoke: InvokeKind
    ref: str
    node_id: str
    params: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ConditionNode:
    """A leaf predicate (``check``) over the belief/context, with optional ``params``."""

    check: ConditionKind
    node_id: str
    params: Mapping[str, str] = field(default_factory=dict)


#: The BT node union.
BTNode = ControlNode | DecoratorNode | ActionNode | ConditionNode


@dataclass(frozen=True, slots=True)
class BehaviorTree:
    """A parsed, validated behavior tree: its ``tree_id`` (the Groot ``main_tree`` ID), the
    ``root`` node, and the ``format_version`` (the Groot ``BTCPP_format``). Versioned with
    the stack spec (mind.md §5); the authored form is the XML, this is its resolved AST."""

    tree_id: str
    root: BTNode
    format_version: str = "4"

    def tier_refs(self) -> frozenset[str]:
        """The set of tier roles the tree's planner/policy leaves invoke — what the composer
        cross-checks against the composed hierarchy's tiers."""
        return frozenset(_iter_tier_refs(self.root))


def _iter_tier_refs(node: BTNode) -> frozenset[str]:
    if isinstance(node, ActionNode):
        return frozenset({node.ref}) if node.invoke is not InvokeKind.PRIMITIVE else frozenset()
    if isinstance(node, ControlNode):
        return (
            frozenset().union(*(_iter_tier_refs(c) for c in node.children))
            if node.children
            else frozenset()
        )
    if isinstance(node, DecoratorNode):
        return _iter_tier_refs(node.child)
    return frozenset()
