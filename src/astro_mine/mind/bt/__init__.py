# SPDX-License-Identifier: Apache-2.0
"""Behavior-tree execution scaffold (RM-P1-MIND-02).

The ``behavior_tree`` execution representation: a Groot-compatible XML dialect
(:mod:`~astro_mine.mind.bt.xml`) over a closed node AST (:mod:`~astro_mine.mind.bt.model`),
ticked reactively by a deterministic engine (:mod:`~astro_mine.mind.bt.engine`) and offered to
the executive as a drop-in :class:`~astro_mine.mind.bt.strategy.BehaviorTreeStrategy`. Behavior
trees are the reactive glue that sequences and guards the tier backends; their selector/decorator
composites are the explicit graceful-degradation branches (principle 4) the RM-P1-MIND-06
degrade-not-collapse work later validates. Every action the tree proposes still passes through
the mandatory shield — Guard-wrapped output remains the only output.

**No native BehaviorTree.CPP binding — and there will not be one (re-scoped, astro-mine-mind#13).**
mind.md §4/§11 names BehaviorTree.CPP "exposed to Python via pybind11" as the BT engine, and the
sibling native backends (unified-planning, OMPL/FCL) *were* integrated behind their contracts. BT
is the one that cannot be: **BehaviorTree.CPP distributes no Python binding** — the project ships
none, and no candidate package exists on PyPI. Binding it would mean vendoring a CMake + pybind11
build of a C++ library into this pure-Python wheel, which is disproportionate for Phase 1 and
would break the tier-1 "local install MUST always work" rule (conventions.md §7).

The cost of *not* binding it is close to zero, because the thing BehaviorTree.CPP actually gives
the platform is the **interop format**, and that is implemented here: the v4 Groot XML dialect
parses, validates, and round-trips (``parse(to_xml(t)) == t``), so trees author and inspect in
Groot and would load unchanged into a native engine. The AST carries BT.CPP's ``running`` status
and the composites propagate it faithfully, so a native engine drops in behind this same seam if a
binding ever ships. What a native engine would add over the reference walker is throughput and
stateful multi-tick decorators — neither on the Phase-1 critical path, and the reactive per-tick
posture is what keeps a seeded run bit-reproducible.

This re-scopes the *implementation* named in RM-P1-MIND-02, not the deliverable: the Groot-XML
execution scaffold with selector/decorator fallbacks ships. A docs PR against ``mind.md §4/§11``
and ``roadmap/phase-1-autonomy-studio.md`` should record the substitution.
"""

from __future__ import annotations

from astro_mine.mind.bt.engine import BehaviorTreeEngine, PrimitiveError
from astro_mine.mind.bt.model import (
    ActionNode,
    BehaviorTree,
    ConditionNode,
    ControlNode,
    DecoratorNode,
    NodeStatus,
)
from astro_mine.mind.bt.strategy import BehaviorTreeStrategy
from astro_mine.mind.bt.xml import BehaviorTreeXMLError, parse_behavior_tree, to_xml

__all__ = [
    "ActionNode",
    "BehaviorTree",
    "BehaviorTreeEngine",
    "BehaviorTreeStrategy",
    "BehaviorTreeXMLError",
    "ConditionNode",
    "ControlNode",
    "DecoratorNode",
    "NodeStatus",
    "PrimitiveError",
    "parse_behavior_tree",
    "to_xml",
]
