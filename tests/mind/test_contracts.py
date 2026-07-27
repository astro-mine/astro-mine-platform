"""Consumer-driven contract tests against Core (conventions.md §11).

Mind proves it honors the Core Policy/Planner and Environment contracts it composes over,
using Core's own conformance checkers — so a Core interface change surfaces here.
"""

from __future__ import annotations

from collections.abc import Mapping

from astro_mine.core.env.conformance import check_environment
from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.model import Observation
from astro_mine.core.policy.compose import ComposedPolicy
from astro_mine.core.policy.conformance import check_policy
from astro_mine.core.policy.model import DecisionContext
from astro_mine.mind.registry import TierRegistry
from tests.mind.support.toy_env import ToyProspectingEnv

_TIERS = ["mind.reference.mission", "mind.reference.tamp", "mind.reference.control"]
_SHIELD = "mind.reference.shield"


def _observations() -> Mapping[AgentId, Observation]:
    return ToyProspectingEnv().reset().observations


def test_toy_env_satisfies_environment_contract() -> None:
    check_environment(ToyProspectingEnv())


def test_reference_tiers_satisfy_policy_contract() -> None:
    registry = TierRegistry.from_entry_points()
    observations = _observations()
    context = DecisionContext()
    for name in [*_TIERS, _SHIELD]:
        check_policy(registry.instantiate(name), observations, context)


def test_composed_tiers_satisfy_policy_contract() -> None:
    # The composed hierarchy is itself a Policy (Core's ComposedPolicy seam), so the
    # composition honors the same contract each tier does.
    registry = TierRegistry.from_entry_points()
    composed = ComposedPolicy(*(registry.instantiate(name) for name in _TIERS))
    check_policy(composed, _observations(), DecisionContext())
