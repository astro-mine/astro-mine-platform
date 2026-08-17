# SPDX-License-Identifier: Apache-2.0
"""Policy/Planner API contract-test utility (RM-P0-CORE-03).

The consumer-driven conformance check an implementor (a Bench baseline, a Mind tier, an
Allocate solver) runs in its own CI — analogous to
:func:`astro_mine.core.env.check_environment`. It drives ``decide`` once and asserts the
contract: the object satisfies :class:`Policy`, returns an
:class:`~astro_mine.core.messages.ActionBatch`, and that batch is a well-formed,
**Sim-consumable** control-plane message (the same validation the Environment API's
``step`` applies — Bench's "drives the anchor through Sim"). Raises
:class:`PolicyContractError`.

Determinism is intentionally **not** asserted here: it is a property the runtime
enforces via the seed (sim determinism gates), and a stateful policy legitimately
advances across ticks, so a double-call check would be wrong.
"""

from __future__ import annotations

from collections.abc import Mapping

from astro_mine.core.env.model import AgentId
from astro_mine.core.messages.loader import MessagesValidationError, validate_action_batch
from astro_mine.core.messages.model import ActionBatch, Observation
from astro_mine.core.policy.compose import ComposedPolicy
from astro_mine.core.policy.model import DecisionContext
from astro_mine.core.policy.protocol import Policy

__all__ = ["PolicyContractError", "check_composition", "check_policy"]


class PolicyContractError(AssertionError):
    """Raised when a policy violates the Core Policy/Planner API contract."""


def check_policy(
    policy: Policy,
    observations: Mapping[AgentId, Observation],
    context: DecisionContext,
) -> ActionBatch:
    """Assert ``policy`` honors the Policy/Planner API v0.1 contract; return its output.

    Checks it satisfies the :class:`Policy` protocol, that ``decide`` returns an
    :class:`ActionBatch`, and that the batch passes
    :func:`~astro_mine.core.messages.validate_action_batch` (the Sim-consumable,
    tagged-union-consistent control-plane contract).
    """
    if not isinstance(policy, Policy):
        raise PolicyContractError("object does not satisfy the Policy protocol (missing decide)")
    result = policy.decide(observations, context)
    if not isinstance(result, ActionBatch):
        raise PolicyContractError(
            f"decide() must return an ActionBatch, got {type(result).__name__}"
        )
    try:
        validate_action_batch(result)
    except MessagesValidationError as exc:
        raise PolicyContractError(f"decide() produced an invalid ActionBatch: {exc}") from exc
    return result


def check_composition(
    *stages: Policy,
    observations: Mapping[AgentId, Observation],
    context: DecisionContext,
) -> ActionBatch:
    """Assert a *composed* policy stack honors the Policy contract end-to-end; return its
    output batch.

    The composition contract test a consumer runs in its own CI (e.g. Mind's ``compose/``
    step validating a hierarchy graph): it builds a :class:`ComposedPolicy` from the tiers
    — a Guard shield wrapping an allocator delegating to an ONNX controller, say — and
    drives :func:`check_policy` over the whole stack, proving the composition type-checks
    and yields a Sim-consumable :class:`ActionBatch`. Raises :class:`PolicyContractError`
    (from the underlying check) or :class:`ValueError` if no stages are given."""
    return check_policy(ComposedPolicy(*stages), observations, context)
