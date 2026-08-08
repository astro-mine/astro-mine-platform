"""Environment API — how a simulatable world is observed and acted upon (RM-P0-CORE-02).

``reset()/step(action) -> observation, reward?, info``, generalized for multi-agent
operation, partial observability, variable timesteps, and explicit comms/observation
masks. Maps cleanly onto Gymnasium/PettingZoo (via :mod:`astro_mine.core.env.adapter`)
without being limited to them. Implemented by Sim; fed by Worlds/Prospect/Link; wrapped
by Learn; pinned by Bench.

The contract reuses the message catalog (RM-P0-CORE-04): a per-agent
:class:`~astro_mine.core.messages.Observation` (carrying its
:class:`~astro_mine.core.messages.CommsObservationMask`) is what ``step()`` returns, and an
:class:`~astro_mine.core.messages.ActionBatch` is what it consumes.

Core defines only the *shape* — no physics, no mask enforcement, no world loading, no
recording (those are Sim's). :func:`check_environment` is the consumer-driven contract
test an implementor runs in its own CI.

Public API:

- the contract — :class:`Environment` (multi-agent Protocol);
- the returns — :class:`ResetResult`, :class:`StepResult` (and :data:`AgentId` /
  :data:`Info`);
- Gym/PettingZoo mapping — :func:`as_gymnasium_reset` / :func:`as_gymnasium_step` /
  :func:`as_pettingzoo_reset` / :func:`as_pettingzoo_step`;
- conformance — :func:`check_environment` and :class:`EnvironmentContractError`.

Backlog: RM-P0-CORE-02 — astro-mine-core#2
"""

from __future__ import annotations

from astro_mine.core.env import adapter, conformance, model, protocol
from astro_mine.core.env.adapter import (
    as_gymnasium_reset,
    as_gymnasium_step,
    as_pettingzoo_reset,
    as_pettingzoo_step,
)
from astro_mine.core.env.conformance import (
    ActionSource,
    EnvironmentContractError,
    check_environment,
)
from astro_mine.core.env.model import AgentId, Info, ResetResult, StepResult
from astro_mine.core.env.protocol import Environment

__all__ = [
    "ActionSource",
    "AgentId",
    "Environment",
    "EnvironmentContractError",
    "Info",
    "ResetResult",
    "StepResult",
    "adapter",
    "as_gymnasium_reset",
    "as_gymnasium_step",
    "as_pettingzoo_reset",
    "as_pettingzoo_step",
    "check_environment",
    "conformance",
    "model",
    "protocol",
]
