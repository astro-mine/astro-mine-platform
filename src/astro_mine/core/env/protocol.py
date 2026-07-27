"""Environment API v0.1 — the contract (RM-P0-CORE-02).

``reset()/step(action) -> observation, reward?, info`` generalized for multi-agent
operation, partial observability, variable timestep, and explicit comms/observation
masks (core.md §3). Implemented by Sim (the reference stepping core), wrapped as
Gymnasium/PettingZoo by Learn, and pinned by Bench. It maps cleanly onto Gymnasium
(single-agent) / PettingZoo (multi-agent) via :mod:`astro_mine.core.env.adapter`
without being limited to them.

Core defines only the *shape*: it neither steps physics, enforces masks, loads worlds,
nor records traces — those are the implementor's (Sim's) job. The agent-facing surface
deliberately exposes **no** ground-truth accessor (the isolation boundary Prospect
enforces); a resource observation is a noisy reading, never a point ground-truth guess.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from astro_mine.core.env.model import AgentId, ResetResult, StepResult
from astro_mine.core.messages.model import ActionBatch

__all__ = ["Environment"]


@runtime_checkable
class Environment(Protocol):
    """The Core Environment contract Sim implements and Learn wraps.

    Multi-agent by construction; a single-agent env is the one-agent case (view it as a
    Gymnasium env with :func:`astro_mine.core.env.adapter.as_gymnasium_step`).
    Determinism is contractual: the same ``seed`` plus pinned inputs MUST yield identical
    observations — asserted by :func:`astro_mine.core.env.check_environment`.
    """

    @property
    def possible_agents(self) -> tuple[AgentId, ...]:
        """Every agent id that may appear in this environment."""
        ...

    @property
    def agents(self) -> tuple[AgentId, ...]:
        """Agent ids currently active.

        Attrition is **monotonic**: the active set only shrinks as agents terminate/truncate, and a
        departed agent does not reappear (in the active set or in later observations) — asserted by
        :func:`astro_mine.core.env.check_environment`.
        """
        ...

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> ResetResult:
        """Reset the episode and return the initial per-agent observations."""
        ...

    def step(self, actions: ActionBatch) -> StepResult:
        """Advance one tick given a per-agent action batch and return the outcome."""
        ...
