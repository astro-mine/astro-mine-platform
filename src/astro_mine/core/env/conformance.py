"""Environment API contract-test utility (RM-P0-CORE-02).

The consumer-driven conformance check an implementor (Sim) runs in its own CI to prove it honors the
Core Environment contract — analogous to :func:`astro_mine.core.compat.assert_core_compatible`. It
drives a **seeded multi-step rollout** on a candidate environment and asserts the contract: the
multi-agent reset/step shapes, per-agent observations that are the canonical
:class:`~astro_mine.core.messages.Observation`, agent-consistent step maps, **monotonic agent
attrition** (terminations/truncations only shrink the active set; a departed agent never reappears),
**full-trace determinism** under a fixed seed (two rollouts hash identically — catching step-level
and multi-step non-determinism a single reset/step cannot), and a clean mapping onto the
Gymnasium/PettingZoo tuples. Raises :class:`EnvironmentContractError` on any violation.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from typing import Any

from astro_mine.core.env.adapter import as_gymnasium_step, as_pettingzoo_step
from astro_mine.core.env.model import AgentId, ResetResult, StepResult
from astro_mine.core.env.protocol import Environment
from astro_mine.core.messages.model import ActionBatch, Observation

__all__ = ["ActionSource", "EnvironmentContractError", "check_environment"]

#: An optional action source for the conformance rollout: a *pure* function of the step index and
#: the active agents, returning the ActionBatch to apply. It MUST be deterministic so the two
#: same-seed rollouts receive identical actions (default: the empty ActionBatch each step).
ActionSource = Callable[[int, tuple[AgentId, ...]], ActionBatch]

_Frame = dict[str, Any]


class EnvironmentContractError(AssertionError):
    """Raised when an environment violates the Core Environment API contract."""


def check_environment(
    env: Environment,
    *,
    seed: int = 0,
    steps: int = 8,
    action_fn: ActionSource | None = None,
) -> None:
    """Assert ``env`` honors the Core Environment API v0.1 contract.

    Drives a seeded multi-step rollout and checks the reset/step shapes, that observations are
    agent-facing :class:`~astro_mine.core.messages.Observation`\\ s, that per-agent step maps are
    agent-consistent, that agent attrition is monotonic (the active set only shrinks and a departed
    agent never reappears), and that the Gymnasium/PettingZoo adapters apply cleanly. Determinism is
    enforced by a **full-trace hash**: two rollouts under the same ``seed`` (and ``action_fn``) MUST
    produce byte-identical traces. Returns ``None`` on success.

    ``steps`` bounds the rollout (it also stops once no agents remain active); ``action_fn`` is an
    optional deterministic action source (default: the empty :class:`ActionBatch` each step).
    """
    if not isinstance(env, Environment):
        raise EnvironmentContractError(
            "object does not satisfy the Environment protocol (missing possible_agents/agents/"
            "reset/step)"
        )

    first = _rollout(env, seed=seed, steps=steps, action_fn=action_fn, validate=True)
    second = _rollout(env, seed=seed, steps=steps, action_fn=action_fn, validate=False)
    if _digest(first) != _digest(second):
        raise EnvironmentContractError(
            f"non-deterministic rollout: two same-seed rollouts diverged "
            f"{_divergence(first, second)} — the environment is not reproducible under a fixed seed"
        )


def _rollout(
    env: Environment,
    *,
    seed: int,
    steps: int,
    action_fn: ActionSource | None,
    validate: bool,
) -> list[_Frame]:
    possible = set(env.possible_agents)
    reset = env.reset(seed=seed)
    if validate:
        _check_reset(reset, possible)
        _check_attrition(tuple(env.agents), possible, previous=None, where="reset")
    trace: list[_Frame] = [_reset_frame(reset, env.agents)]
    departed: set[AgentId] = set()
    previous = set(env.agents)
    for index in range(steps):
        if not env.agents:
            break
        actions = action_fn(index, tuple(env.agents)) if action_fn is not None else ActionBatch()
        result = env.step(actions)
        active = tuple(env.agents)
        if validate:
            _check_step(result, possible, departed, where=f"step {index}")
            _check_attrition(active, possible, previous=previous, where=f"step {index}")
        departed |= previous - set(active)
        previous = set(active)
        trace.append(_step_frame(result, active))
    return trace


def _check_reset(result: ResetResult, possible: set[AgentId]) -> None:
    if not isinstance(result, ResetResult):
        raise EnvironmentContractError(
            f"reset() must return a ResetResult, got {type(result).__name__}"
        )
    if not result.observations:
        raise EnvironmentContractError("reset() returned no observations")
    _check_observations(result.observations, possible, where="reset")


def _check_step(
    result: StepResult, possible: set[AgentId], departed: set[AgentId], *, where: str
) -> None:
    if not isinstance(result, StepResult):
        raise EnvironmentContractError(
            f"{where}: step() must return a StepResult, got {type(result).__name__}"
        )
    obs_agents = set(result.observations)
    for name, mapping in (
        ("rewards", result.rewards),
        ("terminations", result.terminations),
        ("truncations", result.truncations),
        ("infos", result.infos),
    ):
        extra = set(mapping) - obs_agents
        if extra:
            raise EnvironmentContractError(
                f"{where}: {name} references agents with no observation this tick: {sorted(extra)}"
            )
    _check_observations(result.observations, possible, where=where)
    resurrected = obs_agents & departed
    if resurrected:
        raise EnvironmentContractError(
            f"{where}: departed agents observed again (attrition is final): {sorted(resurrected)}"
        )

    # The contract must map cleanly onto Gymnasium / PettingZoo (must not raise).
    as_pettingzoo_step(result)
    if obs_agents:
        as_gymnasium_step(result, next(iter(result.observations)))


def _check_observations(
    observations: Mapping[AgentId, Observation], possible: set[AgentId], *, where: str
) -> None:
    for agent, obs in observations.items():
        if not isinstance(obs, Observation):
            raise EnvironmentContractError(
                f"{where}: agent {agent!r} observation must be a messages.Observation, "
                f"got {type(obs).__name__}"
            )
    ghost = set(observations) - possible
    if ghost:
        raise EnvironmentContractError(
            f"{where}: observations for agents not in possible_agents: {sorted(ghost)}"
        )


def _check_attrition(
    active: tuple[AgentId, ...],
    possible: set[AgentId],
    *,
    previous: set[AgentId] | None,
    where: str,
) -> None:
    active_set = set(active)
    ghost = active_set - possible
    if ghost:
        raise EnvironmentContractError(
            f"{where}: active agents not in possible_agents: {sorted(ghost)}"
        )
    if previous is not None:
        grew = active_set - previous
        if grew:
            raise EnvironmentContractError(
                f"{where}: active agents grew (attrition must be monotonic): {sorted(grew)}"
            )


def _reset_frame(result: ResetResult, active: tuple[AgentId, ...]) -> _Frame:
    return {"kind": "reset", "agents": sorted(active), "observations": _dump(result.observations)}


def _step_frame(result: StepResult, active: tuple[AgentId, ...]) -> _Frame:
    return {
        "kind": "step",
        "agents": sorted(active),
        "observations": _dump(result.observations),
        "rewards": dict(result.rewards),
        "terminations": dict(result.terminations),
        "truncations": dict(result.truncations),
    }


def _dump(observations: Mapping[AgentId, Observation]) -> dict[str, Any]:
    # Compare by serialized message content (determinism), not object identity.
    return {agent: obs.model_dump(mode="json") for agent, obs in observations.items()}


def _digest(trace: list[_Frame]) -> str:
    canonical = json.dumps(trace, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _divergence(first: list[_Frame], second: list[_Frame]) -> str:
    if len(first) != len(second):
        return f"in length ({len(first)} vs {len(second)} frames)"
    index = next(i for i, (a, b) in enumerate(zip(first, second, strict=True)) if a != b)
    return f"at frame {index} ({first[index]['kind']})"
