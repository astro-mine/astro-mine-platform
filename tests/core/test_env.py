"""Tests for ``astro_mine.core.env`` — the Environment API contract, a trivial reference
env, the Gym/PettingZoo adapters, and the conformance utility (RM-P0-CORE-02)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from astro_mine.core.env import (
    Environment,
    EnvironmentContractError,
    ResetResult,
    StepResult,
    as_gymnasium_reset,
    as_gymnasium_step,
    as_pettingzoo_reset,
    as_pettingzoo_step,
    check_environment,
)
from astro_mine.core.messages import hotpath
from astro_mine.core.messages.enums import ActionKind
from astro_mine.core.messages.model import (
    Action,
    ActionBatch,
    CommsObservationMask,
    ModeCommand,
    Observation,
    PeerLink,
    Quat,
    StateSample,
    Transform,
    Vec3,
)
from astro_mine.core.units import MOON_BODY_FIXED


def _obs(
    agent: str, peers: list[str], tick: int, t: float, *, observable: bool = True
) -> Observation:
    """A minimal but complete per-agent observation carrying a comms mask."""
    return Observation(
        tick=tick,
        sim_time_s=t,
        agent_id=agent,
        observable=observable,
        self_state=StateSample(
            agent_id=agent,
            frame=MOON_BODY_FIXED,
            pose=Transform(
                translation_m=Vec3(x=0.0, y=0.0, z=0.0),
                rotation_quat_xyzw=Quat(x=0.0, y=0.0, z=0.0, w=1.0),
            ),
        ),
        comms=CommsObservationMask(
            agent_id=agent,
            links=[PeerLink(peer=p, reachable=False) for p in peers],
            earth_contact=False,
        ),
    )


class TrivialEnv:
    """A minimal multi-agent Environment: agents masked from one another, advancing a
    fixed ``dt``, truncating at a horizon. Deterministic; ignores action content."""

    def __init__(self, agents: tuple[str, ...] = ("rover_a", "rover_b"), dt_s: float = 1.0) -> None:
        self._possible = agents
        self._agents = agents
        self._dt = dt_s
        self._tick = 0
        self._t = 0.0

    @property
    def possible_agents(self) -> tuple[str, ...]:
        return self._possible

    @property
    def agents(self) -> tuple[str, ...]:
        return self._agents

    def _peers(self, agent: str) -> list[str]:
        return [p for p in self._possible if p != agent]

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> ResetResult:
        self._tick = 0
        self._t = 0.0
        self._agents = self._possible
        observations = {a: _obs(a, self._peers(a), 0, 0.0) for a in self._agents}
        return ResetResult(
            observations=observations, infos={a: {"seed": seed} for a in self._agents}
        )

    def step(self, actions: ActionBatch) -> StepResult:
        self._tick += 1
        self._t += self._dt
        observations = {a: _obs(a, self._peers(a), self._tick, self._t) for a in self._agents}
        return StepResult(
            observations=observations,
            sim_time_s=self._t,
            rewards={},  # reward-free by default
            terminations={a: False for a in self._agents},
            truncations={a: self._tick >= 3 for a in self._agents},
            infos={a: {} for a in self._agents},
            dt_s=self._dt,
        )


# --- contract & conformance -------------------------------------------------------


def test_trivial_env_satisfies_protocol() -> None:
    assert isinstance(TrivialEnv(), Environment)


def test_check_environment_passes() -> None:
    assert check_environment(TrivialEnv()) is None


def test_check_environment_rejects_non_env() -> None:
    with pytest.raises(EnvironmentContractError):
        check_environment(object())  # type: ignore[arg-type]


class _BadResetType(TrivialEnv):
    def reset(self, *, seed: int | None = None, options: Mapping[str, Any] | None = None) -> Any:
        return {}


class _EmptyReset(TrivialEnv):
    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> ResetResult:
        return ResetResult(observations={})


class _BadResetObs(TrivialEnv):
    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> ResetResult:
        return ResetResult(observations={"rover_a": "not-an-observation"})  # type: ignore[dict-item]


class _NonDeterministic(TrivialEnv):
    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> ResetResult:
        self._tick += 1
        return ResetResult(
            observations={"rover_a": _obs("rover_a", [], self._tick, float(self._tick))}
        )


class _BadStepType(TrivialEnv):
    def step(self, actions: ActionBatch) -> Any:
        return {}


class _ExtraAgentStep(TrivialEnv):
    def step(self, actions: ActionBatch) -> StepResult:
        base = super().step(actions)
        return StepResult(
            observations=base.observations, sim_time_s=base.sim_time_s, rewards={"ghost": 1.0}
        )


class _BadStepObs(TrivialEnv):
    def step(self, actions: ActionBatch) -> StepResult:
        return StepResult(observations={"rover_a": "nope"}, sim_time_s=1.0)  # type: ignore[dict-item]


class _ObsGhostAgent(TrivialEnv):
    """Emits an observation for an agent not in ``possible_agents``."""

    def step(self, actions: ActionBatch) -> StepResult:
        base = super().step(actions)
        observations = {**base.observations, "ghost": _obs("ghost", [], self._tick, self._t)}
        return StepResult(observations=observations, sim_time_s=base.sim_time_s)


class _ActiveGhost(TrivialEnv):
    """Reports an active agent that is not in ``possible_agents``."""

    def step(self, actions: ActionBatch) -> StepResult:
        base = super().step(actions)
        self._agents = (*self._agents, "ghost")
        return base


class _StepNonDeterministic(TrivialEnv):
    """Valid shapes, but a later step depends on the run count -> two same-seed rollouts diverge."""

    def __init__(self) -> None:
        super().__init__()
        self._runs = 0

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> ResetResult:
        self._runs += 1
        return super().reset(seed=seed, options=options)

    def step(self, actions: ActionBatch) -> StepResult:
        base = super().step(actions)
        if self._tick != 2:
            return base
        observations = {a: _obs(a, [], self._tick + self._runs, self._t) for a in self._agents}
        return StepResult(observations=observations, sim_time_s=base.sim_time_s)


class _VaryingLength(TrivialEnv):
    """The rollout length depends on the run count -> two same-seed rollouts differ in length."""

    def __init__(self) -> None:
        super().__init__()
        self._runs = 0

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> ResetResult:
        self._runs += 1
        return super().reset(seed=seed, options=options)

    def step(self, actions: ActionBatch) -> StepResult:
        base = super().step(actions)
        if self._tick >= self._runs:
            self._agents = ()
        return base


@pytest.mark.parametrize(
    "bad",
    [
        _BadResetType,
        _EmptyReset,
        _BadResetObs,
        _NonDeterministic,
        _BadStepType,
        _ExtraAgentStep,
        _BadStepObs,
        _ObsGhostAgent,
        _ActiveGhost,
        _StepNonDeterministic,
        _VaryingLength,
    ],
)
def test_check_environment_rejects_violations(bad: type[TrivialEnv]) -> None:
    with pytest.raises(EnvironmentContractError):
        check_environment(bad())


# --- agent attrition (Core #20) ---------------------------------------------------


class AttritionEnv:
    """A multi-agent env with real attrition: agent ``c`` terminates at tick 2, then is removed."""

    def __init__(self) -> None:
        self._possible = ("a", "b", "c")
        self._active: tuple[str, ...] = self._possible
        self._tick = 0

    @property
    def possible_agents(self) -> tuple[str, ...]:
        return self._possible

    @property
    def agents(self) -> tuple[str, ...]:
        return self._active

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> ResetResult:
        self._tick = 0
        self._active = self._possible
        return ResetResult(observations={a: _obs(a, [], 0, 0.0) for a in self._active})

    def step(self, actions: ActionBatch) -> StepResult:
        self._tick += 1
        t = float(self._tick)
        terminating = "c" in self._active and self._tick == 2
        observations = {a: _obs(a, [], self._tick, t) for a in self._active}
        result = StepResult(
            observations=observations,
            sim_time_s=t,
            terminations={a: (a == "c" and terminating) for a in self._active},
            truncations={a: False for a in self._active},
        )
        if terminating:
            self._active = tuple(a for a in self._active if a != "c")
        return result


class _RegrowsAgents(AttritionEnv):
    """Re-adds the departed agent to the active set -> non-monotonic attrition."""

    def step(self, actions: ActionBatch) -> StepResult:
        result = super().step(actions)
        if self._tick == 4:
            self._active = self._possible
        return result


class _ResurrectsInObs(AttritionEnv):
    """Re-emits an observation for the departed agent -> attrition is not final."""

    def step(self, actions: ActionBatch) -> StepResult:
        result = super().step(actions)
        if self._tick != 4:
            return result
        revived = _obs("c", [], self._tick, float(self._tick))
        observations = {**result.observations, "c": revived}
        return StepResult(observations=observations, sim_time_s=result.sim_time_s)


def test_check_environment_accepts_attrition() -> None:
    assert check_environment(AttritionEnv()) is None


def test_check_environment_rejects_agent_regrowth() -> None:
    with pytest.raises(EnvironmentContractError, match="grew"):
        check_environment(_RegrowsAgents())


def test_check_environment_rejects_resurrected_observation() -> None:
    with pytest.raises(EnvironmentContractError, match="departed agents observed again"):
        check_environment(_ResurrectsInObs())


def test_check_environment_accepts_deterministic_action_fn() -> None:
    def actions(index: int, agents: tuple[str, ...]) -> ActionBatch:
        return ActionBatch(
            actions=[
                Action(agent_id=a, kind=ActionKind.MODE, mode=ModeCommand(mode="idle"))
                for a in agents
            ]
        )

    assert check_environment(TrivialEnv(), action_fn=actions) is None


# --- reset / step shape -----------------------------------------------------------


def test_reset_returns_per_agent_observations_with_masks() -> None:
    result = TrivialEnv().reset(seed=7)
    assert set(result.observations) == {"rover_a", "rover_b"}
    for agent, obs in result.observations.items():
        assert isinstance(obs, Observation)
        assert obs.comms is not None and obs.comms.agent_id == agent
    assert result.infos["rover_a"]["seed"] == 7


def test_step_variable_timestep_surfaced() -> None:
    env = TrivialEnv(dt_s=2.5)
    env.reset(seed=0)
    r1 = env.step(ActionBatch())
    assert r1.dt_s == 2.5 and r1.sim_time_s == 2.5
    assert env.step(ActionBatch()).sim_time_s == 5.0


def test_step_rewards_empty_by_default() -> None:
    env = TrivialEnv()
    env.reset(seed=0)
    assert env.step(ActionBatch()).rewards == {}


def test_truncation_at_horizon() -> None:
    env = TrivialEnv()
    env.reset(seed=0)
    result = StepResult(observations={}, sim_time_s=0.0)
    for _ in range(3):
        result = env.step(ActionBatch())
    assert all(result.truncations.values())
    assert not any(result.terminations.values())


def test_step_consumes_action_batch() -> None:
    env = TrivialEnv()
    env.reset(seed=0)
    batch = ActionBatch(
        actions=[Action(agent_id="rover_a", kind=ActionKind.MODE, mode=ModeCommand(mode="idle"))]
    )
    assert isinstance(env.step(batch), StepResult)


def test_reset_is_deterministic() -> None:
    env = TrivialEnv()
    a = {k: v.model_dump(mode="json") for k, v in env.reset(seed=0).observations.items()}
    b = {k: v.model_dump(mode="json") for k, v in env.reset(seed=0).observations.items()}
    assert a == b


# --- comms masks round-trip through the message schemas (AC #2) --------------------


def test_comms_masks_round_trip_through_messages() -> None:
    env = TrivialEnv()
    env.reset(seed=0)
    obs = env.step(ActionBatch()).observations["rover_a"]
    assert obs.comms is not None
    restored = hotpath.from_bytes(hotpath.to_bytes(obs))
    assert restored.comms == obs.comms
    assert restored == obs


# --- Gymnasium / PettingZoo adapters ----------------------------------------------


def test_pettingzoo_reset_and_step_mapping() -> None:
    env = TrivialEnv()
    reset = env.reset(seed=0)
    pz_obs, pz_infos = as_pettingzoo_reset(reset)
    assert set(pz_obs) == set(env.possible_agents) == set(pz_infos)

    step = env.step(ActionBatch())
    observations, rewards, terminations, truncations, infos = as_pettingzoo_step(step)
    agents = set(env.possible_agents)
    assert set(observations) == agents
    assert rewards == {a: 0.0 for a in agents}  # reward-free filled to 0.0
    assert set(terminations) == agents and set(truncations) == agents and set(infos) == agents


def test_gymnasium_single_agent_view() -> None:
    env = TrivialEnv(agents=("solo",))
    obs, info = as_gymnasium_reset(env.reset(seed=0), "solo")
    assert isinstance(obs, Observation) and isinstance(info, Mapping)

    o, reward, terminated, truncated, i = as_gymnasium_step(env.step(ActionBatch()), "solo")
    assert isinstance(o, Observation)
    assert reward == 0.0 and terminated is False and truncated is False
    assert isinstance(i, Mapping)


def test_gymnasium_adapter_missing_agent_raises() -> None:
    env = TrivialEnv()
    env.reset(seed=0)
    step = env.step(ActionBatch())
    with pytest.raises(KeyError):
        as_gymnasium_step(step, "nonexistent")
    with pytest.raises(KeyError):
        as_gymnasium_reset(env.reset(seed=0), "nonexistent")


# --- RFC-0001 reserved regime dimension (RM-P1-CORE-04) --------------------------


def test_reset_and_step_results_reserve_the_regime_dimension() -> None:
    from astro_mine.core.sadf.enums import Regime

    obs = {"a": _obs("a", [], 0, 0.0)}
    # Default: unset — a single-regime (surface) run behaves exactly as before.
    assert ResetResult(observations=obs).regime is None
    assert StepResult(observations=obs, sim_time_s=1.0).regime is None
    # The reserved bounded regime dimension accepts a Regime (mission-model.md §2.2).
    reset = ResetResult(observations=obs, regime=Regime.SURFACE)
    step = StepResult(observations=obs, sim_time_s=1.0, regime=Regime.PROXIMITY_ORBIT)
    assert reset.regime is Regime.SURFACE
    assert step.regime is Regime.PROXIMITY_ORBIT
