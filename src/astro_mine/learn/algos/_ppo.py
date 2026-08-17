# SPDX-License-Identifier: Apache-2.0
"""Shared PPO trainer for IPPO / MAPPO / comms-learning PPO — RM-P1-LEARN-03.

One reproducible on-policy PPO loop parameterised by two flags: ``centralized`` and ``comms``.

- **IPPO** (``centralized=False``): each heterogeneous agent owns a
  :class:`~astro_mine.learn.models.mlp.DictActorCritic`; advantages use that agent's *own*
  reward and *local* value head — no shared information (the simple control).
- **MAPPO** (``centralized=True``): decentralised actors, one shared
  :class:`~astro_mine.learn.models.critics.CentralizedCritic` over the global
  ``SwarmEnv.state()``, and a **team** (summed) reward — the honest CTDE bargain declared by
  a :class:`~astro_mine.learn.algos._contract.CentralizedCriticSpec`.
- **comms_ppo** (``centralized=True, comms=True``): MAPPO **plus a learned message channel**
  (:class:`~astro_mine.learn.models.comms.CommsEncoder`). Each agent encodes a message from its
  own observation; every actor conditions on the mean-pool of the messages from the peers the
  :class:`~astro_mine.learn.envs.CommsModel` actually **delivered** this tick (gate → budget →
  drop → delay), read off the executor's recorded reachability — never a side channel. The
  aggregate is **recomputed inside the update** from that same recorded reachability, so the
  message encoder gets real gradients from the team objective: a *differentiable-message* CTDE
  learner, trained under the identical comms constraint the comms-blind baselines are scored on
  (learn.md §11 "comms-learning as a first-class research track"; charter §8).

Experience is collected through the :class:`~astro_mine.learn.train.executor.RolloutExecutor`
seam so RM-P1-LEARN-04 can swap in a distributed or GPU-vectorized executor unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from astro_mine.core.env.model import AgentId
from astro_mine.core.policy import Policy
from astro_mine.learn.algos._contract import (
    AlgorithmSpec,
    CentralizedCriticSpec,
    IoSignature,
    PolicyAssumptions,
    PolicyExport,
)
from astro_mine.learn.algos._torch_common import (
    MESSAGE_DIM,
    MESSAGE_FEAT_DIM,
    compute_gae,
    fidelity_caveats,
    make_generator,
    provenance,
    seed_everything,
)
from astro_mine.learn.algos.config import TrainConfig
from astro_mine.learn.algos.policy import (
    InferFn,
    LearnedPolicy,
    action_heads,
    agent_io_signature,
    flat_obs_dim,
)
from astro_mine.learn.envs import SwarmEnv
from astro_mine.learn.models.comms import CommsEncoder, reach_matrix
from astro_mine.learn.models.critics import CentralizedCritic
from astro_mine.learn.models.mlp import DictActorCritic
from astro_mine.learn.train.executor import (
    AgentStepFn,
    LocalExecutor,
    RewardFn,
    Rollout,
    RolloutExecutor,
    RolloutStep,
    StepDecision,
    default_reward_fn,
)

__all__ = ["PpoTrainer"]


@dataclass
class _AgentBatch:
    obs: torch.Tensor
    actions: dict[str, torch.Tensor]
    old_log_prob: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


def _build_comms(arch: Mapping[AgentId, dict[str, Any]]) -> CommsEncoder | None:
    """The shared learned-message channel for a comms-learning run (``None`` otherwise).

    Built from the same serialized per-agent architecture the actors are, so the learner and a
    rebuilt KubeRay worker agree on the projection widths exactly."""
    if not any(agent_arch.get("comms_dim", 0) for agent_arch in arch.values()):
        return None
    return CommsEncoder(
        {agent: agent_arch["obs_dim"] for agent, agent_arch in arch.items()},
        feat_dim=MESSAGE_FEAT_DIM,
        msg_dim=MESSAGE_DIM,
    )


class _PpoStep:
    """The PPO sampling step as a :class:`~astro_mine.learn.train.executor.BroadcastableStep`.

    Callable exactly like the in-process step (sample each live agent's action from its
    :class:`~astro_mine.learn.models.mlp.DictActorCritic` with the shared seeded generator),
    and additionally:

    - a :class:`~astro_mine.learn.train.executor.BroadcastableStep` — :meth:`broadcast` returns
      a picklable ``(rebuild, state)`` pair (per-agent ``state_dict`` + net architecture + the
      shared comms encoder + the generator state) so a KubeRay worker reconstructs a
      byte-equivalent step, and :meth:`rng_state`/:meth:`set_rng_state` keep the learner RNG in
      lockstep;
    - a :class:`~astro_mine.learn.train.executor.CommsAwareStep` when the comms-learning channel
      is on — :meth:`observe_reach` takes the executor's per-tick reachability verdict (the
      CommsModel's post-gate/drop/delay result) and the messages aggregate over exactly the
      peers that got through;
    - a :class:`~astro_mine.learn.train.executor.BatchedStep` — :meth:`batch` decides for a whole
      batch of vectorized env copies in **one** forward per agent (the GPU-vectorized kernel).

    Holds the trainer's *live* nets + comms encoder + generator, so it always samples from
    current weights.
    """

    def __init__(
        self,
        nets: Mapping[AgentId, DictActorCritic],
        generator: torch.Generator,
        arch: Mapping[AgentId, dict[str, Any]],
        comms: CommsEncoder | None = None,
    ) -> None:
        self._nets = dict(nets)
        self._generator = generator
        self._arch = dict(arch)
        self._comms = comms
        self._agents: tuple[AgentId, ...] = tuple(arch)
        #: The most recent per-tick reachability verdict handed over by the executor (empty
        #: before the first tick — every agent is isolated, so every message aggregate is zero).
        self._reach: dict[AgentId, tuple[AgentId, ...]] = {}

    # --- CommsAwareStep -------------------------------------------------------------

    def observe_reach(self, reach: Mapping[AgentId, tuple[AgentId, ...]]) -> None:
        self._reach = dict(reach)

    def _messages(
        self, stacked: Mapping[AgentId, NDArray[np.float32]], reach: NDArray[np.float32] | None
    ) -> dict[AgentId, torch.Tensor]:
        """The aggregated peer-message context per agent for a ``(batch, obs_dim)`` obs stack.

        Empty (and free) for a comms-blind baseline. Rollout-time only — **no gradient**; the
        update recomputes the aggregate differentiably from the *recorded* reachability, which
        is what actually trains the message encoder (:meth:`PpoTrainer._comms_context`).
        ``reach=None`` means "nothing was delivered": every agent is isolated and gets the zero
        message."""
        if self._comms is None:
            return {}
        rows = int(next(iter(stacked.values())).shape[0])
        n = len(self._agents)
        mask = (
            np.asarray(reach, dtype=np.float32)
            if reach is not None
            else np.zeros((rows, n, n), dtype=np.float32)
        )
        obs = {
            agent: torch.from_numpy(np.asarray(row, dtype=np.float32))
            for agent, row in stacked.items()
        }
        with torch.no_grad():
            return self._comms(obs, torch.from_numpy(mask))

    def _decide(
        self,
        stacked: Mapping[AgentId, NDArray[np.float32]],
        messages: Mapping[AgentId, torch.Tensor],
    ) -> dict[AgentId, tuple[list[dict[str, object]], list[float], list[float]]]:
        """One batched forward per agent over a ``(rows, obs_dim)`` observation stack."""
        out: dict[AgentId, tuple[list[dict[str, object]], list[float], list[float]]] = {}
        for agent, obs in stacked.items():
            net = self._nets[agent]
            trunk = net.trunk_input(
                torch.from_numpy(np.asarray(obs, dtype=np.float32)), messages.get(agent)
            )
            actions, log_probs, values, _ = net.act_batch(trunk, self._generator)
            out[agent] = (actions, log_probs.tolist(), values.tolist())
        return out

    # --- AgentStepFn ----------------------------------------------------------------

    def __call__(self, flat_obs: Mapping[AgentId, np.ndarray]) -> Mapping[AgentId, StepDecision]:
        stacked = {
            agent: np.asarray(obs, dtype=np.float32).reshape(1, -1)
            for agent, obs in flat_obs.items()
        }
        mask = (
            reach_matrix(self._agents, self._reach, live=list(flat_obs))[np.newaxis, ...]
            if self._comms is not None
            else None
        )
        decided = self._decide(stacked, self._messages(stacked, mask))
        return {
            agent: StepDecision(action_sample=actions[0], log_prob=log_probs[0], value=values[0])
            for agent, (actions, log_probs, values) in decided.items()
        }

    # --- BatchedStep ----------------------------------------------------------------

    def batch(
        self,
        flat_obs: Mapping[AgentId, NDArray[np.float32]],
        reach: NDArray[np.float32] | None = None,
    ) -> Mapping[AgentId, list[StepDecision]]:
        """Decide for ``num_envs`` env copies in **one forward per agent** (the batched kernel).

        ``flat_obs`` stacks each agent's observation across the batch ``(num_envs, obs_dim)``;
        ``reach`` is the per-copy reachability mask ``(num_envs, n_agents, n_agents)`` (``None``
        ⇒ nothing delivered, the zero message). Draws from the same seeded generator in the same
        order as the sequential path, so a batch of one is byte-identical to it."""
        with torch.no_grad():
            decided = self._decide(flat_obs, self._messages(flat_obs, reach))
        return {
            agent: [
                StepDecision(action_sample=action, log_prob=lp, value=value)
                for action, lp, value in zip(actions, log_probs, values, strict=True)
            ]
            for agent, (actions, log_probs, values) in decided.items()
        }

    # --- BroadcastableStep ----------------------------------------------------------

    def broadcast(self) -> tuple[Callable[[Any], AgentStepFn], dict[str, Any]]:
        state: dict[str, Any] = {
            "arch": self._arch,
            "weights": {agent: net.state_dict() for agent, net in self._nets.items()},
            "comms_weights": None if self._comms is None else self._comms.state_dict(),
            "generator_state": self._generator.get_state(),
        }
        return _rebuild_ppo_step, state

    def rng_state(self) -> Any:
        return self._generator.get_state()

    def set_rng_state(self, state: Any) -> None:
        self._generator.set_state(state)


def _rebuild_ppo_step(state: dict[str, Any]) -> _PpoStep:
    """Reconstruct a :class:`_PpoStep` on a rollout worker from a broadcast ``state`` (a
    module-level function so it is picklable by reference for Ray)."""
    generator = torch.Generator()
    generator.set_state(state["generator_state"])
    nets: dict[AgentId, DictActorCritic] = {}
    for agent, arch in state["arch"].items():
        net = DictActorCritic(
            arch["obs_dim"],
            arch["discrete_heads"],
            arch["box_heads"],
            arch["hidden_sizes"],
            use_rnn=arch["use_rnn"],
            comms_dim=arch.get("comms_dim", 0),
        )
        net.load_state_dict(state["weights"][agent])
        nets[agent] = net
    comms = _build_comms(state["arch"])
    if comms is not None and state["comms_weights"] is not None:
        comms.load_state_dict(state["comms_weights"])
    return _PpoStep(nets, generator, state["arch"], comms)


class PpoTrainer:
    """A reproducible multi-agent PPO trainer (IPPO / MAPPO / comms-learning PPO).

    ``centralized`` selects the CTDE centralized critic; ``comms`` additionally attaches the
    learned message channel (:class:`~astro_mine.learn.models.comms.CommsEncoder`) that the
    ``comms_ppo`` baseline trains end-to-end over the CommsModel's delivered links."""

    def __init__(
        self,
        spec: AlgorithmSpec,
        env: SwarmEnv,
        config: TrainConfig,
        *,
        centralized: bool,
        comms: bool = False,
        reward_fn: RewardFn | None = None,
        executor: RolloutExecutor | None = None,
    ) -> None:
        seed_everything(config.seed)
        self._spec = spec
        self._env = env
        self._config = config
        self._centralized = centralized
        self._reward_fn: RewardFn = reward_fn if reward_fn is not None else default_reward_fn
        self._executor: RolloutExecutor = executor if executor is not None else LocalExecutor()
        self._generator = make_generator(config.seed)
        self._history: list[dict[str, float]] = []

        self._specs = env.agent_specs
        self._agents: tuple[AgentId, ...] = tuple(self._specs)
        comms_dim = MESSAGE_DIM if comms else 0
        self._nets: dict[AgentId, DictActorCritic] = {}
        self._arch: dict[AgentId, dict[str, Any]] = {}
        params: list[torch.nn.Parameter] = []
        for agent, aspec in self._specs.items():
            heads = action_heads(aspec.action_space)
            obs_dim = flat_obs_dim(aspec.observation_space)
            net = DictActorCritic(
                obs_dim,
                heads.discrete,
                heads.box,
                config.hidden_sizes,
                use_rnn=config.use_rnn,
                comms_dim=comms_dim,
            )
            self._nets[agent] = net
            params += list(net.parameters())
            # Serializable per-agent net architecture — the payload a KubeRay worker rebuilds
            # the actor from (broadcast alongside the state_dict) and the export path rebuilds
            # the ONNX graph from; see :class:`_PpoStep` and ``export/package.py``.
            self._arch[agent] = {
                "obs_dim": obs_dim,
                "discrete_heads": dict(heads.discrete),
                "box_heads": dict(heads.box),
                "hidden_sizes": tuple(config.hidden_sizes),
                "use_rnn": config.use_rnn,
                "comms_dim": comms_dim,
            }
        # The learned message channel is a *shared* module across the heterogeneous swarm (one
        # message vocabulary), trained jointly with the actors by the same optimizer.
        self._comms = _build_comms(self._arch)
        if self._comms is not None:
            params += list(self._comms.parameters())
        self._step = _PpoStep(self._nets, self._generator, self._arch, self._comms)

        self._state_dim = int(np.asarray(env.state()).shape[0])
        self._critic: CentralizedCritic | None = None
        self._critic_spec: CentralizedCriticSpec | None = None
        if centralized:
            self._critic_spec = CentralizedCriticSpec(
                global_state_dim=self._state_dim,
                per_agent_obs_dims={
                    a: flat_obs_dim(s.observation_space) for a, s in self._specs.items()
                },
            )
            self._critic = CentralizedCritic(self._critic_spec, config.hidden_sizes)
            params += list(self._critic.parameters())

        self._opt = torch.optim.Adam(params, lr=config.lr)

    # --- contract ------------------------------------------------------------------

    @property
    def spec(self) -> AlgorithmSpec:
        return self._spec

    @property
    def centralized_critic(self) -> CentralizedCriticSpec | None:
        return self._critic_spec

    @property
    def rollout_step(self) -> _PpoStep:
        """The live rollout step the executor drives — a ``BroadcastableStep``, a
        ``CommsAwareStep`` (when comms-learning), and a ``BatchedStep``. Exposed so a tool can
        drive the *real* policy without a training loop (the throughput benchmark does)."""
        return self._step

    def set_env(self, env: SwarmEnv) -> None:
        """Swap the environment the next rollout collects from — the **curriculum seam**.

        A curriculum advances the *world* (a harder comms regime, a re-sampled domain
        randomization, a different fidelity tier), never the learner: the nets, optimizer, and
        seeded generator carry over untouched, so learning *continues* across a stage promotion
        instead of restarting. The new env must declare the same agents and the same global-state
        width — a curriculum changes the difficulty, not the tensor contract."""
        _assert_same_shape(self._specs, self._state_dim, env)
        self._env = env

    def learning_curve(self) -> list[float]:
        return [h["mean_reward"] for h in self._history]

    def train_iteration(self) -> dict[str, float]:
        seed = self._config.seed + 1000 * (len(self._history) + 1)
        rollout = self._executor.rollout(
            self._env,
            self._step,
            steps=self._config.rollout_steps,
            seed=seed,
            reward_fn=self._reward_fn,
        )
        metrics = self._update(rollout)
        self._history.append(metrics)
        return metrics

    def policy(self) -> Policy:
        infer: dict[AgentId, InferFn] = {a: self._greedy_infer(a) for a in self._nets}
        return LearnedPolicy(self._specs, infer)

    def export(self) -> PolicyExport:
        io = IoSignature(
            agent_ids=tuple(self._specs),
            per_agent={a: agent_io_signature(s) for a, s in self._specs.items()},
            global_state_dim=self._state_dim if self._centralized else 0,
        )
        assumptions = PolicyAssumptions(
            comms_observability=self._env.comms_provenance(),
            partial_observability=True,
            surrogate_fidelity_caveats=fidelity_caveats(self._config),
        )
        weights: dict[str, object] = {a: net.state_dict() for a, net in self._nets.items()}
        if self._critic is not None:
            weights["__critic__"] = self._critic.state_dict()
        if self._comms is not None:
            # The shared message encoder stays *internal* to Learn (like the CTDE critic): the
            # exported per-agent actor graph takes the aggregated peer-message context as an
            # explicit `msg` input, so a host binds the aggregate — it does not re-derive it.
            weights["__comms__"] = self._comms.state_dict()
        return PolicyExport(
            algorithm=self._spec.name,
            backend="torch",
            io_signature=io,
            assumptions=assumptions,
            provenance=provenance(self._config, comms_provenance=self._env.comms_provenance()),
            weights=weights,
            metrics=self._history[-1] if self._history else {},
            net_kind="actor_critic",
            net_arch=self._arch,
        )

    # --- rollout + update ----------------------------------------------------------

    def _comms_context(self, rollout: Rollout) -> dict[AgentId, torch.Tensor]:
        """The **differentiable** per-agent peer-message context over the whole rollout.

        Re-runs the shared :class:`CommsEncoder` on every recorded step — over that step's *own*
        recorded reachability (``RolloutStep.reach``: precisely the peers the CommsModel
        delivered through gate → budget → drop → delay) — and stacks each agent's rows in the
        same order :meth:`_agent_batches` stacks its observations. Because this runs **inside**
        the update (not under ``no_grad`` at rollout time), the PPO loss backpropagates into the
        message encoder: the swarm learns *what to say* under the real channel, not just how to
        act on what it heard.

        Recomputed once per update epoch (the graph is consumed by each ``backward``)."""
        assert self._comms is not None
        rows: dict[AgentId, list[torch.Tensor]] = {agent: [] for agent in self._nets}
        for step in rollout.steps:
            mask = reach_matrix(self._agents, step.reach, live=list(step.obs))[np.newaxis, ...]
            obs = {
                agent: torch.from_numpy(np.asarray(step.obs[agent], dtype=np.float32)).unsqueeze(0)
                for agent in step.obs
            }
            aggregate = self._comms(obs, torch.from_numpy(mask))
            for agent in step.obs:
                rows[agent].append(aggregate[agent][0])
        return {agent: torch.stack(items) for agent, items in rows.items() if items}

    def _update(self, rollout: Rollout) -> dict[str, float]:
        cfg = self._config
        team_returns = self._team_returns(rollout) if self._centralized else None
        states = self._states_tensor(rollout) if self._centralized else None
        batches = self._agent_batches(rollout, team_advantages=self._team_advantages(rollout))

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        for _ in range(cfg.update_epochs):
            self._opt.zero_grad()
            total = torch.zeros(())
            # Fresh each epoch: the previous epoch's backward consumed this graph.
            messages = self._comms_context(rollout) if self._comms is not None else {}
            if self._centralized and states is not None and team_returns is not None:
                assert self._critic is not None
                v_state = self._critic(states)
                value_loss = 0.5 * ((v_state - team_returns) ** 2).mean()
                total = total + cfg.value_coef * value_loss
                value_losses.append(float(value_loss.item()))
            for agent, batch in batches.items():
                net = self._nets[agent]
                trunk = net.trunk_input(batch.obs, messages.get(agent))
                new_lp, entropy, v_local = net.evaluate(trunk, batch.actions)
                ratio = torch.exp(new_lp - batch.old_log_prob)
                surr1 = ratio * batch.advantages
                surr2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * batch.advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                ent = entropy.mean()
                loss = policy_loss - cfg.entropy_coef * ent
                if not self._centralized:
                    local_vloss = 0.5 * ((v_local - batch.returns) ** 2).mean()
                    loss = loss + cfg.value_coef * local_vloss
                    value_losses.append(float(local_vloss.item()))
                total = total + loss
                policy_losses.append(float(policy_loss.item()))
                entropies.append(float(ent.item()))
            total.backward()
            self._opt.step()

        rewards = [r for step in rollout.steps for r in step.reward.values()]
        metrics = {
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "policy_loss": float(np.mean(policy_losses)) if policy_losses else 0.0,
            "value_loss": float(np.mean(value_losses)) if value_losses else 0.0,
            "entropy": float(np.mean(entropies)) if entropies else 0.0,
            "env_steps": float(rollout.env_steps()),
        }
        if self._comms is not None:
            # The comms-stress denominator for the learned channel: the mean fraction of peers
            # whose message actually arrived. A comms-learning curve is only interpretable next
            # to the delivery rate it was learned under (learn.md §10).
            metrics["message_delivery_rate"] = _delivery_rate(rollout)
        return metrics

    # --- tensor assembly -----------------------------------------------------------

    def _agent_batches(
        self, rollout: Rollout, *, team_advantages: list[float] | None
    ) -> dict[AgentId, _AgentBatch]:
        batches: dict[AgentId, _AgentBatch] = {}
        for agent in self._nets:
            indices = [i for i, step in enumerate(rollout.steps) if agent in step.obs]
            if not indices:  # pragma: no cover - every conformant agent is live for ≥1 step
                continue
            steps = [rollout.steps[i] for i in indices]
            obs = torch.from_numpy(np.stack([s.obs[agent] for s in steps])).float()
            actions = self._action_tensors(agent, steps)
            old_lp = torch.tensor([s.log_prob[agent] for s in steps], dtype=torch.float32)
            if team_advantages is not None:
                adv_list = [team_advantages[i] for i in indices]
                ret_list = adv_list  # centralized value uses team returns; per-agent returns unused
            else:
                rewards = [s.reward[agent] for s in steps]
                values = [s.value[agent] for s in steps]
                dones = [s.done[agent] for s in steps]
                adv_list, ret_list = compute_gae(
                    rewards,
                    values,
                    dones,
                    gamma=self._config.gamma,
                    gae_lambda=self._config.gae_lambda,
                )
            batches[agent] = _AgentBatch(
                obs=obs,
                actions=actions,
                old_log_prob=old_lp,
                advantages=torch.tensor(adv_list, dtype=torch.float32),
                returns=torch.tensor(ret_list, dtype=torch.float32),
            )
        return batches

    def _action_tensors(self, agent: AgentId, steps: list[RolloutStep]) -> dict[str, torch.Tensor]:
        heads = action_heads(self._specs[agent].action_space)
        out: dict[str, torch.Tensor] = {}
        samples = [s.action[agent] for s in steps]
        for name in heads.discrete:
            out[name] = torch.tensor([int(sample[name]) for sample in samples], dtype=torch.long)
        for name in heads.box:
            out[name] = torch.from_numpy(
                np.stack([np.asarray(sample[name], dtype=np.float32) for sample in samples])
            ).float()
        return out

    def _team_advantages(self, rollout: Rollout) -> list[float] | None:
        if not self._centralized or self._critic is None:
            return None
        states = self._states_tensor(rollout)
        with torch.no_grad():
            values = self._critic(states).tolist()
        rewards = [sum(step.reward.values()) for step in rollout.steps]
        dones = [i == len(rollout.steps) - 1 for i in range(len(rollout.steps))]
        advantages, _ = compute_gae(
            rewards, values, dones, gamma=self._config.gamma, gae_lambda=self._config.gae_lambda
        )
        return advantages

    def _team_returns(self, rollout: Rollout) -> torch.Tensor:
        assert self._critic is not None  # only called on the centralized (MAPPO) path
        advantages = self._team_advantages(rollout) or []
        states = self._states_tensor(rollout)
        with torch.no_grad():
            values = self._critic(states).tolist()
        returns = [a + v for a, v in zip(advantages, values, strict=True)]
        return torch.tensor(returns, dtype=torch.float32)

    def _states_tensor(self, rollout: Rollout) -> torch.Tensor:
        return torch.from_numpy(np.stack([step.state for step in rollout.steps])).float()

    def _greedy_infer(self, agent: AgentId) -> InferFn:
        net = self._nets[agent]

        def infer(flat: np.ndarray) -> Mapping[str, object]:
            obs = torch.from_numpy(np.asarray(flat, dtype=np.float32)).unsqueeze(0)
            # A decentralized Core Policy decides from *one* agent's observation and has no
            # peers to aggregate, so a comms-learning actor sees the zero message here — the
            # honest isolated-agent case (MessageModule semantics), and exactly what the
            # exported ONNX graph's host feeds its `msg` input by default (export/host.py).
            return net.greedy(net.trunk_input(obs))

        return infer


def _assert_same_shape(specs: Mapping[AgentId, Any], state_dim: int, env: SwarmEnv) -> None:
    """Guard the curriculum env swap: the new stage's world must keep the tensor contract.

    A curriculum stage changes the *difficulty* (comms regime, domain randomization, fidelity
    tier) of the same scenario — so the agent set and the CTDE global-state width must be
    unchanged. A mismatch means the caller swapped in a *different world*, which would silently
    feed garbage into nets sized for the old one; fail loudly instead."""
    if tuple(env.agent_specs) != tuple(specs):
        raise ValueError(
            f"curriculum env swap changed the agent set: {tuple(env.agent_specs)} != {tuple(specs)}"
        )
    new_state_dim = int(np.asarray(env.state()).shape[0])
    if new_state_dim != state_dim:
        raise ValueError(
            f"curriculum env swap changed the global state width: {new_state_dim} != {state_dim}"
        )


def _delivery_rate(rollout: Rollout) -> float:
    """Mean fraction of the swarm's peers whose message reached an agent, over the rollout.

    ``1.0`` on an unconstrained channel; it falls as the CommsModel gates, sheds, drops, and
    delays links — the denominator a comms-learning score must be read against."""
    received = 0
    possible = 0
    for step in rollout.steps:
        live = len(step.obs)
        if live < 2:
            continue
        for agent in step.obs:
            received += len(step.reach.get(agent, ()))
            possible += live - 1
    return float(received / possible) if possible else 0.0
