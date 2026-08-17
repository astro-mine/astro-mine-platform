# SPDX-License-Identifier: Apache-2.0
"""QMIX / VDN — the value-based CTDE-default baseline (RM-P1-LEARN-03; learn.md §11).

**Design choice (flagged for the PR).** RLlib's new-API-stack QMIX is *not* first-class with
heterogeneous, capability-keyed Dict observation/action spaces — it assumes homogeneous,
groupable agents over a single discrete action. Rather than block on that limitation
(learn.md §11 "don't block on RLlib limitations"), Learn ships a **lightweight in-house**
QMIX/VDN over shared-shape per-agent Q-nets on the **discrete task selector** (the Core
tagged-union ``kind``): each agent's :class:`~astro_mine.learn.models.mlp.AgentQNet` scores
its ``kind`` choices, chosen Q-values are mixed by a monotonic :class:`QMixer` (or additive
:class:`VDNMixer`) conditioned on the global ``SwarmEnv.state()``, and the joint value is
regressed onto the 1-step TD target. Continuous action blocks (``goto``/``hop``) are held at
their neutral value — QMIX decides *which task*, not its continuous parameters — a documented
scope of the value-based baseline. It is a registered, reproducible plugin; the PPO baselines
cover the continuous-control regime.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import torch

from astro_mine.core.env.model import AgentId
from astro_mine.core.policy import Policy
from astro_mine.learn.algos._contract import (
    AlgorithmSpec,
    CentralizedCriticSpec,
    IoSignature,
    PolicyAssumptions,
    PolicyExport,
)
from astro_mine.learn.algos._ppo import _assert_same_shape
from astro_mine.learn.algos._specs import QMIX_SPEC
from astro_mine.learn.algos._torch_common import (
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
from astro_mine.learn.models.critics import QMixer, VDNMixer
from astro_mine.learn.models.mlp import AgentQNet
from astro_mine.learn.train.executor import (
    AgentStepFn,
    LocalExecutor,
    RewardFn,
    Rollout,
    RolloutExecutor,
    StepDecision,
    default_reward_fn,
)

__all__ = ["QmixAlgorithm", "QmixTrainer"]


def _kind_sample_from_arch(agent_arch: dict[str, Any], kind: int) -> dict[str, object]:
    """A neutral action sample selecting ``kind`` — continuous blocks held at zero (QMIX
    decides *which task*, not its parameters) and any non-``kind`` discrete head at 0."""
    sample: dict[str, object] = {"kind": int(kind)}
    for name in agent_arch["other_discrete"]:
        sample[name] = 0
    for name, width in agent_arch["box_heads"].items():
        sample[name] = np.zeros(width, dtype=np.float32)
    return sample


class _QmixStep:
    """The QMIX ε-greedy step as a :class:`~astro_mine.learn.train.executor.BroadcastableStep`.

    Callable like the in-process step (per-agent :class:`~astro_mine.learn.models.mlp.AgentQNet`
    argmax over the ``kind`` selector, ε-greedy exploration from the shared generator), and
    shippable to a KubeRay worker via :meth:`broadcast` (per-agent ``state_dict`` + Q-net
    architecture + ε + generator state). Holds the trainer's live Q-nets + generator."""

    def __init__(
        self,
        qnets: Mapping[AgentId, AgentQNet],
        generator: torch.Generator,
        arch: Mapping[AgentId, dict[str, Any]],
        epsilon: float,
    ) -> None:
        self._qnets = dict(qnets)
        self._generator = generator
        self._arch = dict(arch)
        self._epsilon = epsilon

    def __call__(self, flat_obs: Mapping[AgentId, np.ndarray]) -> Mapping[AgentId, StepDecision]:
        decisions: dict[AgentId, StepDecision] = {}
        for agent, obs in flat_obs.items():
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).unsqueeze(0)
            with torch.no_grad():
                q = self._qnets[agent](obs_t).squeeze(0)
            n_kinds = self._arch[agent]["n_kinds"]
            explore = torch.rand((), generator=self._generator).item() < self._epsilon
            if explore:
                kind = int(torch.randint(n_kinds, (), generator=self._generator).item())
            else:
                kind = int(q.argmax().item())
            decisions[agent] = StepDecision(
                action_sample=_kind_sample_from_arch(self._arch[agent], kind)
            )
        return decisions

    def batch(
        self,
        flat_obs: Mapping[AgentId, np.ndarray],
        reach: np.ndarray | None = None,
    ) -> Mapping[AgentId, list[StepDecision]]:
        """ε-greedy decisions for a whole batch of vectorized env copies in **one** Q-forward
        per agent — the :class:`~astro_mine.learn.train.executor.BatchedStep` contract that lets
        the GPU-vectorized executor run a batched kernel instead of a sequential loop.

        The value-based baseline is comms-blind, so ``reach`` is accepted (uniform seam) and
        ignored. Exploration draws from the same seeded generator in the same per-env order as
        the sequential path, so a batch of one is byte-identical to it."""
        decisions: dict[AgentId, list[StepDecision]] = {}
        for agent, obs in flat_obs.items():
            obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32))
            with torch.no_grad():
                q = self._qnets[agent](obs_t)  # (num_envs, n_kinds) — one forward, whole batch
            agent_arch = self._arch[agent]
            n_kinds = agent_arch["n_kinds"]
            greedy = q.argmax(dim=-1)
            rows: list[StepDecision] = []
            for env_index in range(int(obs_t.shape[0])):
                explore = torch.rand((), generator=self._generator).item() < self._epsilon
                if explore:
                    kind = int(torch.randint(n_kinds, (), generator=self._generator).item())
                else:
                    kind = int(greedy[env_index].item())
                rows.append(StepDecision(action_sample=_kind_sample_from_arch(agent_arch, kind)))
            decisions[agent] = rows
        return decisions

    def broadcast(self) -> tuple[Callable[[Any], AgentStepFn], dict[str, Any]]:
        state: dict[str, Any] = {
            "arch": self._arch,
            "weights": {agent: net.state_dict() for agent, net in self._qnets.items()},
            "generator_state": self._generator.get_state(),
            "epsilon": self._epsilon,
        }
        return _rebuild_qmix_step, state

    def rng_state(self) -> Any:
        return self._generator.get_state()

    def set_rng_state(self, state: Any) -> None:
        self._generator.set_state(state)


def _rebuild_qmix_step(state: dict[str, Any]) -> _QmixStep:
    """Reconstruct a :class:`_QmixStep` on a rollout worker from a broadcast ``state``."""
    generator = torch.Generator()
    generator.set_state(state["generator_state"])
    qnets: dict[AgentId, AgentQNet] = {}
    for agent, arch in state["arch"].items():
        net = AgentQNet(arch["obs_dim"], arch["n_kinds"], arch["hidden_sizes"])
        net.load_state_dict(state["weights"][agent])
        qnets[agent] = net
    return _QmixStep(qnets, generator, state["arch"], state["epsilon"])


class QmixTrainer:
    """A reproducible QMIX/VDN trainer over the discrete ``kind`` selector."""

    def __init__(
        self,
        env: SwarmEnv,
        config: TrainConfig,
        *,
        reward_fn: RewardFn | None = None,
        executor: RolloutExecutor | None = None,
    ) -> None:
        seed_everything(config.seed)
        self._env = env
        self._config = config
        self._reward_fn: RewardFn = reward_fn if reward_fn is not None else default_reward_fn
        self._executor: RolloutExecutor = executor if executor is not None else LocalExecutor()
        self._generator = make_generator(config.seed)
        self._history: list[dict[str, float]] = []

        self._specs = env.agent_specs
        self._agents = tuple(self._specs)
        self._col = {agent: i for i, agent in enumerate(self._agents)}
        self._n_kinds = {
            a: action_heads(s.action_space).discrete["kind"] for a, s in self._specs.items()
        }
        self._qnets: dict[AgentId, AgentQNet] = {
            a: AgentQNet(flat_obs_dim(s.observation_space), self._n_kinds[a], config.hidden_sizes)
            for a, s in self._specs.items()
        }
        # Serializable per-agent Q-net architecture + action-head layout — the payload a
        # KubeRay worker rebuilds the ε-greedy step from (see :class:`_QmixStep`).
        self._arch: dict[AgentId, dict[str, Any]] = {}
        for a, s in self._specs.items():
            heads = action_heads(s.action_space)
            self._arch[a] = {
                "obs_dim": flat_obs_dim(s.observation_space),
                "n_kinds": self._n_kinds[a],
                "hidden_sizes": tuple(config.hidden_sizes),
                "other_discrete": [name for name in heads.discrete if name != "kind"],
                "box_heads": dict(heads.box),
            }
        self._state_dim = int(np.asarray(env.state()).shape[0])
        self._critic_spec = CentralizedCriticSpec(
            global_state_dim=self._state_dim,
            per_agent_obs_dims={
                a: flat_obs_dim(s.observation_space) for a, s in self._specs.items()
            },
        )
        self._mixer: torch.nn.Module = (
            VDNMixer() if config.mixer == "vdn" else QMixer(len(self._agents), self._state_dim)
        )
        params = [p for net in self._qnets.values() for p in net.parameters()]
        params += list(self._mixer.parameters())
        self._opt = torch.optim.Adam(params, lr=config.lr)
        self._step = _QmixStep(self._qnets, self._generator, self._arch, config.epsilon)

    # --- contract ------------------------------------------------------------------

    @property
    def spec(self) -> AlgorithmSpec:
        return QMIX_SPEC

    @property
    def centralized_critic(self) -> CentralizedCriticSpec:
        return self._critic_spec

    @property
    def rollout_step(self) -> _QmixStep:
        """The live ε-greedy rollout step the executor drives — a ``BroadcastableStep`` and a
        ``BatchedStep``. Exposed so a tool can drive the *real* policy without a training loop
        (the throughput benchmark does)."""
        return self._step

    def set_env(self, env: SwarmEnv) -> None:
        """Swap the environment the next rollout collects from — the **curriculum seam**.

        The Q-nets, mixer, optimizer, and seeded generator carry over untouched across a stage
        promotion; only the world's difficulty changes. The new env must declare the same agents
        and global-state width (see :meth:`PpoTrainer.set_env`)."""
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
        infer: dict[AgentId, InferFn] = {a: self._greedy_infer(a) for a in self._agents}
        return LearnedPolicy(self._specs, infer)

    def export(self) -> PolicyExport:
        io = IoSignature(
            agent_ids=self._agents,
            per_agent={a: agent_io_signature(s) for a, s in self._specs.items()},
            global_state_dim=self._state_dim,
        )
        weights: dict[str, object] = {a: net.state_dict() for a, net in self._qnets.items()}
        weights["__mixer__"] = self._mixer.state_dict()
        return PolicyExport(
            algorithm=QMIX_SPEC.name,
            backend="torch",
            io_signature=io,
            assumptions=PolicyAssumptions(
                comms_observability=self._env.comms_provenance(),
                partial_observability=True,
                surrogate_fidelity_caveats=fidelity_caveats(self._config),
            ),
            provenance=provenance(self._config, comms_provenance=self._env.comms_provenance()),
            weights=weights,
            metrics=self._history[-1] if self._history else {},
            net_kind="q_net",
            net_arch=self._arch,
        )

    # --- rollout + update ----------------------------------------------------------

    def _update(self, rollout: Rollout) -> dict[str, float]:
        n_steps = len(rollout.steps)
        if n_steps == 0:  # pragma: no cover - FakeSwarmWorld always yields steps
            metrics = {"mean_reward": 0.0, "td_loss": 0.0, "env_steps": 0.0}
            return metrics
        n_agents = len(self._agents)
        chosen = torch.zeros(n_steps, n_agents)
        next_q = torch.zeros(n_steps, n_agents)
        for agent, net in self._qnets.items():
            col = self._col[agent]
            idx = [t for t, step in enumerate(rollout.steps) if agent in step.obs]
            if not idx:  # pragma: no cover - every conformant agent is live for ≥1 step
                continue
            obs = torch.from_numpy(np.stack([rollout.steps[t].obs[agent] for t in idx])).float()
            q = net(obs)
            kinds = torch.tensor(
                [int(rollout.steps[t].action[agent]["kind"]) for t in idx], dtype=torch.long
            )
            chosen_vals = q.gather(1, kinds.unsqueeze(1)).squeeze(1)
            chosen[:, col] = torch.zeros(n_steps).index_put((torch.tensor(idx),), chosen_vals)
            # 1-step lookahead: the same agent's obs at t+1 (0 when it left or episode ended).
            nxt_idx = [t for t in idx if t + 1 < n_steps and agent in rollout.steps[t + 1].obs]
            if nxt_idx:
                nxt_obs = torch.from_numpy(
                    np.stack([rollout.steps[t + 1].obs[agent] for t in nxt_idx])
                ).float()
                with torch.no_grad():
                    nxt_max = net(nxt_obs).max(dim=1).values
                next_q[:, col] = torch.zeros(n_steps).index_put((torch.tensor(nxt_idx),), nxt_max)

        states = torch.from_numpy(np.stack([s.state for s in rollout.steps])).float()
        next_states = torch.zeros(n_steps, self._state_dim)
        if n_steps > 1:
            next_states[:-1] = states[1:]
        rewards = torch.tensor(
            [sum(step.reward.values()) for step in rollout.steps], dtype=torch.float32
        )
        dones = torch.zeros(n_steps)
        dones[-1] = 1.0

        joint = self._mixer(chosen, states)
        with torch.no_grad():
            target_next = self._mixer(next_q, next_states)
            target = rewards + self._config.gamma * (1.0 - dones) * target_next
        loss = ((joint - target) ** 2).mean()
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()

        reward_vals = [r for step in rollout.steps for r in step.reward.values()]
        return {
            "mean_reward": float(np.mean(reward_vals)) if reward_vals else 0.0,
            "td_loss": float(loss.item()),
            "env_steps": float(rollout.env_steps()),
        }

    def _greedy_infer(self, agent: AgentId) -> InferFn:
        net = self._qnets[agent]
        agent_arch = self._arch[agent]

        def infer(flat: np.ndarray) -> Mapping[str, object]:
            obs = torch.from_numpy(np.asarray(flat, dtype=np.float32)).unsqueeze(0)
            with torch.no_grad():
                kind = int(net(obs).argmax(dim=-1).item())
            return _kind_sample_from_arch(agent_arch, kind)

        return infer


class QmixAlgorithm:
    """The registered QMIX plugin (capability tag ``marl.ctde.qmix``)."""

    @property
    def spec(self) -> AlgorithmSpec:
        return QMIX_SPEC

    def make_trainer(
        self,
        env: SwarmEnv,
        config: TrainConfig,
        *,
        reward_fn: RewardFn | None = None,
        executor: RolloutExecutor | None = None,
    ) -> QmixTrainer:
        return QmixTrainer(env, config, reward_fn=reward_fn, executor=executor)
