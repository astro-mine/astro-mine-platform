# SPDX-License-Identifier: Apache-2.0
"""Comms-learning PPO — the differentiable-message CTDE baseline (RM-P1-LEARN-03; learn.md §11).

learn.md §11 makes **comms-learning a first-class research track** alongside IPPO/MAPPO/QMIX,
"because comms-limited cooperation *is* the charter §8 problem". This is that track's
registered baseline: MAPPO's centralized critic **plus a learned message channel**.

The whole point is that the learned messages ride the *same* channel every other baseline is
scored on. Each tick, an agent's message reaches a peer only if the
:class:`~astro_mine.learn.envs.CommsModel` delivered it — after line-of-sight/range **gating**,
the per-agent bandwidth **budget**, the stochastic **drop**, and the delivery **delay**. The
actor conditions on the mean-pool of the messages that actually arrived (the zero vector if it
is isolated), and the message encoder is trained **end-to-end** by the team objective: the
aggregate is recomputed inside the PPO update from the executor's *recorded* reachability, so
gradients flow back into "what to say" (:meth:`PpoTrainer._comms_context`).

That is what makes a comms-learning result comparable rather than a special case: the same
:class:`~astro_mine.learn.envs.CommsModel` config, the same comms-stress curves, the same
leaderboard (learn.md §2, §3) — and a policy that exports to the *same* ONNX
:class:`~astro_mine.core.policy.PolicyPackage`, with the aggregated peer-message context
declared as an explicit ``msg`` graph input rather than smuggled inside the observation.

A thin :class:`~astro_mine.learn.algos._contract.Algorithm` factory over the shared
:class:`~astro_mine.learn.algos._ppo.PpoTrainer` (``centralized=True, comms=True``).
"""

from __future__ import annotations

from astro_mine.learn.algos._contract import AlgorithmSpec
from astro_mine.learn.algos._ppo import PpoTrainer
from astro_mine.learn.algos._specs import COMMS_PPO_SPEC
from astro_mine.learn.algos.config import TrainConfig
from astro_mine.learn.envs import SwarmEnv
from astro_mine.learn.train.executor import RewardFn, RolloutExecutor

__all__ = ["CommsPpoAlgorithm"]


class CommsPpoAlgorithm:
    """The registered comms-learning plugin (capability tag ``marl.ctde.comms_ppo``)."""

    @property
    def spec(self) -> AlgorithmSpec:
        return COMMS_PPO_SPEC

    def make_trainer(
        self,
        env: SwarmEnv,
        config: TrainConfig,
        *,
        reward_fn: RewardFn | None = None,
        executor: RolloutExecutor | None = None,
    ) -> PpoTrainer:
        return PpoTrainer(
            COMMS_PPO_SPEC,
            env,
            config,
            centralized=True,
            comms=True,
            reward_fn=reward_fn,
            executor=executor,
        )
