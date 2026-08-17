# SPDX-License-Identifier: Apache-2.0
"""IPPO — independent PPO, the simple control baseline (RM-P1-LEARN-03; learn.md §11).

Each agent optimises its own actor-critic against its own reward with **no** shared
information — the honest lower bound the CTDE baselines (MAPPO/QMIX) must beat on the Bench
leaderboard. A thin :class:`~astro_mine.learn.algos._contract.Algorithm` factory over the
shared :class:`~astro_mine.learn.algos._ppo.PpoTrainer` (``centralized=False``).
"""

from __future__ import annotations

from astro_mine.learn.algos._contract import AlgorithmSpec
from astro_mine.learn.algos._ppo import PpoTrainer
from astro_mine.learn.algos._specs import IPPO_SPEC
from astro_mine.learn.algos.config import TrainConfig
from astro_mine.learn.envs import SwarmEnv
from astro_mine.learn.train.executor import RewardFn, RolloutExecutor

__all__ = ["IppoAlgorithm"]


class IppoAlgorithm:
    """The registered IPPO plugin (capability tag ``marl.independent.ppo``)."""

    @property
    def spec(self) -> AlgorithmSpec:
        return IPPO_SPEC

    def make_trainer(
        self,
        env: SwarmEnv,
        config: TrainConfig,
        *,
        reward_fn: RewardFn | None = None,
        executor: RolloutExecutor | None = None,
    ) -> PpoTrainer:
        return PpoTrainer(
            IPPO_SPEC, env, config, centralized=False, reward_fn=reward_fn, executor=executor
        )
