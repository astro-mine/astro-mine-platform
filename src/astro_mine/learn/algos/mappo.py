# SPDX-License-Identifier: Apache-2.0
"""MAPPO — the CTDE-default PPO baseline (RM-P1-LEARN-03; learn.md §11).

Decentralised actors, one shared centralized critic over the global ``SwarmEnv.state()``,
and a cooperative team reward — the CTDE bargain: training uses global information the
execution-time policy does not. The algorithm declares its centralized-critic input spec (a
:class:`~astro_mine.learn.algos._contract.CentralizedCriticSpec`, exposed on the trainer).
A thin factory over the shared :class:`~astro_mine.learn.algos._ppo.PpoTrainer`
(``centralized=True``).
"""

from __future__ import annotations

from astro_mine.learn.algos._contract import AlgorithmSpec
from astro_mine.learn.algos._ppo import PpoTrainer
from astro_mine.learn.algos._specs import MAPPO_SPEC
from astro_mine.learn.algos.config import TrainConfig
from astro_mine.learn.envs import SwarmEnv
from astro_mine.learn.train.executor import RewardFn, RolloutExecutor

__all__ = ["MappoAlgorithm"]


class MappoAlgorithm:
    """The registered MAPPO plugin (capability tag ``marl.ctde.mappo``)."""

    @property
    def spec(self) -> AlgorithmSpec:
        return MAPPO_SPEC

    def make_trainer(
        self,
        env: SwarmEnv,
        config: TrainConfig,
        *,
        reward_fn: RewardFn | None = None,
        executor: RolloutExecutor | None = None,
    ) -> PpoTrainer:
        return PpoTrainer(
            MAPPO_SPEC, env, config, centralized=True, reward_fn=reward_fn, executor=executor
        )
