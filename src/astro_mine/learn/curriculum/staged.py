# SPDX-License-Identifier: Apache-2.0
"""Staged curricula + domain randomization — the MVP curriculum (learn.md §3, §11).

The :class:`Curriculum` **plugin contract** and its hand-authored realization.

learn.md §11: *"Hand-authored staged curricula + domain randomization for the MVP, with an
automatic-curriculum plugin interface for research."* Both halves live here:

- :class:`StagedCurriculum` — the hand-authored ladder. Runs a stage until its
  :class:`~astro_mine.learn.curriculum.spec.AdvanceRule` is satisfied, then promotes. Each stage
  is a comms regime (the charter §8 difficulty dial), a set of ``TrainConfig`` overrides
  (including the *fidelity* tier — learn.md §2.7: "the fidelity dial is a curriculum axis"), and
  an optional domain-randomization spec.
- :class:`DomainRandomizer` — the per-episode sampler. Seeded by ``(seed, stage, episode)``, so
  a randomized curriculum stays byte-reproducible: the same seed replays the same sequence of
  sampled worlds (conventions.md §5).
- :class:`Curriculum` (Protocol) — the **plugin interface an automatic curriculum implements**.
  A PLR / teacher-student / regret-based curriculum is exactly a :class:`Curriculum` whose
  :meth:`~Curriculum.update` chooses the next stage from observed performance instead of walking
  a fixed ladder; it registers through the same entry-point group
  (:mod:`astro_mine.learn.curriculum.registry`) and needs no change here. That is the Phase-2
  "automatic curricula" deferral (roadmap phase-1 "Deferred → P2") kept open by construction.

Torch-free: a curriculum only produces *configs*, so this whole module runs without the
``[rllib]`` extra.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from astro_mine.core.hashing import content_hash_json
from astro_mine.learn.algos.config import TrainConfig
from astro_mine.learn.curriculum.spec import CurriculumSpec, RandomizationSpec, StageSpec
from astro_mine.learn.envs.comms import CommsModelConfig, DelayConfig, DropConfig

__all__ = ["Curriculum", "DomainRandomizer", "Stage", "StagedCurriculum"]


@dataclass(frozen=True)
class Stage:
    """A **resolved** curriculum stage — the concrete env + training config to run right now.

    What the trainer actually consumes: the stage's :class:`TrainConfig` (base config + the
    stage's overrides), the :class:`CommsModelConfig` for the ``SwarmEnv``'s
    :class:`~astro_mine.learn.envs.CommsModel` (already domain-randomized for this episode), and
    the opaque ``world_provider`` selector Core resolves to a fidelity tier. ``provenance`` is
    the content-addressed record of exactly what was sampled — the reproducibility key for this
    episode."""

    index: int
    name: str
    train_config: TrainConfig
    comms: CommsModelConfig
    world_provider: dict[str, Any] = field(default_factory=dict)
    episode: int = 0

    def provenance(self) -> dict[str, Any]:
        """The JSON-serializable record of this resolved stage (hashable into run provenance)."""
        return {
            "stage_index": self.index,
            "stage_name": self.name,
            "episode": self.episode,
            "train_config": self.train_config.model_dump(mode="json"),
            "comms": self.comms.model_dump(mode="json"),
            "world_provider": dict(self.world_provider),
        }

    def content_hash(self) -> str:
        """``sha256:…`` of the resolved stage — two episodes with this hash are the same world."""
        return content_hash_json(self.provenance())


@runtime_checkable
class Curriculum(Protocol):
    """The curriculum plugin contract (learn.md §3).

    "Produce a (possibly stateful) stream of env configs; ``update(metrics)`` advances staged or
    automatic curricula based on observed performance." A **hand-authored** curriculum walks a
    fixed ladder (:class:`StagedCurriculum`); an **automatic** one (PLR, teacher-student,
    regret-based) implements this same interface and picks the next stage from the metrics it is
    handed — no change to the trainer, the registry, or this contract.
    """

    @property
    def spec(self) -> CurriculumSpec: ...

    @property
    def stage_index(self) -> int:
        """Which stage is current (0-based)."""
        ...

    @property
    def done(self) -> bool:
        """Whether the final stage's advance rule has been satisfied (the curriculum is over)."""
        ...

    def stage(self, *, episode: int = 0) -> Stage:
        """The resolved stage to run for ``episode`` (domain randomization sampled per episode)."""
        ...

    def update(self, metrics: Mapping[str, float]) -> bool:
        """Feed one iteration's metrics back; return ``True`` iff the curriculum advanced."""
        ...


class DomainRandomizer:
    """Seeded per-episode sampler for a stage's :class:`RandomizationSpec` (learn.md §11).

    Domain randomization varies the comms difficulty *itself* — drop probability, delivery
    delay, range gate — because a policy that only cooperates at one drop rate has not solved
    the charter §8 problem. The stream is derived from ``(seed, stage_index, episode)`` through
    :class:`numpy.random.SeedSequence` (the salting pattern ``CommsModel.reset`` uses), so
    sampling is *deterministic and replayable*: same seed ⇒ same sequence of sampled worlds,
    independent of how far any other RNG has advanced."""

    def __init__(self, spec: RandomizationSpec, *, seed: int = 0, stage_index: int = 0) -> None:
        self.spec = spec
        self._seed = int(seed)
        self._stage = int(stage_index)

    def _rng(self, episode: int) -> np.random.Generator:
        return np.random.default_rng(
            np.random.SeedSequence([self._seed, self._stage, int(episode)])
        )

    def sample(
        self, base: CommsModelConfig, *, episode: int = 0
    ) -> tuple[CommsModelConfig, dict[str, Any]]:
        """Sample this episode's comms regime + world-provider knobs from the stage's ranges.

        Undeclared knobs keep ``base``'s values, so a stage randomizes exactly what it declares
        and nothing else."""
        if self.spec.is_empty:
            return base, {}
        rng = self._rng(episode)
        comms = base

        if self.spec.drop_probability is not None:
            low, high = self.spec.drop_probability
            comms = comms.model_copy(
                update={"drop": DropConfig(probability=float(rng.uniform(low, high)))}
            )
        if self.spec.delay_mean_ticks is not None:
            low, high = self.spec.delay_mean_ticks
            mean = float(rng.uniform(low, high))
            comms = comms.model_copy(
                update={
                    "delay": DelayConfig(
                        kind="geometric", mean_ticks=mean, max_ticks=base.delay.max_ticks
                    )
                }
            )
        if self.spec.max_range_m is not None:
            low, high = self.spec.max_range_m
            gate = comms.range_gate
            comms = comms.model_copy(
                update={
                    "range_gate": gate.model_copy(
                        update={"max_range_m": float(rng.uniform(low, high))}
                    )
                }
            )
        world_provider = {
            key: float(rng.uniform(low, high))
            for key, (low, high) in sorted(self.spec.world_provider.items())
        }
        return comms, world_provider


class StagedCurriculum:
    """A hand-authored staged curriculum — the MVP realization of :class:`Curriculum`.

    Walks :attr:`CurriculumSpec.stages` in order, holding each stage until its
    :class:`AdvanceRule` has been satisfied for ``patience`` consecutive updates (and never
    before ``min_iterations``), then promoting. The final stage's satisfaction sets
    :attr:`done` — the curriculum is complete, but :meth:`stage` keeps returning the last stage
    so a training loop can simply keep going."""

    def __init__(self, spec: CurriculumSpec, *, seed: int = 0, base: TrainConfig | None = None):
        self._spec = spec
        self._seed = seed
        self._base = base if base is not None else TrainConfig(seed=seed)
        self._index = 0
        self._streak = 0
        self._iterations = 0
        self._done = False
        #: Per-stage promotion record — what the run report / MLflow tags carry.
        self.history: list[dict[str, Any]] = []

    # --- Curriculum -----------------------------------------------------------------

    @property
    def spec(self) -> CurriculumSpec:
        return self._spec

    @property
    def stage_index(self) -> int:
        return self._index

    @property
    def stage_spec(self) -> StageSpec:
        """The current stage's authored spec."""
        return self._spec.stages[self._index]

    @property
    def done(self) -> bool:
        return self._done

    def stage(self, *, episode: int = 0) -> Stage:
        """Resolve the current stage for ``episode``: base config + stage overrides, with this
        episode's domain randomization sampled in."""
        authored = self.stage_spec
        randomizer = DomainRandomizer(authored.randomize, seed=self._seed, stage_index=self._index)
        comms, sampled_world = randomizer.sample(authored.comms, episode=episode)
        overrides = dict(authored.config_overrides)
        world_provider = {**self._base.world_provider, **overrides.pop("world_provider", {})}
        world_provider.update(sampled_world)
        train_config = self._base.model_copy(update={**overrides, "world_provider": world_provider})
        return Stage(
            index=self._index,
            name=authored.name,
            train_config=train_config,
            comms=comms,
            world_provider=world_provider,
            episode=episode,
        )

    def update(self, metrics: Mapping[str, float]) -> bool:
        """Feed one iteration's metrics back; advance when the stage's rule is met.

        A missing metric simply does not satisfy the rule (it never promotes on absent
        evidence). Returns ``True`` iff this call promoted the curriculum to a new stage."""
        rule = self.stage_spec.advance
        self._iterations += 1
        value = metrics.get(rule.metric)
        if value is not None and rule.satisfied(float(value)):
            self._streak += 1
        else:
            self._streak = 0

        ready = self._streak >= rule.patience and self._iterations >= rule.min_iterations
        if not ready:
            return False
        self.history.append(
            {
                "stage_index": self._index,
                "stage_name": self.stage_spec.name,
                "iterations": self._iterations,
                "metric": rule.metric,
                "value": None if value is None else float(value),
            }
        )
        if self._index + 1 >= len(self._spec.stages):
            self._done = True
            return False  # the last stage cannot promote to a next one
        self._index += 1
        self._streak = 0
        self._iterations = 0
        return True
