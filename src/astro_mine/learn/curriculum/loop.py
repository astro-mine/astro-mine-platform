"""Train a baseline *through* a curriculum (learn.md §3, §11) — the wiring, not just the schema.

A curriculum that nothing consumes is a document, not a feature. This is the loop that makes it
one: it drives any registered baseline through a :class:`Curriculum`'s stages, rebuilding the
**world** at each promotion (a harder comms regime, a freshly randomized domain, a different
fidelity tier) while the **learner carries over untouched** — nets, optimizer, and seeded
generator all survive a stage change, so competence accumulates instead of restarting. That
carry-over is the whole point of a curriculum, and it is what
:meth:`~astro_mine.learn.algos._ppo.PpoTrainer.set_env` exists for.

The world is supplied as a :data:`StageEnvFactory` — ``Stage -> SwarmEnv`` — because only the
caller knows how to build its world; Learn hands it the resolved stage (the sampled
:class:`CommsModelConfig`, the ``world_provider`` selector, the stage's ``TrainConfig``) and
takes back a :class:`~astro_mine.learn.envs.SwarmEnv`. Everything stays behind the Core
Environment contract (learn.md §2.2).

Reproducibility is preserved end to end: the stage sequence is a deterministic function of the
metrics, the per-episode domain randomization is seeded by ``(seed, stage, episode)``, and both
are recorded — so the same seed replays the same ladder over the same sampled worlds
(conventions.md §5, §11).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from astro_mine.learn.algos.registry import AlgorithmRegistry, default_registry
from astro_mine.learn.curriculum.staged import Curriculum, Stage

if TYPE_CHECKING:
    from astro_mine.learn.algos._contract import Trainer
    from astro_mine.learn.envs import SwarmEnv
    from astro_mine.learn.track import TrackedRun

__all__ = ["CurriculumReport", "StageEnvFactory", "run_curriculum", "stage_metrics"]

#: Build the world for a resolved curriculum stage. The caller owns world construction; Learn
#: hands it the stage (sampled comms regime + world_provider + TrainConfig) and takes a SwarmEnv.
StageEnvFactory = Callable[[Stage], "SwarmEnv"]


@dataclass
class CurriculumReport:
    """The reproducible result of a curriculum run."""

    curriculum: str
    algorithm: str
    #: The per-iteration metrics, in order — the learning curve *across* stages.
    metrics: list[dict[str, float]] = field(default_factory=list)
    #: Which stage each iteration ran in (same length as :attr:`metrics`).
    stage_at_iteration: list[int] = field(default_factory=list)
    #: The resolved stage (with its content hash) at every promotion — the replay record.
    stages: list[dict[str, Any]] = field(default_factory=list)
    completed: bool = False

    @property
    def learning_curve(self) -> list[float]:
        return [m.get("mean_reward", 0.0) for m in self.metrics]

    @property
    def stages_reached(self) -> int:
        """How many distinct stages the run actually got through."""
        return len(set(self.stage_at_iteration))

    def to_dict(self) -> dict[str, Any]:
        return {
            "curriculum": self.curriculum,
            "algorithm": self.algorithm,
            "metrics": [dict(m) for m in self.metrics],
            "stage_at_iteration": list(self.stage_at_iteration),
            "stages": list(self.stages),
            "completed": self.completed,
        }


def run_curriculum(
    algorithm_tag: str,
    curriculum: Curriculum,
    env_factory: StageEnvFactory,
    *,
    iterations: int,
    registry: AlgorithmRegistry | None = None,
    track: TrackedRun | None = None,
) -> tuple[CurriculumReport, Trainer]:
    """Train ``algorithm_tag`` through ``curriculum`` for ``iterations`` iterations.

    Each iteration runs one training iteration in the current stage's world. The stage's metrics
    are fed back to :meth:`Curriculum.update`; on a promotion the world is rebuilt from the new
    stage and swapped into the *same* trainer (``set_env``) — so the policy keeps everything it
    learned. An **automatic** curriculum (PLR, teacher-student) drops in here unchanged: it is
    just a :class:`Curriculum` whose ``update`` picks the next stage differently.

    Returns the report and the live trainer (so the caller can ``export()`` the trained policy).
    ``track`` optionally streams every iteration's metrics and every stage promotion into a
    :class:`~astro_mine.learn.track.TrackedRun`.

    The trainer is built once, from **stage 0**'s resolved :class:`TrainConfig` (the curriculum's
    base config with that stage's overrides applied). Later stages may retune the *world* and the
    rollout knobs; they do not rebuild the learner, which is the invariant that makes a curriculum
    a curriculum."""
    reg = registry if registry is not None else default_registry()
    stage = curriculum.stage(episode=0)

    trainer = reg.get(algorithm_tag).make_trainer(env_factory(stage), stage.train_config)
    report = CurriculumReport(curriculum=curriculum.spec.name, algorithm=algorithm_tag)
    _record_stage(report, stage, track)

    for iteration in range(iterations):
        metrics = trainer.train_iteration()
        report.metrics.append(dict(metrics))
        report.stage_at_iteration.append(curriculum.stage_index)
        if track is not None:
            track.log_iteration({**metrics, "curriculum_stage": float(curriculum.stage_index)})

        advanced = curriculum.update(metrics)
        if curriculum.done:
            report.completed = True
        if not advanced:
            continue
        # Promotion: a harder world, the SAME learner. Nets/optimizer/RNG all carry over.
        stage = curriculum.stage(episode=iteration + 1)
        trainer.set_env(env_factory(stage))  # type: ignore[attr-defined]
        _record_stage(report, stage, track)

    return report, trainer


def _record_stage(report: CurriculumReport, stage: Stage, track: TrackedRun | None) -> None:
    entry: dict[str, Any] = {**stage.provenance(), "content_hash": stage.content_hash()}
    report.stages.append(entry)
    if track is not None:
        track.log_stage(stage)


def stage_metrics(report: CurriculumReport) -> Mapping[int, list[float]]:
    """The learning curve split per stage — how the swarm did at each difficulty level."""
    per_stage: dict[int, list[float]] = {}
    for metrics, index in zip(report.metrics, report.stage_at_iteration, strict=True):
        per_stage.setdefault(index, []).append(metrics.get("mean_reward", 0.0))
    return per_stage
