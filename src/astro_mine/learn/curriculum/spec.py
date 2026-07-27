"""Declarative curriculum + domain-randomization schema (learn.md §3, §11).

learn.md §11 recommends **"hand-authored staged curricula + domain randomization for the MVP,
with an automatic-curriculum plugin interface for research"**, and learn.md §3 makes
``Curriculum`` / ``ScenarioGenerator`` a key abstraction: "produce a (possibly stateful) stream
of env configs; ``update(metrics)`` advances staged or automatic curricula based on observed
performance".

Like :class:`~astro_mine.learn.algos.TrainConfig` and
:class:`~astro_mine.learn.envs.CommsModelConfig`, a curriculum is a **Pydantic v2 document
validated by JSON Schema** (conventions.md §3) — so it is part of the run's reproducibility key,
round-trips through JSON, is content-hashable, and lands verbatim in provenance
(conventions.md §5, §11). Same curriculum + same seed ⇒ the same stage sequence and the same
sampled randomizations.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from astro_mine.learn.envs.comms import CommsModelConfig

__all__ = [
    "CURRICULUM_SCHEMA_VERSION",
    "AdvanceRule",
    "CurriculumSpec",
    "RandomizationSpec",
    "StageSpec",
    "UniformRange",
]

#: Bumped when the *meaning* of the curriculum schema changes (mirrors CommsModelConfig).
CURRICULUM_SCHEMA_VERSION = "0.1.0"

_Prob = Annotated[float, Field(ge=0.0, le=1.0)]
_Pos = Annotated[int, Field(gt=0)]

#: An inclusive ``[low, high]`` sampling range for a domain-randomized knob.
UniformRange = tuple[float, float]


class _Doc(BaseModel):
    """Frozen, ``extra``-forbidding base — a typo'd stage key fails loudly rather than
    silently disabling a curriculum stage."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class AdvanceRule(_Doc):
    """When a staged curriculum promotes the swarm to the next stage.

    "``update(metrics)`` advances staged or automatic curricula based on observed performance"
    (learn.md §3): the stage advances once ``metric`` has held past ``threshold`` for
    ``patience`` consecutive updates, and never before ``min_iterations`` — so a single lucky
    iteration cannot promote a policy that has not actually learned the stage (the same
    anti-single-seat-luck posture as the honest-eval harness, learn.md §2.8).

    ``mode='ge'`` advances on a metric that should *rise* (reward); ``mode='le'`` on one that
    should *fall* (a loss, a violation count)."""

    metric: str = "mean_reward"
    threshold: float = 0.0
    mode: Literal["ge", "le"] = "ge"
    patience: _Pos = 1
    min_iterations: _Pos = 1

    def satisfied(self, value: float) -> bool:
        """Whether one observation of ``metric`` clears the bar."""
        return value >= self.threshold if self.mode == "ge" else value <= self.threshold


class RandomizationSpec(_Doc):
    """Per-stage **domain randomization** over the comms regime and the world knobs.

    Each declared range is sampled i.i.d. per episode from a stream seeded by
    ``(run seed, stage, episode)``, so a randomized curriculum is *still* byte-reproducible: the
    same seed replays the same sequence of sampled worlds (conventions.md §5). An undeclared
    (``None``) knob is not randomized — it keeps the stage's base value.

    The comms knobs are exactly the CommsModel's degradation axes (drop, delay, range gate), so
    randomizing them varies *the charter §8 difficulty itself*, which is the point: a policy that
    only works at one drop rate has not solved comms-limited cooperation.

    ``world_provider`` randomizes the opaque, Core-resolved world selector
    (:attr:`TrainConfig.world_provider`) — Learn does not interpret those keys, it only samples
    and records them."""

    drop_probability: UniformRange | None = None
    delay_mean_ticks: UniformRange | None = None
    max_range_m: UniformRange | None = None
    world_provider: dict[str, UniformRange] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_ranges(self) -> RandomizationSpec:
        named = {
            "drop_probability": self.drop_probability,
            "delay_mean_ticks": self.delay_mean_ticks,
            "max_range_m": self.max_range_m,
            **self.world_provider,
        }
        for key, bounds in named.items():
            if bounds is not None and bounds[0] > bounds[1]:
                raise ValueError(f"{key} range requires low <= high, got {bounds}")
        if self.drop_probability is not None and not (
            self.drop_probability[0] >= 0.0 and self.drop_probability[1] <= 1.0
        ):
            raise ValueError("drop_probability range must lie in [0, 1]")
        return self

    @property
    def is_empty(self) -> bool:
        """True when the stage randomizes nothing (a purely hand-authored stage)."""
        return (
            self.drop_probability is None
            and self.delay_mean_ticks is None
            and self.max_range_m is None
            and not self.world_provider
        )


class StageSpec(_Doc):
    """One hand-authored curriculum stage: a comms regime + training knobs + an advance rule.

    ``config_overrides`` are :class:`~astro_mine.learn.algos.TrainConfig` fields applied on top
    of the run's base config for this stage — notably ``fidelity``, which is why learn.md §2.7
    calls the fidelity dial "a curriculum axis, not an afterthought": a stage can train cheaply
    on ``surrogate``/``gpu_vectorized`` and a later one validate on ``sim_high``."""

    name: str
    comms: CommsModelConfig = CommsModelConfig()
    randomize: RandomizationSpec = RandomizationSpec()
    config_overrides: dict[str, Any] = Field(default_factory=dict)
    advance: AdvanceRule = AdvanceRule()


class CurriculumSpec(_Doc):
    """An ordered curriculum — the staged-difficulty document recorded with a run.

    Emit its JSON Schema with :meth:`model_json_schema` and round-trip it through JSON with
    :meth:`model_dump_json` / :meth:`model_validate_json`; its content hash is part of the run's
    reproducibility key."""

    name: str
    stages: tuple[StageSpec, ...]
    schema_version: Literal["0.1.0"] = "0.1.0"

    @model_validator(mode="after")
    def _check_stages(self) -> CurriculumSpec:
        if not self.stages:
            raise ValueError("a curriculum needs at least one stage")
        names = [stage.name for stage in self.stages]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate stage names: {names}")
        return self

    def __len__(self) -> int:
        return len(self.stages)
