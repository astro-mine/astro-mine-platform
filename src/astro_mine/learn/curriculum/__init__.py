"""Curricula & scenario generation — staged difficulty + domain randomization (learn.md §3).

The ``curriculum/`` module of learn.md §3's tree: *"staged difficulty, automatic curricula,
scenario generators"*, realized to learn.md §11's MVP recommendation — **"hand-authored staged
curricula + domain randomization for the MVP, with an automatic-curriculum plugin interface for
research"**.

Why a swarm needs one: comms-limited cooperation under partial observability (charter §8) is
often not learnable from a cold start on the *hard* channel. A curriculum makes it learnable —
train on a clear channel, then degrade it (drop → delay → range gate → bandwidth budget) as the
swarm demonstrates competence — and domain randomization keeps the result honest by never
showing the same regime twice.

- :class:`CurriculumSpec` / :class:`StageSpec` / :class:`RandomizationSpec` — the declarative,
  JSON-Schema-validated documents (conventions.md §3), part of the run's reproducibility key.
- :class:`StagedCurriculum` — the hand-authored ladder; :class:`DomainRandomizer` — the seeded
  per-episode sampler. Same seed ⇒ same stage sequence ⇒ same sampled worlds.
- :class:`Curriculum` (Protocol) — the plugin contract an **automatic** curriculum (PLR,
  teacher-student) implements, discovered through :class:`CurriculumRegistry`'s entry-point
  group. Deferred to Phase 2 by the roadmap; the interface is open now, by construction.
- :func:`comms_ladder` / :func:`randomized_comms` — the shipped examples.
- :func:`run_curriculum` — the loop that actually *trains through* a curriculum: it rebuilds the
  world at each promotion while the learner (nets, optimizer, seeded RNG) carries over, which is
  what makes staged difficulty accumulate competence rather than restart it.

The schema/registry surface is Torch-free (a curriculum produces *configs*); only
:func:`run_curriculum` touches a trainer, and it imports one lazily through the registry.
"""

from __future__ import annotations

from astro_mine.learn.curriculum.library import comms_ladder, randomized_comms
from astro_mine.learn.curriculum.loop import (
    CurriculumReport,
    StageEnvFactory,
    run_curriculum,
    stage_metrics,
)
from astro_mine.learn.curriculum.registry import (
    CURRICULUM_ENTRY_POINT_GROUP,
    CurriculumFactory,
    CurriculumRegistry,
    default_curriculum_registry,
)
from astro_mine.learn.curriculum.spec import (
    CURRICULUM_SCHEMA_VERSION,
    AdvanceRule,
    CurriculumSpec,
    RandomizationSpec,
    StageSpec,
    UniformRange,
)
from astro_mine.learn.curriculum.staged import (
    Curriculum,
    DomainRandomizer,
    Stage,
    StagedCurriculum,
)

__all__ = [
    "CURRICULUM_ENTRY_POINT_GROUP",
    "CURRICULUM_SCHEMA_VERSION",
    "AdvanceRule",
    "Curriculum",
    "CurriculumFactory",
    "CurriculumRegistry",
    "CurriculumReport",
    "CurriculumSpec",
    "DomainRandomizer",
    "RandomizationSpec",
    "Stage",
    "StageEnvFactory",
    "StageSpec",
    "StagedCurriculum",
    "UniformRange",
    "comms_ladder",
    "default_curriculum_registry",
    "randomized_comms",
    "run_curriculum",
    "stage_metrics",
]
