"""The shipped curricula — *replaceable examples*, not privileged internals (charter §10.2).

Two, one per half of learn.md §11's MVP recommendation:

- :func:`comms_ladder` — the **hand-authored staged curriculum**. Three stages that walk the
  swarm up the charter §8 difficulty gradient: a clean channel → a lossy, delayed one → a lossy,
  delayed, *range-gated and bandwidth-starved* one. This is the pedagogy the recommendation is
  about: comms-limited cooperation is hard to learn from scratch, so teach it on an easy channel
  first and degrade the channel as competence is demonstrated.
- :func:`randomized_comms` — the **domain-randomization** example. One stage whose channel is
  re-sampled every episode from declared ranges, so the policy never sees the same comms regime
  twice and cannot overfit to one drop rate. It also randomizes the opaque ``world_provider``
  selector, showing the same mechanism applies to world knobs Learn does not interpret.

Both are ordinary :class:`~astro_mine.learn.curriculum.spec.CurriculumSpec` documents — a
contributor writes their own as JSON/YAML, or registers a whole new
:class:`~astro_mine.learn.curriculum.staged.Curriculum` plugin (an automatic curriculum) through
the entry-point group, without patching this file.
"""

from __future__ import annotations

from astro_mine.learn.curriculum.spec import (
    AdvanceRule,
    CurriculumSpec,
    RandomizationSpec,
    StageSpec,
)
from astro_mine.learn.envs.comms import (
    BandwidthBudgetConfig,
    CommsModelConfig,
    DelayConfig,
    DropConfig,
    RangeGateConfig,
)

__all__ = ["comms_ladder", "randomized_comms"]


def comms_ladder() -> CurriculumSpec:
    """A three-stage hand-authored comms-degradation ladder (the staged-difficulty MVP).

    Stage 1 **clear** — an identity channel (``CommsModelConfig()`` degrades nothing): learn the
    cooperative task itself, with messages always delivered.
    Stage 2 **lossy** — 30% Bernoulli drop and a geometric delivery delay: the same task, now
    under intermittent, *stale* information.
    Stage 3 **denied** — 50% drop, a longer delay, a 120 m range gate, and a bandwidth budget
    that admits only the best-margin links: the full charter §8 regime, where the swarm must
    degrade gracefully rather than collapse.

    Each stage promotes on sustained ``mean_reward`` (two consecutive iterations above the bar),
    so the ladder never advances on one lucky iteration."""
    return CurriculumSpec(
        name="comms_ladder",
        stages=(
            StageSpec(
                name="clear",
                comms=CommsModelConfig(),
                advance=AdvanceRule(metric="mean_reward", threshold=-0.05, patience=2),
            ),
            StageSpec(
                name="lossy",
                comms=CommsModelConfig(
                    drop=DropConfig(probability=0.3),
                    delay=DelayConfig(kind="geometric", mean_ticks=1.0, max_ticks=4),
                ),
                advance=AdvanceRule(metric="mean_reward", threshold=-0.10, patience=2),
            ),
            StageSpec(
                name="denied",
                comms=CommsModelConfig(
                    range_gate=RangeGateConfig(max_range_m=120.0),
                    bandwidth=BandwidthBudgetConfig(
                        per_agent_bits_per_tick=1.0, message_bits=1.0, priority="margin_db"
                    ),
                    drop=DropConfig(probability=0.5),
                    delay=DelayConfig(kind="geometric", mean_ticks=2.0, max_ticks=8),
                ),
                advance=AdvanceRule(metric="mean_reward", threshold=-0.20, patience=2),
            ),
        ),
    )


def randomized_comms() -> CurriculumSpec:
    """A single domain-randomized stage: a fresh comms regime every episode.

    Drop probability, mean delivery delay, and the range gate are re-sampled per episode from
    declared ranges (seeded by ``(seed, stage, episode)``, so it stays byte-reproducible), and
    an opaque ``world_provider`` knob is randomized alongside them. The policy therefore trains
    across a *distribution* of channels rather than one — the property that makes the
    comms-stress curve (``eval/comms_stress.py``) degrade gracefully instead of falling off a
    cliff outside the training point."""
    return CurriculumSpec(
        name="randomized_comms",
        stages=(
            StageSpec(
                name="randomized",
                comms=CommsModelConfig(),
                randomize=RandomizationSpec(
                    drop_probability=(0.0, 0.6),
                    delay_mean_ticks=(0.0, 3.0),
                    max_range_m=(80.0, 400.0),
                    world_provider={"terrain_roughness": (0.1, 0.9)},
                ),
                # A randomized stage has no "next" stage to earn: it runs for the whole budget.
                advance=AdvanceRule(metric="mean_reward", threshold=0.0, patience=1),
            ),
        ),
    )
