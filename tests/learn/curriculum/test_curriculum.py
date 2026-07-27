"""Staged curricula, domain randomization, and the plugin seam (learn.md §3, §11).

learn.md §11 recommends "hand-authored staged curricula + domain randomization for the MVP, with
an automatic-curriculum plugin interface for research", and learn.md §3 defines the contract:
"produce a (possibly stateful) stream of env configs; ``update(metrics)`` advances staged or
automatic curricula based on observed performance".

What has to be true:

- **progression** — a stage promotes only on *sustained* performance (never one lucky iteration),
  and the ladder ends cleanly;
- **randomization is reproducible** — the same seed replays the same sampled worlds, or the
  curriculum would destroy the CX-REPRO guarantee it rides on (conventions.md §5);
- **the plugin seam is real** — an *automatic* curriculum (the Phase-2 deferral) registers and
  runs through the same interface, with no change to Learn.

Torch-free except the one integration test that trains through a ladder (marked ``ray``).
"""

from __future__ import annotations

import pytest

from astro_mine.core.hashing import content_hash_json
from astro_mine.learn.algos.config import TrainConfig
from astro_mine.learn.curriculum import (
    CURRICULUM_ENTRY_POINT_GROUP,
    AdvanceRule,
    Curriculum,
    CurriculumRegistry,
    CurriculumSpec,
    DomainRandomizer,
    RandomizationSpec,
    StagedCurriculum,
    StageSpec,
    comms_ladder,
    default_curriculum_registry,
    randomized_comms,
)
from astro_mine.learn.envs.comms import CommsModelConfig, DropConfig


def _ladder(**advance) -> CurriculumSpec:
    rule = AdvanceRule(metric="mean_reward", threshold=0.0, **advance)
    return CurriculumSpec(
        name="test_ladder",
        stages=(
            StageSpec(name="easy", advance=rule),
            StageSpec(
                name="hard",
                comms=CommsModelConfig(drop=DropConfig(probability=0.5)),
                advance=rule,
            ),
        ),
    )


# --- the declarative document --------------------------------------------------------


def test_curriculum_is_a_validated_json_schema_document() -> None:
    spec = comms_ladder()
    # Like TrainConfig / CommsModelConfig: a Pydantic v2 doc that round-trips and is hashable —
    # so it is part of the run's reproducibility key (conventions.md §3, §5).
    assert spec.model_json_schema()["title"] == "CurriculumSpec"
    assert CurriculumSpec.model_validate_json(spec.model_dump_json()) == spec
    assert content_hash_json(spec.model_dump(mode="json")).startswith("sha256:")
    assert spec.schema_version == "0.1.0"


def test_a_typo_or_an_empty_curriculum_fails_loudly() -> None:
    with pytest.raises(ValueError, match="at least one stage"):
        CurriculumSpec(name="empty", stages=())
    with pytest.raises(ValueError, match="duplicate stage names"):
        CurriculumSpec(name="dupes", stages=(StageSpec(name="a"), StageSpec(name="a")))
    with pytest.raises(ValueError):  # extra="forbid": a typo'd key never silently disables a stage
        StageSpec(name="s", commms=CommsModelConfig())
    with pytest.raises(ValueError, match="low <= high"):
        RandomizationSpec(drop_probability=(0.8, 0.2))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        RandomizationSpec(drop_probability=(0.0, 1.5))


def test_the_shipped_ladder_degrades_the_channel_monotonically() -> None:
    # The pedagogy: each stage is strictly harder than the last (that is what a ladder IS).
    stages = comms_ladder().stages
    assert [s.name for s in stages] == ["clear", "lossy", "denied"]
    assert stages[0].comms.is_identity  # stage 1 degrades nothing
    drops = [s.comms.drop.probability for s in stages]
    assert drops == sorted(drops) and drops[-1] > drops[0]
    assert stages[-1].comms.range_gate.max_range_m is not None  # the full charter §8 regime
    assert stages[-1].comms.bandwidth.per_agent_bits_per_tick is not None


# --- progression ---------------------------------------------------------------------


def test_a_stage_promotes_only_on_sustained_performance() -> None:
    curriculum = StagedCurriculum(_ladder(patience=2), seed=0)
    assert curriculum.stage_index == 0

    assert curriculum.update({"mean_reward": 1.0}) is False  # one good iteration is not enough
    assert curriculum.stage_index == 0
    assert curriculum.update({"mean_reward": -1.0}) is False  # a bad one resets the streak
    assert curriculum.update({"mean_reward": 1.0}) is False
    assert curriculum.update({"mean_reward": 1.0}) is True  # two IN A ROW ⇒ promoted
    assert curriculum.stage_index == 1
    assert curriculum.stage_spec.name == "hard"


def test_a_missing_metric_never_promotes() -> None:
    curriculum = StagedCurriculum(_ladder(), seed=0)
    assert curriculum.update({"policy_loss": 0.1}) is False  # no evidence ⇒ no promotion
    assert curriculum.stage_index == 0


def test_min_iterations_holds_a_stage_even_when_the_bar_is_cleared() -> None:
    curriculum = StagedCurriculum(_ladder(patience=1, min_iterations=3), seed=0)
    assert curriculum.update({"mean_reward": 5.0}) is False
    assert curriculum.update({"mean_reward": 5.0}) is False
    assert curriculum.update({"mean_reward": 5.0}) is True


def test_the_final_stage_completes_rather_than_promoting() -> None:
    curriculum = StagedCurriculum(_ladder(), seed=0)
    curriculum.update({"mean_reward": 1.0})  # -> stage 1
    assert curriculum.stage_index == 1 and not curriculum.done
    assert curriculum.update({"mean_reward": 1.0}) is False  # no stage 2 to promote to
    assert curriculum.done is True
    assert curriculum.stage_index == 1  # ... and it stays runnable on the last stage
    assert curriculum.stage().name == "hard"
    assert [h["stage_name"] for h in curriculum.history] == ["easy", "hard"]


def test_le_mode_advances_on_a_falling_metric() -> None:
    rule = AdvanceRule(metric="td_loss", threshold=0.1, mode="le")
    assert rule.satisfied(0.05) and not rule.satisfied(0.5)
    curriculum = StagedCurriculum(
        CurriculumSpec(
            name="loss_ladder",
            stages=(StageSpec(name="a", advance=rule), StageSpec(name="b", advance=rule)),
        ),
        seed=0,
    )
    assert curriculum.update({"td_loss": 0.9}) is False
    assert curriculum.update({"td_loss": 0.01}) is True


# --- the resolved stage --------------------------------------------------------------


def test_a_stage_resolves_the_base_config_plus_its_overrides() -> None:
    # learn.md §2.7: "the fidelity dial is a curriculum axis, not an afterthought" — a stage can
    # train cheaply and a later one validate at high fidelity.
    spec = CurriculumSpec(
        name="fidelity_ladder",
        stages=(
            StageSpec(name="cheap", config_overrides={"fidelity": "surrogate"}),
            StageSpec(name="validate", config_overrides={"fidelity": "sim_high"}),
        ),
    )
    base = TrainConfig(seed=11, rollout_steps=64, hidden_sizes=(64, 64))
    curriculum = StagedCurriculum(spec, seed=11, base=base)

    stage = curriculum.stage()
    assert stage.train_config.fidelity == "surrogate"
    assert stage.train_config.rollout_steps == 64  # the base config carries through
    assert stage.train_config.hidden_sizes == (64, 64)
    assert stage.train_config.seed == 11

    curriculum.update({"mean_reward": 1.0})
    assert curriculum.stage().train_config.fidelity == "sim_high"


def test_a_resolved_stage_is_content_addressed() -> None:
    curriculum = StagedCurriculum(comms_ladder(), seed=3)
    stage = curriculum.stage(episode=2)
    assert stage.content_hash().startswith("sha256:")
    assert (
        stage.content_hash()
        == StagedCurriculum(comms_ladder(), seed=3).stage(episode=2).content_hash()
    )
    # Two episodes of a randomized stage are DIFFERENT worlds and must hash differently.
    randomized = StagedCurriculum(randomized_comms(), seed=3)
    assert randomized.stage(episode=0).content_hash() != randomized.stage(episode=1).content_hash()
    assert "comms" in stage.provenance() and "train_config" in stage.provenance()


# --- domain randomization ------------------------------------------------------------


def test_domain_randomization_samples_every_declared_knob() -> None:
    spec = RandomizationSpec(
        drop_probability=(0.2, 0.6),
        delay_mean_ticks=(1.0, 3.0),
        max_range_m=(100.0, 200.0),
        world_provider={"roughness": (0.0, 1.0)},
    )
    comms, world = DomainRandomizer(spec, seed=1).sample(CommsModelConfig(), episode=0)
    assert 0.2 <= comms.drop.probability <= 0.6
    assert comms.delay.kind == "geometric" and 1.0 <= comms.delay.mean_ticks <= 3.0
    assert comms.range_gate.max_range_m is not None
    assert 100.0 <= comms.range_gate.max_range_m <= 200.0
    assert 0.0 <= world["roughness"] <= 1.0


def test_an_undeclared_knob_is_not_randomized() -> None:
    base = CommsModelConfig(drop=DropConfig(probability=0.33))
    spec = RandomizationSpec(max_range_m=(50.0, 60.0))  # only the range gate
    comms, world = DomainRandomizer(spec, seed=0).sample(base, episode=0)
    assert comms.drop.probability == 0.33  # untouched
    assert comms.delay.kind == "none"
    assert world == {}


def test_an_empty_randomization_is_the_identity() -> None:
    base = CommsModelConfig(drop=DropConfig(probability=0.5))
    spec = RandomizationSpec()
    assert spec.is_empty
    comms, world = DomainRandomizer(spec, seed=0).sample(base, episode=3)
    assert comms is base and world == {}


def test_randomization_is_seed_reproducible_and_episode_varying() -> None:
    # The invariant that keeps a randomized curriculum inside CX-REPRO: same seed ⇒ the same
    # SEQUENCE of sampled worlds; a different seed ⇒ a different one.
    spec = RandomizationSpec(drop_probability=(0.0, 1.0))

    def draws(seed: int) -> list[float]:
        randomizer = DomainRandomizer(spec, seed=seed)
        return [
            randomizer.sample(CommsModelConfig(), episode=e)[0].drop.probability for e in range(6)
        ]

    assert draws(7) == draws(7)  # replayable
    assert draws(7) != draws(8)  # and actually random
    assert len(set(draws(7))) > 1  # episodes differ (it is not one value repeated)


def test_the_shipped_randomized_curriculum_varies_the_channel_per_episode() -> None:
    curriculum = StagedCurriculum(randomized_comms(), seed=2)
    drops = [curriculum.stage(episode=e).comms.drop.probability for e in range(5)]
    assert len(set(drops)) > 1
    assert all(0.0 <= d <= 0.6 for d in drops)
    # It also randomizes the opaque world_provider selector Learn does not interpret.
    assert "terrain_roughness" in curriculum.stage(episode=0).world_provider


# --- the plugin / registry seam ------------------------------------------------------


def test_registry_lists_and_builds_the_builtin_curricula() -> None:
    registry = default_curriculum_registry()
    assert registry.names() == ["comms_ladder", "randomized_comms"]
    assert "comms_ladder" in registry and "nope" not in registry
    assert len(registry) == 2
    assert list(registry) == registry.names()

    built = registry.build("comms_ladder", seed=4)
    assert isinstance(built, Curriculum)
    assert isinstance(built, StagedCurriculum)
    assert built.spec.name == "comms_ladder"
    assert registry.spec("comms_ladder").stages[0].name == "clear"
    with pytest.raises(KeyError):
        registry.build("nonexistent")


def test_entry_point_group_is_the_documented_plugin_seam() -> None:
    assert CURRICULUM_ENTRY_POINT_GROUP == "astro_mine.learn.curricula"
    assert default_curriculum_registry().discover_entry_points() == []  # none installed in tests


def test_a_third_party_staged_curriculum_registers_by_spec(monkeypatch) -> None:
    from astro_mine.learn.curriculum import registry as registry_mod

    custom = CurriculumSpec(name="thirdparty", stages=(StageSpec(name="only"),))

    class _EP:
        name = "thirdparty"

        def load(self):
            return lambda: custom

    monkeypatch.setattr(registry_mod, "entry_points", lambda *, group: [_EP()])
    registry = CurriculumRegistry(builtins=False)
    assert registry.discover_entry_points() == ["thirdparty"]
    assert registry.spec("thirdparty") == custom
    assert registry.build("thirdparty", seed=0).spec.name == "thirdparty"


def test_an_automatic_curriculum_plugs_into_the_same_interface(monkeypatch) -> None:
    # THE Phase-2 seam (learn.md §11 "automatic-curriculum plugin interface for research"): a
    # PLR/teacher-student curriculum is just a Curriculum whose update() picks the next stage
    # from performance instead of walking a ladder. Nothing in Learn changes to accept it.
    from astro_mine.learn.curriculum import registry as registry_mod

    spec = CurriculumSpec(
        name="auto", stages=(StageSpec(name="s0"), StageSpec(name="s1"), StageSpec(name="s2"))
    )

    class RegretCurriculum:
        """A toy 'automatic' curriculum: jump to the stage the swarm is doing WORST on."""

        def __init__(self, seed: int, base: TrainConfig) -> None:
            self._spec = spec
            self._index = 0
            self._base = base
            self._seed = seed

        @property
        def spec(self):
            return self._spec

        @property
        def stage_index(self) -> int:
            return self._index

        @property
        def done(self) -> bool:
            return False

        def stage(self, *, episode: int = 0):
            return StagedCurriculum(self._spec, seed=self._seed, base=self._base).stage(
                episode=episode
            )

        def update(self, metrics) -> bool:
            # Regret-driven: a bad reward means "this is too hard, go easier"; a good one, harder.
            target = min(self._index + 1, 2) if metrics["mean_reward"] > 0 else 0
            advanced = target != self._index
            self._index = target
            return advanced

    class _EP:
        name = "auto"

        def load(self):
            return lambda: RegretCurriculum

    monkeypatch.setattr(registry_mod, "entry_points", lambda *, group: [_EP()])
    registry = CurriculumRegistry(builtins=False)
    assert registry.discover_entry_points() == ["auto"]

    curriculum = registry.build("auto", seed=0)
    assert isinstance(curriculum, Curriculum)  # satisfies the SAME protocol
    assert curriculum.update({"mean_reward": 1.0}) is True and curriculum.stage_index == 1
    assert curriculum.update({"mean_reward": -1.0}) is True and curriculum.stage_index == 0
    # A programmatic curriculum has no declarative spec to hand out.
    with pytest.raises(KeyError, match="no declarative CurriculumSpec"):
        registry.spec("auto")
