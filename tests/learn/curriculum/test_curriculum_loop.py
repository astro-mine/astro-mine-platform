"""Training *through* a curriculum (learn.md §3, §11) — needs the [rllib] extra.

A curriculum nothing consumes is a document, not a feature. This is the integration gate:
``run_curriculum`` drives a real baseline up a real ladder, and the invariant that makes a
curriculum a curriculum must hold — **the world gets harder while the learner carries over**.
If a promotion rebuilt the trainer, the swarm would forget the easy stage it just mastered and
the whole exercise would be pointless.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

pytestmark = pytest.mark.ray

from astro_mine.learn import TrainConfig, make_swarm_env  # noqa: E402
from astro_mine.learn.curriculum import (  # noqa: E402
    AdvanceRule,
    CurriculumSpec,
    Stage,
    StagedCurriculum,
    StageSpec,
    comms_ladder,
    run_curriculum,
    stage_metrics,
)
from astro_mine.learn.envs import CommsModel, CommsModelConfig, DropConfig  # noqa: E402
from astro_mine.learn.track import InMemoryBackend, TrackedRun  # noqa: E402
from tests.learn.fakes import FakeSwarmWorld, build_assets  # noqa: E402


def _stage_env(stage: Stage):
    """The StageEnvFactory: Learn hands the caller a resolved stage; the caller builds the world
    with that stage's (possibly domain-randomized) comms regime."""
    return make_swarm_env(FakeSwarmWorld(), build_assets(), comms_model=CommsModel(stage.comms))


def _always_promotes() -> CurriculumSpec:
    # A rule that is satisfied by any reward the fake world produces (they are small negatives),
    # so the ladder advances every iteration and the test stays fast.
    rule = AdvanceRule(metric="mean_reward", threshold=-10.0, patience=1)
    return CurriculumSpec(
        name="fast_ladder",
        stages=(
            StageSpec(name="clear", comms=CommsModelConfig(), advance=rule),
            StageSpec(
                name="lossy",
                comms=CommsModelConfig(drop=DropConfig(probability=0.4)),
                advance=rule,
            ),
            StageSpec(
                name="denied",
                comms=CommsModelConfig(drop=DropConfig(probability=0.9)),
                advance=rule,
            ),
        ),
    )


def _curriculum(spec: CurriculumSpec | None = None, seed: int = 0) -> StagedCurriculum:
    return StagedCurriculum(
        spec if spec is not None else _always_promotes(),
        seed=seed,
        base=TrainConfig(seed=seed, rollout_steps=6, hidden_sizes=(16, 16)),
    )


def test_a_baseline_trains_up_the_whole_ladder() -> None:
    report, trainer = run_curriculum("mappo", _curriculum(), _stage_env, iterations=4)
    assert report.curriculum == "fast_ladder"
    assert report.algorithm == "mappo"
    assert len(report.metrics) == 4
    # It really climbed: iteration 0 in stage 0, then promoted each time (and held at the top).
    assert report.stage_at_iteration == [0, 1, 2, 2]
    assert report.stages_reached == 3
    assert report.completed is True
    assert len(trainer.learning_curve()) == 4
    # Each promotion recorded a content-addressed, replayable stage.
    assert [s["stage_name"] for s in report.stages] == ["clear", "lossy", "denied"]
    assert all(s["content_hash"].startswith("sha256:") for s in report.stages)


def test_the_learner_carries_over_across_a_promotion() -> None:
    # THE invariant. A promotion swaps the WORLD, not the policy: the same net objects (and the
    # same optimizer state) must survive, or the curriculum is just three unrelated short runs.
    curriculum = _curriculum()
    stage0 = curriculum.stage()
    trainer_env = _stage_env(stage0)
    from astro_mine.learn.algos import default_registry

    trainer = default_registry().get("ippo").make_trainer(trainer_env, stage0.train_config)
    nets_before = {agent: id(net) for agent, net in trainer._nets.items()}
    optimizer_before = id(trainer._opt)
    weight_before = trainer._nets["rover"].value_head.weight.detach().clone()

    trainer.train_iteration()
    curriculum.update({"mean_reward": 0.0})
    trainer.set_env(_stage_env(curriculum.stage(episode=1)))  # the promotion
    trainer.train_iteration()

    import torch

    assert {a: id(n) for a, n in trainer._nets.items()} == nets_before  # same net objects
    assert id(trainer._opt) == optimizer_before  # same optimizer (momentum survives)
    # ... and it kept LEARNING across the boundary rather than resetting to init.
    assert not torch.equal(trainer._nets["rover"].value_head.weight.detach(), weight_before)
    assert len(trainer.learning_curve()) == 2


def test_the_stage_env_actually_gets_harder() -> None:
    # The comms regime the trainer sees must be the stage's — otherwise the ladder is cosmetic.
    seen: list[float] = []

    def recording_factory(stage: Stage):
        seen.append(stage.comms.drop.probability)
        return _stage_env(stage)

    run_curriculum("ippo", _curriculum(), recording_factory, iterations=3)
    assert seen == [0.0, 0.4, 0.9]  # clear -> lossy -> denied


def test_an_env_swap_that_changes_the_tensor_contract_fails_loudly() -> None:
    # A curriculum changes the DIFFICULTY, not the world. Swapping in a different agent set would
    # silently feed garbage into nets sized for the old one.
    from astro_mine.learn.algos import default_registry

    curriculum = _curriculum()
    trainer = (
        default_registry()
        .get("ippo")
        .make_trainer(_stage_env(curriculum.stage()), curriculum.stage().train_config)
    )
    assets = build_assets()
    del assets["relay"]

    class TwoAgentWorld(FakeSwarmWorld):
        @property
        def possible_agents(self):
            return ("rover", "digger")

        @property
        def agents(self):
            return tuple(a for a in self._active if a != "relay")

    with pytest.raises(ValueError, match="agent set"):
        trainer.set_env(make_swarm_env(TwoAgentWorld(), assets))


def test_a_curriculum_run_is_reproducible() -> None:
    def run() -> list[dict[str, float]]:
        report, _ = run_curriculum("ippo", _curriculum(seed=5), _stage_env, iterations=3)
        return report.metrics

    assert run() == run()  # same seed ⇒ same ladder over the same sampled worlds


def test_stage_metrics_split_the_curve_by_difficulty() -> None:
    report, _ = run_curriculum("ippo", _curriculum(), _stage_env, iterations=4)
    per_stage = stage_metrics(report)
    assert sorted(per_stage) == [0, 1, 2]
    assert len(per_stage[2]) == 2  # it held on the final stage for the last two iterations


def test_a_curriculum_run_is_tracked_end_to_end() -> None:
    # The two Phase-1 modules meet: the curriculum drives the world, the tracked run captures
    # every stage + iteration into one content-addressed record.
    backend = InMemoryBackend()
    spec = comms_ladder()
    config = TrainConfig(seed=1, rollout_steps=6, hidden_sizes=(16, 16))
    curriculum = StagedCurriculum(_always_promotes(), seed=1, base=config)

    with TrackedRun(config, algorithm="mappo", curriculum=spec, backend=backend) as run:
        report, trainer = run_curriculum("mappo", curriculum, _stage_env, iterations=3, track=run)
        run.log_export(trainer.export(), digests={"rover": "sha256:deadbeef"})

    assert backend.ended
    assert len(backend.curve("mean_reward")) == 3
    assert backend.curve("curriculum_stage") == [0.0, 1.0, 2.0]
    # Every promotion left a replayable artifact, and the produced policy is linked by digest.
    assert "curriculum/stage_0.json" in backend.artifacts
    assert "curriculum/stage_2.json" in backend.artifacts
    assert backend.artifacts["policy_export.json"]["onnx_digests"] == {"rover": "sha256:deadbeef"}
    assert report.stages_reached == 3
