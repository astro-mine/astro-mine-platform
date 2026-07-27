"""CX-REPRO determinism gate for the baselines (RM-P1-LEARN-03) — needs the [rllib] extra.

Same scenario + same seed + same pinned config ⇒ the *same* learning curve (issue AC;
conventions.md §5, §11; learn.md §2.4). CI fails on non-reproducibility. Mirrors
``tests/envs/test_determinism.py`` for the training loop: two trainers built with the
identical config produce byte-identical per-iteration metrics (the golden series), and a
different seed produces a different curve so the gate actually bites.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

pytestmark = pytest.mark.ray

from astro_mine.learn.algos import TrainConfig, default_registry  # noqa: E402

BASELINES = ["ippo", "mappo", "qmix", "comms_ppo"]


def _train(tag: str, config: TrainConfig, env_factory) -> list[dict[str, float]]:
    trainer = default_registry().get(tag).make_trainer(env_factory(), config)
    return [trainer.train_iteration() for _ in range(config.iterations)]


@pytest.mark.parametrize("tag", BASELINES)
def test_same_seed_yields_identical_learning_curve(tag, env_factory) -> None:
    config = TrainConfig(seed=42, iterations=3, rollout_steps=8, hidden_sizes=(16, 16))
    first = _train(tag, config, env_factory)
    second = _train(tag, config, env_factory)
    assert first == second  # byte-identical metrics across the whole run


@pytest.mark.parametrize("tag", BASELINES)
def test_different_seed_changes_the_curve(tag, env_factory) -> None:
    base = TrainConfig(seed=1, iterations=2, rollout_steps=8, hidden_sizes=(16, 16))
    other = base.model_copy(update={"seed": 2})
    assert _train(tag, base, env_factory) != _train(tag, other, env_factory)


def test_repro_key_is_the_config_plus_comms_plus_seed(comms_env_factory) -> None:
    # The reproducibility key recorded with a run: TrainConfig (JSON) + CommsModelConfig + seed.
    config = TrainConfig(seed=7, iterations=2, rollout_steps=8, hidden_sizes=(16, 16))
    assert config.model_dump_json()  # serializable → recorded verbatim in provenance
    first = _train("ippo", config, comms_env_factory)
    second = _train("ippo", config, comms_env_factory)
    assert first == second
