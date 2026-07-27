"""The shipped reference ``TrainConfig`` (#24).

Learn shipped no reference config and no emitted schema, so a newcomer's first config had to be
reverse-engineered from the Pydantic model. These tests pin that the shipped document stays valid
against the model it claims to instance — the two are coupled by nothing else.
"""

from __future__ import annotations

import json

from jsonschema import Draft202012Validator

from astro_mine.learn.algos import TrainConfig
from astro_mine.learn.reference import TRAIN_CONFIG_FILE, load_train_config, train_config_path


def test_the_reference_config_loads_into_a_train_config() -> None:
    assert isinstance(load_train_config(), TrainConfig)


def test_the_reference_config_validates_against_the_emitted_schema() -> None:
    """AC3 — the document and the model cannot drift apart silently."""
    document = json.loads(train_config_path().read_text(encoding="utf-8"))
    Draft202012Validator(TrainConfig.model_json_schema()).validate(document)


def test_the_reference_config_is_readable_from_the_installed_package() -> None:
    path = train_config_path()
    assert path.name == TRAIN_CONFIG_FILE
    assert path.is_file()


def test_the_reference_config_sets_only_real_fields() -> None:
    """``TrainConfig`` forbids extras, so a stale key would fail loudly — assert it directly."""
    document = json.loads(train_config_path().read_text(encoding="utf-8"))
    assert set(document) <= set(TrainConfig.model_fields)


def test_the_reference_config_is_a_replaceable_example_not_a_behaviour_change() -> None:
    """Every field it sets already matches the tier-1 default, so pointing at it changes nothing."""
    assert load_train_config() == TrainConfig()
