"""The shipped reference training configuration — a copy-pasteable starting point.

Learn shipped no reference config and no emitted schema, so a newcomer's first
:class:`~astro_mine.learn.algos.TrainConfig` had to be reverse-engineered from the Pydantic
model. Mind ships six reference stacks and thirteen manifests; Guard ships an anchor spec; this
is Learn's equivalent — one small, valid document a user can point ``--config-json`` at and then
edit.

It is a **replaceable example, not a privileged path** (conventions.md §1.3). Every field it sets
is already a tier-1-friendly default, so it changes no behaviour — its value is that it exists,
is discoverable, and shows which knobs are worth reaching for. The fields it omits
(``world_provider``, ``surrogate_validation_threshold``) are the ones a quickstart should not
need.

The canonical schema is the ``TrainConfig`` model itself; emit it with
:meth:`~pydantic.BaseModel.model_json_schema` (the pattern the curriculum, comms and
comms-stress configs already document). A test validates this document against that schema, so
the two cannot drift.
"""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path

from astro_mine.learn.algos import TrainConfig

__all__ = [
    "TRAIN_CONFIG_FILE",
    "load_train_config",
    "train_config_path",
]

#: The packaged reference config, resolved through :mod:`importlib.resources` so it is readable
#: from an installed wheel and not only from a source checkout.
TRAIN_CONFIG_FILE = "train_config.json"


def train_config_path() -> Path:
    """The on-disk path of the reference config — what ``--config-json`` takes."""
    return Path(str(files(__package__).joinpath(TRAIN_CONFIG_FILE)))


def load_train_config() -> TrainConfig:
    """Load and validate the reference config into a :class:`TrainConfig`."""
    text = files(__package__).joinpath(TRAIN_CONFIG_FILE).read_text(encoding="utf-8")
    return TrainConfig.model_validate_json(text)
