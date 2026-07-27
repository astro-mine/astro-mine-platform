"""Learned-DEM surrogate models (RM-P1-SURR-02) — the torch layer.

The granular/excavation GNS: a deep-ensemble message-passing network trained on the frozen DEM
fixture, with split-conformal calibrated bounds and an enforced trust region, wrapped as a
:class:`~astro_mine.surrogate.model.SurrogateModel`. Importing this package pulls torch/numpy;
the contract layer (:mod:`astro_mine.surrogate.model`/``report``/``manifest``) never imports it.
"""

from __future__ import annotations

from astro_mine.surrogate.models.excavation import (
    ExcavationSurrogate,
    build_excavation_surrogate,
)
from astro_mine.surrogate.models.train import TrainConfig

__all__ = ["ExcavationSurrogate", "TrainConfig", "build_excavation_surrogate"]
