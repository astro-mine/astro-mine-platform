"""Sealed GroundTruthField + Bayesian BeliefField updating.

A sealed ground-truth realization sampled from a prior, and a belief field updated
from an ordered observation log into a replayable prior -> posterior chain.

- :class:`GroundTruthField` / :func:`sample_ground_truth` — the scenario's fixed, seeded truth
  (consumed by Sim, never reached through the belief path; prospect.md §9). Realized per-cell
  independently or through a **spatially-correlated** GMRF/generative backend
  (:data:`REALIZATION_KINDS`).
- :class:`BeliefField` — the agents' posterior, conditioned on an ordered
  :class:`FieldObservation` log; :meth:`BeliefField.update` returns a new, content-addressed,
  replayable posterior, conditioning each reading under its own
  :mod:`~astro_mine.prospect.sensors` likelihood.
- :class:`FieldObservation` / :func:`load_observations` — the typed sensor return and the CSV feed
  that drives belief updates.

Backlog: RM-P0-PROSPECT-04, RM-P1-PROSPECT-10 —
https://github.com/astro-mine/astro-mine-prospect/issues/4
"""

from __future__ import annotations

from astro_mine.prospect.belief.field import BeliefField
from astro_mine.prospect.belief.ground_truth import (
    DEFAULT_REALIZATION,
    REALIZATION_KINDS,
    GroundTruthField,
    sample_ground_truth,
)
from astro_mine.prospect.belief.observation import FieldObservation, load_observations
from astro_mine.prospect.belief.seam import (
    CELL_ID_FORMAT,
    GriddedBelief,
    belief_from_bundle,
    cell_id,
)

__all__ = [
    "CELL_ID_FORMAT",
    "DEFAULT_REALIZATION",
    "REALIZATION_KINDS",
    "BeliefField",
    "FieldObservation",
    "GriddedBelief",
    "GroundTruthField",
    "belief_from_bundle",
    "cell_id",
    "load_observations",
    "sample_ground_truth",
]
