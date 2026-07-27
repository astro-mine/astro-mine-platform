"""Adversarial falsification harness — the central validation strategy for the shield (GUARD-05).

Untrusted, stdlib-only tooling that *attacks the trusted core from outside* (guard.md §10; issue
#5): a seeded, reproducible search over policy actions and disturbances that tries to drive a
hard-constraint violation, closing the loop through a minimal in-repo double-integrator plant on the
anchor scenario. It is never part of the TCB — it only reads the shield's public ``decide`` output
and verdict stream — and it carries no numpy / Sim dependency, so it runs in the ordinary CI pytest
job as a determinism gate.

Public surface:

- :mod:`~astro_mine.guard.falsify.rollout` — :func:`shielded_rollout` / :func:`control_rollout`, the
  :class:`PlantState` / :class:`RolloutStep` records, and the :class:`AdversaryPolicy` adapter;
- :mod:`~astro_mine.guard.falsify.adversary` — the :class:`Adversary` contract and the
  :class:`SeededAdversary` / :class:`WorstCaseAdversary` attackers (+ the anchor start helpers);
- :mod:`~astro_mine.guard.falsify.oracle` — :func:`shielded_violations` (empty ⇒ the shield held) /
  :func:`control_violations` (non-empty ⇒ the search is real), the :class:`Violation` record, and
  the reusable safe-set predicates :func:`keepout_barrier` / :func:`scalar_violations`.
"""

from __future__ import annotations

from astro_mine.guard.falsify.adversary import (
    ANCHOR_SAFE_SIGNALS,
    Adversary,
    SeededAdversary,
    WorstCaseAdversary,
    anchor_initial_state,
)
from astro_mine.guard.falsify.oracle import (
    Violation,
    control_violations,
    keepout_barrier,
    scalar_violations,
    shielded_violations,
)
from astro_mine.guard.falsify.rollout import (
    DEFAULT_DT,
    DEFAULT_U_MAX,
    AdversaryPolicy,
    PlantState,
    RolloutStep,
    control_rollout,
    shielded_rollout,
)

__all__ = [
    "ANCHOR_SAFE_SIGNALS",
    "DEFAULT_DT",
    "DEFAULT_U_MAX",
    "Adversary",
    "AdversaryPolicy",
    "PlantState",
    "RolloutStep",
    "SeededAdversary",
    "Violation",
    "WorstCaseAdversary",
    "anchor_initial_state",
    "control_rollout",
    "control_violations",
    "keepout_barrier",
    "scalar_violations",
    "shielded_rollout",
    "shielded_violations",
]
