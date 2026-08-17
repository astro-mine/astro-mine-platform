# SPDX-License-Identifier: Apache-2.0
"""Astro-Mine-Studio — the design front door (goal-in, design-out).

Phase-1 library slices against the Core narrow waist:

- **intent capture** (RM-P1-STUDIO-01) — the deterministic, no-LLM ``intent`` path turns a
  stated goal + assets into a Core-validated ``ObjectiveSpec``. The **optional** LLM adapter
  (RM-P1-STUDIO-05, ``intent.llm``) drafts specs a human reviews and Core validates; it is
  fully removable and never required.
- **the design loop** (RM-P1-STUDIO-03) — ``orchestrate`` fans one ``DesignCandidate``
  through the autonomy stack (compose → world → condition → allocate → certify → simulate →
  score) as durable/cancelable/resumable jobs.
- **design-space exploration** (RM-P1-STUDIO-02) — ``designspace`` runs a pluggable,
  multi-fidelity, multi-objective trade study that Pareto-ranks ``DesignCandidate``s.
- **campaign hand-off** (RM-P1-STUDIO-04) — ``campaign`` authors a chosen design into a
  ``Campaign`` and freezes it as the content-addressed artifact Ops consumes unchanged.
- **reproducibility-by-construction** (RM-P1-STUDIO-07) — every produced artifact carries a
  provenance envelope (:func:`~astro_mine.studio.reproducibility.assert_reproducible`) so a
  seeded re-run reproduces the same Pareto front.

Studio computes nothing and imports no sibling package: it *sequences* Core-contract calls
on content-addressed artifacts (studio.md §2). The importable library is the core; the
:mod:`~astro_mine.studio.api` FastAPI app is a deployment of it (principle 8). Local-first
(conventions.md §7 tier-1): everything here runs on one workstation with no cluster.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .campaign import author_campaign, handoff
from .designspace import run_trade_study
from .intent import CapturedObjective, MetricVocabulary, capture_intent
from .models import (
    Campaign,
    DecisionSpace,
    DesignCandidate,
    EvaluatedCandidate,
    IntentDraft,
    TradeStudy,
)
from .orchestrate import evaluate_candidate, local_clients, run_batch
from .reproducibility import assert_reproducible
from .workspace import InMemoryWorkspace

try:
    __version__ = version("astro-mine-platform")
except PackageNotFoundError:  # pragma: no cover - source tree without installed metadata
    __version__ = "0.0.0"

__all__ = [
    "Campaign",
    "CapturedObjective",
    "DecisionSpace",
    "DesignCandidate",
    "EvaluatedCandidate",
    "InMemoryWorkspace",
    "IntentDraft",
    "MetricVocabulary",
    "TradeStudy",
    "__version__",
    "assert_reproducible",
    "author_campaign",
    "capture_intent",
    "evaluate_candidate",
    "handoff",
    "local_clients",
    "run_batch",
    "run_trade_study",
]
