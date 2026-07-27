"""STUDIO-03 — the per-candidate design loop and the content-addressed result cache."""

from __future__ import annotations

import dataclasses

import pytest

from astro_mine.core.objective import ObjectiveDocument
from astro_mine.studio.models import AssetSelection, DesignCandidate
from astro_mine.studio.orchestrate import (
    InMemoryResultCache,
    ResultCache,
    SiblingClients,
    cache_key,
    evaluate_candidate,
    objective_content_hash,
)
from astro_mine.studio.orchestrate.clients import GuardRejection, LocalSwarmComposer


class _CountingComposer:
    def __init__(self) -> None:
        self.calls = 0
        self._inner = LocalSwarmComposer()

    def compose(self, candidate: DesignCandidate) -> tuple[str, ...]:
        self.calls += 1
        return self._inner.compose(candidate)


# ---- cache ---------------------------------------------------------------- #


def test_cache_key_is_deterministic_and_sensitive(
    candidate: DesignCandidate, objective_doc: ObjectiveDocument
) -> None:
    k1 = cache_key(candidate, objective_doc, 1)
    assert k1 == cache_key(candidate, objective_doc, 1)
    assert k1 != cache_key(candidate, objective_doc, 2)  # seed-sensitive
    other = DesignCandidate(id="other", swarm=[AssetSelection(sadf_ref="r", count=1)])
    assert k1 != cache_key(other, objective_doc, 1)  # candidate-sensitive


def test_in_memory_result_cache_roundtrip(
    candidate: DesignCandidate, objective_doc: ObjectiveDocument, clients: SiblingClients
) -> None:
    cache = InMemoryResultCache()
    assert isinstance(cache, ResultCache)
    key = cache_key(candidate, objective_doc, 0)
    assert not cache.has(key)
    evaluated = evaluate_candidate(candidate, objective_doc, clients=clients, seed=0)
    cache.put(key, evaluated)
    assert cache.has(key) and cache.get(key) is evaluated


# ---- loop ----------------------------------------------------------------- #


def test_evaluate_candidate_runs_all_steps_with_provenance(
    candidate: DesignCandidate, objective_doc: ObjectiveDocument, clients: SiblingClients
) -> None:
    evaluated = evaluate_candidate(
        candidate, objective_doc, clients=clients, seed=7, engine_versions={"sim": "0.1"}
    )
    assert evaluated.seed == 7
    assert evaluated.candidate is candidate
    assert evaluated.world_ref in evaluated.provenance.input_hashes
    assert objective_content_hash(objective_doc.objective) in evaluated.provenance.input_hashes
    assert evaluated.provenance.engine_versions == {"sim": "0.1"}
    assert set(evaluated.score.metric_scores) == {"water_production_rate", "power_margin"}


def test_evaluate_candidate_is_deterministic(
    candidate: DesignCandidate, objective_doc: ObjectiveDocument, clients: SiblingClients
) -> None:
    a = evaluate_candidate(candidate, objective_doc, clients=clients, seed=3)
    b = evaluate_candidate(candidate, objective_doc, clients=clients, seed=3)
    assert a.digest() == b.digest()


def test_evaluate_candidate_uses_cache(
    candidate: DesignCandidate, objective_doc: ObjectiveDocument, clients: SiblingClients
) -> None:
    counting = _CountingComposer()
    spied = dataclasses.replace(clients, composer=counting)
    cache = InMemoryResultCache()
    first = evaluate_candidate(candidate, objective_doc, clients=spied, seed=0, cache=cache)
    second = evaluate_candidate(candidate, objective_doc, clients=spied, seed=0, cache=cache)
    assert first == second
    assert counting.calls == 1  # second evaluation is served from the cache, not recomputed


def test_evaluate_candidate_propagates_guard_rejection(
    objective_doc: ObjectiveDocument, clients: SiblingClients
) -> None:
    unsafe = DesignCandidate(
        id="unsafe", swarm=[AssetSelection(sadf_ref="r", count=1)], decision_vector={"unsafe": 1.0}
    )
    cache = InMemoryResultCache()
    with pytest.raises(GuardRejection):
        evaluate_candidate(unsafe, objective_doc, clients=clients, seed=0, cache=cache)
    assert not cache.has(cache_key(unsafe, objective_doc, 0))  # nothing cached on rejection
