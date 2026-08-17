# SPDX-License-Identifier: Apache-2.0
"""The per-candidate design loop (studio.md §6) — turn one ``DesignCandidate`` into a
scored ``EvaluatedCandidate`` by sequencing the seven delegated steps.

Studio only *sequences* the calls on content-addressed artifacts; no physics, solver,
learning, or scoring lives here (studio.md §2 principle 1). Guard's certification
(step 5) can veto a candidate — a :class:`~.clients.GuardRejection` propagates and the
candidate is neither simulated nor scored. Every evaluation records a provenance envelope
(input hashes + Core/engine versions + seed) so it reproduces (studio.md §6).
"""

from __future__ import annotations

from collections.abc import Mapping

from astro_mine.core.objective import ObjectiveDocument

from ..models import DesignCandidate, EvaluatedCandidate
from ..provenance import capture_provenance
from .cache import ResultCache, cache_key
from .clients import SiblingClients, objective_content_hash


def evaluate_candidate(
    candidate: DesignCandidate,
    objective: ObjectiveDocument,
    *,
    clients: SiblingClients,
    seed: int,
    max_steps: int = 8,
    cache: ResultCache | None = None,
    engine_versions: Mapping[str, str] | None = None,
) -> EvaluatedCandidate:
    """Fan one candidate through compose → world → condition → allocate → certify →
    simulate → score. Returns a cached result for an identical ``(design, world, seed)``
    tuple instead of re-evaluating."""
    key = cache_key(candidate, objective, seed)
    if cache is not None and cache.has(key):
        return cache.get(key)

    spec = objective.objective
    agents = clients.composer.compose(candidate)  # 1. compose SADF swarm
    env, world_ref = clients.environment.instantiate(  # 2. instantiate world
        candidate, spec, agents=agents, seed=seed
    )
    policy = clients.conditioner.condition(candidate, spec, seed=seed)  # 3. condition
    policy = clients.allocator.allocate(policy, candidate, spec)  # 4. allocate
    policy = clients.guard.certify(policy, candidate, spec)  # 5. guard-wrap + certify
    episode = clients.simulator.rollout(  # 6. simulate
        env, policy, spec, world_ref, seed=seed, max_steps=max_steps
    )
    score = clients.scorer.score(episode, spec)  # 7. score (Bench)

    provenance = capture_provenance(
        input_hashes=sorted([candidate.digest(), objective_content_hash(spec), world_ref]),
        seed=seed,
        engine_versions=engine_versions,
    )
    evaluated = EvaluatedCandidate(
        candidate=candidate,
        score=score,
        seed=seed,
        world_ref=world_ref,
        provenance=provenance,
    )
    if cache is not None:
        cache.put(key, evaluated)
    return evaluated
