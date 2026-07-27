"""The trade study (studio.md §3 ``TradeStudy``, §8 multi-fidelity).

Drives a multi-objective search over a ``DecisionSpace``, delegating every candidate
*evaluation* to the STUDIO-03 design loop — Studio does the search and the Pareto math and
nothing else (studio.md §2 principle 1). **Multi-fidelity** keeps expensive evaluations
few: each generation is first evaluated at a cheap fidelity (few sim steps) to prune, then
only the promising survivors escalate to full fidelity. Reproducible-by-construction: the
DoE seed drives the search, the loop is deterministic, and pruning is a stable sort — so a
seeded re-run reproduces the identical Pareto front (the STUDIO-02 CI determinism gate).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

from pydantic import Field

from astro_mine.core.objective import MetricDirection, ObjectiveDocument, ObjectiveSpec, to_wire

from .._base import StudioModel
from ..hashing import content_hash, content_hash_json
from ..models import DecisionSpace, EvaluatedCandidate, TradeStudy
from ..orchestrate import ResultCache, SiblingClients, evaluate_candidate
from ..orchestrate.clients import GuardRejection
from ..provenance import capture_provenance
from .encode import decode, encode
from .pareto import non_dominated_sort
from .search import get_backend


class FidelityLadder(StudioModel):
    """The multi-fidelity schedule: evaluate a generation at ``cheap_steps``, keep the top
    ``keep_fraction`` by aggregate score, and re-evaluate only those at ``full_steps``."""

    cheap_steps: int = Field(default=2, ge=1)
    full_steps: int = Field(default=8, ge=1)
    keep_fraction: float = Field(default=0.5, gt=0.0, le=1.0)


def _senses(spec: ObjectiveSpec) -> tuple[bool, ...]:
    return tuple(
        criterion.binding.direction is MetricDirection.HIGHER_BETTER
        for criterion in spec.success_criteria
    )


def _objective_vector(scores: Mapping[str, float], spec: ObjectiveSpec) -> tuple[float, ...]:
    return tuple(scores.get(criterion.binding.metric, 0.0) for criterion in spec.success_criteria)


def _evaluate_generation(
    vectors: Sequence[tuple[int, ...]],
    objective: ObjectiveDocument,
    space: DecisionSpace,
    clients: SiblingClients,
    *,
    seed: int,
    fidelity: FidelityLadder,
    cache: ResultCache | None,
) -> list[EvaluatedCandidate]:
    """Cheap-prune then escalate one generation. Guard-rejected candidates are infeasible
    and dropped. The cheap tier is un-cached (its result is a different fidelity than the
    study's content-addressed cache key implies); only full-fidelity results are cached."""
    candidates = {(candidate := decode(vector, space)).id: candidate for vector in vectors}

    cheap: list[EvaluatedCandidate] = []
    for candidate in candidates.values():
        try:
            cheap.append(
                evaluate_candidate(
                    candidate,
                    objective,
                    clients=clients,
                    seed=seed,
                    max_steps=fidelity.cheap_steps,
                    cache=None,
                )
            )
        except GuardRejection:
            continue

    cheap.sort(key=lambda evaluated: (-evaluated.score.aggregate, evaluated.candidate.id))
    keep = math.ceil(fidelity.keep_fraction * len(cheap)) if cheap else 0

    full: list[EvaluatedCandidate] = []
    for evaluated in cheap[:keep]:
        try:
            full.append(
                evaluate_candidate(
                    candidates[evaluated.candidate.id],
                    objective,
                    clients=clients,
                    seed=seed,
                    max_steps=fidelity.full_steps,
                    cache=cache,
                )
            )
        except GuardRejection:  # pragma: no cover - cheap tier already certified it
            continue
    return full


def study_id_for(objective: ObjectiveDocument) -> str:
    """The id a study takes from the objective it was run against.

    The objective already carries both an id and a human name — the authoring form writes
    ``study-lunar-polar-water-ice`` / ``"Lunar polar water-ice"`` — so a study never needs to fall
    back to a constant. It used to: every UI-launched study was literally named ``study``, which
    made the picker unusable the moment a user had two of them, and named a published campaign's
    provenance after nothing."""
    return objective.objective.id


def build_trade_study(
    evaluated: Iterable[EvaluatedCandidate],
    objective: ObjectiveDocument,
    *,
    backend: str,
    evaluator: str,
    seeds: Sequence[int],
    study_id: str | None = None,
    extra_input_hashes: Sequence[str] = (),
    engine_versions: Mapping[str, str] | None = None,
) -> TradeStudy:
    """Assemble a reproducible :class:`TradeStudy` from already-evaluated candidates — the Pareto
    math and provenance, and nothing else (studio.md §2 principle 1). Shared by the DSE search
    (:func:`run_trade_study`) and the ``POST /studies`` route, which evaluates a *given* candidate
    list rather than searching a ``DecisionSpace`` but needs the identical reproducible artifact the
    comparison view consumes.

    ``evaluator`` is required and has no default: it records what produced the metric values, and a
    default would let a stand-in-scored study quietly claim whatever the default said. ``study_id``
    defaults to :func:`study_id_for` rather than to a constant."""
    spec = objective.objective
    senses = _senses(spec)
    ordered = sorted(evaluated, key=lambda candidate: (candidate.candidate.id, candidate.seed))
    points = [_objective_vector(candidate.score.metric_scores, spec) for candidate in ordered]
    front = non_dominated_sort(points, senses)[0] if points else []
    objective_hash = content_hash(to_wire(objective))
    provenance = capture_provenance(
        input_hashes=sorted({objective_hash, *extra_input_hashes}),
        seed=seeds[0] if seeds else 0,
        engine_versions=engine_versions,
    )
    return TradeStudy(
        id=study_id if study_id is not None else study_id_for(objective),
        objective_hash=objective_hash,
        backend=backend,
        evaluator=evaluator,
        seeds=list(seeds),
        evaluated=list(ordered),
        pareto_front=[ordered[i].candidate.id for i in front],
        provenance=provenance,
    )


def run_trade_study(
    objective: ObjectiveDocument,
    space: DecisionSpace,
    *,
    clients: SiblingClients,
    backend: str = "nsga2",
    seeds: Sequence[int] = (0,),
    population: int = 8,
    generations: int = 2,
    fidelity: FidelityLadder | None = None,
    cache: ResultCache | None = None,
    engine_versions: Mapping[str, str] | None = None,
    study_id: str | None = None,
) -> TradeStudy:
    """Run a multi-objective, multi-fidelity trade study and return the evaluated
    candidates + their Pareto-ranked front."""
    ladder = fidelity if fidelity is not None else FidelityLadder()
    spec = objective.objective
    senses = _senses(spec)
    engine = get_backend(backend)
    base_seed = seeds[0]

    vectors = engine.initial(space, seed=base_seed, n=population)
    evaluated: dict[str, EvaluatedCandidate] = {}
    for generation in range(generations):
        survivors = _evaluate_generation(
            vectors, objective, space, clients, seed=base_seed, fidelity=ladder, cache=cache
        )
        for candidate in survivors:
            evaluated[candidate.candidate.id] = candidate
        ranked = [
            (
                encode(candidate.candidate, space),
                _objective_vector(candidate.score.metric_scores, spec),
            )
            for candidate in survivors
        ]
        vectors = engine.evolve(
            ranked, senses, space, seed=base_seed + generation + 1, n=population
        )

    return build_trade_study(
        evaluated.values(),
        objective,
        backend=backend,
        evaluator=clients.evaluator,
        seeds=seeds,
        study_id=study_id,
        extra_input_hashes=[content_hash_json(space.model_dump(mode="json"))],
        engine_versions=engine_versions,
    )
