"""STUDIO-02 — the Optuna DSE backend: registry, determinism, and drop-in study equivalence.

Exercises the Optuna backend the same way ``test_search.py``/``test_study.py`` exercise the
built-in NSGA-II — same assertions, same fidelity ladder — because the whole point of
RM-P1-STUDIO-02 is that an external engine is a *drop-in* ``SearchBackend``, not a parallel
code path.
"""

from __future__ import annotations

import sys

import pytest

from astro_mine.core.objective import MetricDirection, ObjectiveDocument
from astro_mine.core.units.model import PlanetaryCRS
from astro_mine.studio.designspace import (
    DEFERRED_BACKENDS,
    MissingBackendExtra,
    SearchBackend,
    deferred_backends,
    get_backend,
    registered_backends,
    run_trade_study,
)
from astro_mine.studio.designspace.study import FidelityLadder, _evaluate_generation
from astro_mine.studio.intent.forms import build_objective
from astro_mine.studio.models import (
    AssetChoice,
    DecisionSpace,
    GeoRegion,
    IntentDraft,
    TargetProduct,
    TradeStudy,
)
from astro_mine.studio.orchestrate import InMemoryResultCache, SiblingClients

optuna = pytest.importorskip("optuna", reason="the [optuna] extra is not installed")

from astro_mine.studio.designspace.optuna_backend import (  # noqa: E402  (after importorskip)
    OptunaBackend,
    _generation_key,
)

_MOON = PlanetaryCRS(body="MOON", body_fixed_frame="MOON_ME", reference_radius_m=1737400.0)
_SPACE = DecisionSpace(
    assets=[AssetChoice(sadf_ref="rover", max_count=6), AssetChoice(sadf_ref="relay", max_count=4)]
)
_BACKENDS = ["optuna", "optuna-nsga2", "optuna-tpe"]

# A biobjective population: maximize both. The elite corner is high-x0 / high-x1.
_POPULATION = [(6, 4), (5, 4), (6, 3), (0, 0), (1, 0), (0, 1), (2, 2), (3, 1)]
_RANKED = [(vector, (float(vector[0]), float(vector[1]))) for vector in _POPULATION]
_SENSES = (True, True)


@pytest.fixture
def biobjective() -> ObjectiveDocument:
    """Yield (maximize) vs cost (minimize) — a genuine Pareto trade-off, as in test_study.py."""
    draft = IntentDraft(
        id="prospect",
        name="Ice prospecting",
        author="d",
        region=GeoRegion(name="rim", crs=_MOON),
        products=[
            TargetProduct(
                criterion_id="yield", metric="yield", unit="kg/day", target=50.0, tolerance=80.0
            ),
            TargetProduct(
                criterion_id="cost",
                metric="cost",
                unit="usd",
                direction=MetricDirection.LOWER_BETTER,
                target=0.0,
                tolerance=80.0,
            ),
        ],
    )
    return build_objective(draft)


# ---- registry ------------------------------------------------------------- #


@pytest.mark.parametrize("name", _BACKENDS)
def test_optuna_backends_are_registered(name: str) -> None:
    assert name in registered_backends()
    assert isinstance(get_backend(name), SearchBackend)


def test_missing_extra_is_raised_with_an_actionable_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without the extra the *name* still resolves — only instantiating fails, and it says how to
    fix it. (`None` in sys.modules makes the import raise, as an uninstalled extra does.)"""
    monkeypatch.setitem(sys.modules, "astro_mine.studio.designspace.optuna_backend", None)
    assert "optuna" in registered_backends()  # still discoverable

    with pytest.raises(MissingBackendExtra, match=r"--extra optuna") as caught:
        get_backend("optuna")
    assert isinstance(caught.value, ImportError)
    assert caught.value.extra == "optuna"


def test_deferred_seams_are_documented_not_dropped() -> None:
    """RM-P1-STUDIO-02: the other three studio.md §11 engines stay documented seams."""
    assert {"ax", "pymoo", "ray-tune"} <= set(DEFERRED_BACKENDS)
    assert deferred_backends() is DEFERRED_BACKENDS
    for name, note in DEFERRED_BACKENDS.items():
        assert name not in registered_backends()  # deferred, so not instantiable
        assert len(note) > 40  # a real pointer, not a placeholder


def test_get_backend_points_at_a_deferred_seam() -> None:
    with pytest.raises(ValueError, match="documented but deferred seam"):
        get_backend("pymoo")


def test_get_backend_still_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown search backend"):
        get_backend("does-not-exist")


def test_rejects_an_unknown_sampler() -> None:
    with pytest.raises(ValueError, match="unknown Optuna sampler"):
        OptunaBackend(sampler="cmaes")


@pytest.mark.parametrize("fraction", [0.0, 1.5])
def test_rejects_an_out_of_range_elite_fraction(fraction: float) -> None:
    with pytest.raises(ValueError, match="elite_fraction"):
        OptunaBackend(elite_fraction=fraction)


# ---- the SearchBackend contract (mirrors test_search.py) ------------------- #


@pytest.mark.parametrize("doe", ["sobol", "lhs"])
def test_initial_doe_is_in_bounds_and_deterministic(doe: str) -> None:
    backend = OptunaBackend(doe=doe)
    a = backend.initial(_SPACE, seed=3, n=8)
    b = backend.initial(_SPACE, seed=3, n=8)
    assert a == b  # deterministic in the seed
    assert len(a) == 8
    for vector in a:
        assert 0 <= vector[0] <= 6 and 0 <= vector[1] <= 4


@pytest.mark.parametrize("sampler", ["nsga2", "tpe"])
def test_evolve_produces_in_bounds_children(sampler: str) -> None:
    backend = OptunaBackend(sampler=sampler)
    children = backend.evolve(_RANKED, _SENSES, _SPACE, seed=1, n=16)
    assert len(children) == 16
    for vector in children:
        assert 0 <= vector[0] <= 6 and 0 <= vector[1] <= 4


@pytest.mark.parametrize("sampler", ["nsga2", "tpe"])
def test_evolve_is_deterministic_in_the_seed(sampler: str) -> None:
    """The seed policy: a re-run reproduces the proposals, and a different seed does not.

    This is the property a stateful Optuna `Study` would break — the backend rebuilds the study
    per call, re-deriving Optuna's own RNG from Studio's seed (see optuna_backend's docstring).
    """
    backend = OptunaBackend(sampler=sampler)
    first = backend.evolve(_RANKED, _SENSES, _SPACE, seed=1, n=8)
    # Same instance, same arguments -> same proposals (no hidden state carried between calls).
    assert backend.evolve(_RANKED, _SENSES, _SPACE, seed=1, n=8) == first
    # A fresh instance agrees, and a different seed disagrees.
    assert OptunaBackend(sampler=sampler).evolve(_RANKED, _SENSES, _SPACE, seed=1, n=8) == first
    assert backend.evolve(_RANKED, _SENSES, _SPACE, seed=2, n=8) != first


@pytest.mark.parametrize("sampler", ["nsga2", "tpe"])
def test_evolve_reseeds_when_population_too_small(sampler: str) -> None:
    backend = OptunaBackend(sampler=sampler)
    children = backend.evolve([((1, 1), (1.0, 1.0))], _SENSES, _SPACE, seed=2, n=4)
    assert len(children) == 4  # fewer than two ranked points -> DoE reseed, as NSGAII does
    assert children == OptunaBackend(sampler=sampler).initial(_SPACE, seed=2, n=4)


def test_evolve_honors_the_objective_senses() -> None:
    """Flipping both senses flips which corner of the space is elite, so the proposals move."""
    backend = OptunaBackend(sampler="tpe")
    maximize = backend.evolve(_RANKED, (True, True), _SPACE, seed=5, n=12)
    minimize = backend.evolve(_RANKED, (False, False), _SPACE, seed=5, n=12)
    assert maximize != minimize
    mean = lambda vs: sum(v[0] for v in vs) / len(vs)  # noqa: E731
    assert mean(maximize) > mean(minimize)  # maximizing x0 pulls toward the high corner


# ---- the CI canary: Optuna's private generation key ------------------------ #


def test_nsga2_replay_sets_the_generation_attr() -> None:
    """NSGA-II only sees replayed trials that carry its generation system-attr; TPE has none."""
    assert _generation_key(optuna.samplers.NSGAIISampler(seed=0)) == "NSGAIISampler:generation"
    assert _generation_key(optuna.samplers.TPESampler(seed=0)) is None


def test_nsga2_breeds_children_from_elite_parents() -> None:
    """The canary (see optuna_backend's docstring).

    Optuna's NSGA-II ignores replayed trials that lack its private generation system-attr and
    then **silently falls back to uniform random sampling** — a degradation no bounds/determinism
    assertion would catch. With mutation disabled, every child gene must come from a parent on
    the elite front; if the private-attr contract ever breaks, random sampling will draw a gene
    from outside it and this fails loudly.
    """
    backend = OptunaBackend(sampler="nsga2", mutation_prob=0.0, elite_fraction=0.5)
    children = backend.evolve(_RANKED, _SENSES, _SPACE, seed=1, n=12)

    # The elite half of _POPULATION under "maximize both": the non-dominated / least-crowded four.
    elite = {(6, 4), (5, 4), (6, 3), (2, 2)}
    elite_x0 = {vector[0] for vector in elite}
    elite_x1 = {vector[1] for vector in elite}
    for child in children:
        assert child[0] in elite_x0, f"{child} carries a gene no elite parent has -> random search"
        assert child[1] in elite_x1, f"{child} carries a gene no elite parent has -> random search"

    # And the population as a whole has been pulled to the elite corner (random would be ~3.0/2.0).
    assert sum(c[0] for c in children) / len(children) > 4.0
    assert sum(c[1] for c in children) / len(children) > 3.0


# ---- drop-in through the real study loop (mirrors test_study.py) ----------- #


@pytest.mark.parametrize("backend", _BACKENDS)
def test_run_trade_study_produces_pareto_ranked_candidates(
    backend: str, biobjective: ObjectiveDocument, clients: SiblingClients
) -> None:
    study = run_trade_study(
        biobjective,
        _SPACE,
        clients=clients,
        backend=backend,
        seeds=(7,),
        population=8,
        generations=2,
        cache=InMemoryResultCache(),
    )
    assert isinstance(study, TradeStudy)
    assert study.evaluated
    assert study.pareto_front
    assert set(study.pareto_front) <= {ec.candidate.id for ec in study.evaluated}
    assert study.provenance.seed == 7
    assert study.backend == backend


@pytest.mark.parametrize("backend", _BACKENDS)
def test_run_trade_study_reproduces_the_same_pareto_front(
    backend: str, biobjective: ObjectiveDocument, clients: SiblingClients
) -> None:
    """The STUDIO-02 CI determinism gate, against the external backend."""

    def run() -> TradeStudy:
        return run_trade_study(
            biobjective,
            _SPACE,
            clients=clients,
            backend=backend,
            seeds=(7,),
            population=8,
            generations=2,
        )

    a, b = run(), run()
    assert a.pareto_front == b.pareto_front
    assert a.digest() == b.digest()


def test_multi_fidelity_ladder_prunes_then_escalates_under_optuna(
    biobjective: ObjectiveDocument, clients: SiblingClients
) -> None:
    """The fidelity ladder is backend-agnostic — the proposals it prunes come from Optuna."""
    engine = get_backend("optuna")
    vectors = engine.initial(_SPACE, seed=0, n=4)
    survivors = _evaluate_generation(
        vectors,
        biobjective,
        _SPACE,
        clients,
        seed=0,
        fidelity=FidelityLadder(cheap_steps=1, full_steps=3, keep_fraction=0.5),
        cache=None,
    )
    # ceil(0.5 * n_distinct) escalated to full fidelity; the DoE may repeat a vector, so the
    # distinct-candidate count is what the ladder halves.
    distinct = len(set(vectors))
    assert len(survivors) == -(-distinct // 2)
