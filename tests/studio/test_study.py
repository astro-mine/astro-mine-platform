"""STUDIO-02 — the trade study: Pareto ranking, determinism, multi-fidelity."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from astro_mine.core.objective import MetricDirection, ObjectiveDocument
from astro_mine.core.units.model import PlanetaryCRS
from astro_mine.studio.designspace import build_trade_study, run_trade_study
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
from astro_mine.studio.orchestrate import (
    LOCAL_STAND_IN_EVALUATOR_ID,
    InMemoryResultCache,
    SiblingClients,
)

_MOON = PlanetaryCRS(body="MOON", body_fixed_frame="MOON_ME", reference_radius_m=1737400.0)
_SPACE = DecisionSpace(
    assets=[AssetChoice(sadf_ref="rover", max_count=6), AssetChoice(sadf_ref="relay", max_count=3)]
)


def _objective_doc(name: str = "Ice prospecting") -> ObjectiveDocument:
    """A minimal objective whose id follows its name — the shape the authoring form produces."""
    draft = IntentDraft(
        id=f"study-{name.lower().replace(' ', '-')}",
        name=name,
        author="d",
        region=GeoRegion(name="rim", crs=_MOON),
        products=[
            TargetProduct(
                criterion_id="yield", metric="yield", unit="kg/day", target=50.0, tolerance=80.0
            )
        ],
    )
    return build_objective(draft)


@pytest.fixture
def biobjective() -> ObjectiveDocument:
    # yield (maximize) vs cost (minimize) — both scale with swarm size, so bigger swarms
    # trade more yield for more cost: a genuine Pareto trade-off.
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


def test_fidelity_ladder_validates_keep_fraction() -> None:
    with pytest.raises(ValidationError):
        FidelityLadder(keep_fraction=0.0)
    with pytest.raises(ValidationError):
        FidelityLadder(keep_fraction=1.5)


def test_run_trade_study_produces_pareto_ranked_candidates(
    biobjective: ObjectiveDocument, clients: SiblingClients
) -> None:
    study = run_trade_study(
        biobjective,
        _SPACE,
        clients=clients,
        seeds=(7,),
        population=8,
        generations=2,
        cache=InMemoryResultCache(),
    )
    assert isinstance(study, TradeStudy)
    assert study.evaluated  # candidates were evaluated
    assert study.pareto_front  # a non-empty Pareto front
    evaluated_ids = {ec.candidate.id for ec in study.evaluated}
    assert set(study.pareto_front) <= evaluated_ids
    assert study.provenance.seed == 7
    assert study.backend == "nsga2"


def test_run_trade_study_reproduces_the_same_pareto_front(
    biobjective: ObjectiveDocument, clients: SiblingClients
) -> None:
    def run() -> TradeStudy:
        return run_trade_study(
            biobjective, _SPACE, clients=clients, seeds=(7,), population=8, generations=2
        )

    a, b = run(), run()
    assert a.pareto_front == b.pareto_front  # determinism gate: identical front
    assert a.digest() == b.digest()


def test_multi_fidelity_prunes_then_escalates(
    biobjective: ObjectiveDocument, clients: SiblingClients
) -> None:
    vectors = [(1, 0), (2, 0), (3, 0), (4, 0)]  # 4 distinct feasible candidates
    survivors = _evaluate_generation(
        vectors,
        biobjective,
        _SPACE,
        clients,
        seed=0,
        fidelity=FidelityLadder(cheap_steps=1, full_steps=3, keep_fraction=0.5),
        cache=None,
    )
    assert len(survivors) == 2  # ceil(0.5 * 4) escalated to full fidelity


def test_study_skips_guard_rejected_candidates(
    biobjective: ObjectiveDocument, clients: SiblingClients
) -> None:
    # decoding a positive count for an 'unsafe' asset flags the candidate; Guard rejects it.
    unsafe_space = DecisionSpace(assets=[AssetChoice(sadf_ref="unsafe", max_count=3)])
    survivors = _evaluate_generation(
        [(0,), (2,)],
        biobjective,
        unsafe_space,
        clients,
        seed=0,
        fidelity=FidelityLadder(),
        cache=None,
    )
    assert len(survivors) == 1  # only the count-0 (feasible) candidate survives
    assert survivors[0].candidate.decision_vector == {"unsafe": 0.0}


def test_backend_is_swappable_per_study(
    biobjective: ObjectiveDocument, clients: SiblingClients
) -> None:
    study = run_trade_study(
        biobjective,
        _SPACE,
        clients=clients,
        backend="nsga2-lhs",
        seeds=(1,),
        population=6,
        generations=1,
    )
    assert study.backend == "nsga2-lhs" and study.evaluated


# --------------------------------------------------------------------------- #
# The evaluator identity (#42) and the study id (#43)
# --------------------------------------------------------------------------- #


def test_study_records_which_evaluator_produced_its_numbers(
    biobjective: ObjectiveDocument, clients: SiblingClients
) -> None:
    """A stand-in-scored front and a physics-scored front look identical on screen, so the artifact
    has to carry the difference — Bench's `Scorecard.runner` lesson (gap report G1.1)."""
    study = run_trade_study(biobjective, _SPACE, clients=clients, seeds=(1,), generations=1)
    assert study.evaluator == LOCAL_STAND_IN_EVALUATOR_ID


def test_the_evaluator_is_part_of_the_study_identity(
    biobjective: ObjectiveDocument, clients: SiblingClients
) -> None:
    """Not decoration: two otherwise-identical studies scored by different evaluators are different
    artifacts, so a digest cannot be reused across a change of what produced the numbers."""
    study = run_trade_study(biobjective, _SPACE, clients=clients, seeds=(1,), generations=1)
    physics = study.model_copy(update={"evaluator": "astro-mine-sim/0.1.0"})
    assert physics.digest() != study.digest()


def test_build_trade_study_requires_an_evaluator() -> None:
    """No default. A bundle that forgot to say what it was would publish a study claiming whatever
    the default said — which is the defect, not the fix."""
    with pytest.raises(TypeError, match="evaluator"):
        build_trade_study([], _objective_doc(), backend="batch", seeds=[0])  # type: ignore[call-arg]


def test_study_id_derives_from_the_objective_the_user_authored(
    biobjective: ObjectiveDocument, clients: SiblingClients
) -> None:
    """Every UI-launched study used to be called `study`, so the picker could not tell two apart."""
    study = run_trade_study(biobjective, _SPACE, clients=clients, seeds=(1,), generations=1)
    assert study.id == biobjective.objective.id
    assert study.id != "study"


def test_two_differently_named_studies_get_different_ids(clients: SiblingClients) -> None:
    first = run_trade_study(
        _objective_doc("Lunar polar water-ice"), _SPACE, clients=clients, seeds=(1,), generations=1
    )
    second = run_trade_study(
        _objective_doc("Cabeus survey"), _SPACE, clients=clients, seeds=(1,), generations=1
    )
    assert first.id != second.id


def test_a_front_containing_everything_is_reported_as_degenerate(
    biobjective: ObjectiveDocument, clients: SiblingClients
) -> None:
    """The stand-in makes every metric a positive multiple of swarm size, so nothing dominates
    anything and the front always contains every candidate. A surface must say that rather than
    draw it as a result."""
    study = run_trade_study(biobjective, _SPACE, clients=clients, seeds=(1,), generations=1)
    assert study.front_is_degenerate is (
        len(study.pareto_front) == len({ec.candidate.id for ec in study.evaluated})
    )

    partial = study.model_copy(update={"pareto_front": study.pareto_front[:1]})
    assert partial.front_is_degenerate is (len({ec.candidate.id for ec in study.evaluated}) == 1)
