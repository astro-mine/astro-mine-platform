"""STUDIO-07 — reproducibility-by-construction: provenance contract + determinism gate."""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from astro_mine.core.objective import ObjectiveDocument
from astro_mine.core.units.model import PlanetaryCRS
from astro_mine.studio.campaign import author_campaign
from astro_mine.studio.designspace import run_trade_study
from astro_mine.studio.intent import capture_intent
from astro_mine.studio.intent.llm import LLMDraftResult, LLMUsage, accept_draft, draft_objective
from astro_mine.studio.models import (
    AssetChoice,
    CampaignPhase,
    DecisionSpace,
    DesignCandidate,
    GeoRegion,
    IntentDraft,
    TargetProduct,
)
from astro_mine.studio.orchestrate import SiblingClients, evaluate_candidate
from astro_mine.studio.provenance import (
    ArtifactProvenance,
    capture_provenance,
    environment_fingerprint,
)
from astro_mine.studio.reproducibility import (
    ReproducibilityError,
    assert_reproducible,
    is_reproducible,
    missing_provenance_fields,
)
from astro_mine.studio.workspace import InMemoryWorkspace

_MOON = PlanetaryCRS(body="MOON", body_fixed_frame="MOON_ME", reference_radius_m=1737400.0)
_SPACE = DecisionSpace(assets=[AssetChoice(sadf_ref="rover", max_count=6)])


# ---- environment fingerprint + provenance envelope ------------------------ #


def test_environment_fingerprint_is_deterministic_and_prefixed() -> None:
    fingerprint = environment_fingerprint()
    assert fingerprint.startswith("sha256:")
    assert fingerprint == environment_fingerprint()


def test_capture_provenance_populates_env_lockfile_by_default() -> None:
    assert capture_provenance(input_hashes=["sha256:h"]).env_lockfile == environment_fingerprint()
    assert (
        capture_provenance(input_hashes=["sha256:h"], env_lockfile="explicit").env_lockfile
        == "explicit"
    )


# ---- the reproducibility contract ----------------------------------------- #


def _complete() -> ArtifactProvenance:
    return capture_provenance(input_hashes=["sha256:h"], seed=1)


def test_complete_provenance_is_reproducible() -> None:
    provenance = _complete()
    assert missing_provenance_fields(provenance) == []
    assert is_reproducible(provenance, require_seed=True)
    assert_reproducible(provenance, require_seed=True)  # does not raise


@pytest.mark.parametrize(
    ("update", "field"),
    [
        ({"input_hashes": []}, "input_hashes"),
        ({"core_interface_versions": {}}, "core_interface_versions"),
        ({"code_version": None}, "code_version"),
        ({"toolchain_version": None}, "toolchain_version"),
        ({"env_lockfile": None}, "env_lockfile"),
    ],
)
def test_missing_field_is_detected(update: dict[str, Any], field: str) -> None:
    provenance = _complete().model_copy(update=update)
    assert field in missing_provenance_fields(provenance)
    assert not is_reproducible(provenance)
    with pytest.raises(ReproducibilityError, match=field):
        assert_reproducible(provenance)


def test_seed_required_only_for_computational_artifacts() -> None:
    no_seed = capture_provenance(input_hashes=["sha256:h"])  # e.g. an authored ObjectiveSpec
    assert is_reproducible(no_seed)  # fine without a seed
    assert not is_reproducible(no_seed, require_seed=True)  # a computation must carry one


# ---- every produced artifact carries complete provenance ------------------ #


@given(
    products=st.lists(
        st.tuples(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=8),
            st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=4,
    )
)
def test_captured_objective_is_reproducible(products: list[tuple[str, float]]) -> None:
    draft = IntentDraft(
        id="o",
        name="n",
        author="a",
        region=GeoRegion(name="r", crs=_MOON),
        products=[
            TargetProduct(
                criterion_id=f"c{i}", metric=metric, unit="kg", target=target, tolerance=1.0
            )
            for i, (metric, target) in enumerate(products)
        ],
    )
    captured = capture_intent(draft, workspace=InMemoryWorkspace())
    assert_reproducible(captured.provenance)  # objectives carry provenance, no seed


def test_full_chain_is_reproducible(
    objective_doc: ObjectiveDocument, clients: SiblingClients
) -> None:
    study = run_trade_study(
        objective_doc, _SPACE, clients=clients, seeds=(3,), population=4, generations=1
    )
    assert_reproducible(study.provenance, require_seed=True)
    for evaluated in study.evaluated:
        assert_reproducible(evaluated.provenance, require_seed=True)
    chosen = evaluate_candidate(
        DesignCandidate(id="c", swarm=[]), objective_doc, clients=clients, seed=3
    )
    campaign = author_campaign(
        objective_doc, chosen, name="c", phases=[CampaignPhase(id="p", name="P")], trade_study=study
    )
    assert_reproducible(campaign.provenance, require_seed=True)


# ---- the CI determinism gate ---------------------------------------------- #


def test_seeded_trade_study_reproduces_the_identical_pareto_front(
    objective_doc: ObjectiveDocument, clients: SiblingClients
) -> None:
    def run() -> tuple[list[str], str]:
        study = run_trade_study(
            objective_doc, _SPACE, clients=clients, seeds=(11,), population=6, generations=2
        )
        return study.pareto_front, study.digest()

    first, second = run(), run()
    assert first == second  # CI fails on non-reproducibility


# ---- LLM authorship is recorded, without a live model --------------------- #


class _StubProvider:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def draft(
        self, intent_text: str, *, schema: Any, system: str, model: str, config: Any
    ) -> LLMDraftResult:
        return LLMDraftResult(payload=dict(self.payload), model=model, usage=LLMUsage())


def test_llm_drafted_spec_records_model_and_is_reproducible() -> None:
    payload = {
        "objective_version": "0.1",
        "objective": {
            "id": "ice",
            "name": "Lunar ice",
            "success_criteria": [
                {
                    "id": "w",
                    "binding": {
                        "metric": "water_rate",
                        "unit": "kg/day",
                        "direction": "higher_better",
                        "target": 40.0,
                        "tolerance": 10.0,
                    },
                }
            ],
        },
    }
    draft = draft_objective("Produce water", provider=_StubProvider(payload))
    ws = InMemoryWorkspace()
    captured = accept_draft(draft, workspace=ws, author="designer")
    assert ws.audit()[0].model == draft.model  # audit records the drafting model/version
    assert_reproducible(captured.provenance)  # the artifact stands on its content hashes
