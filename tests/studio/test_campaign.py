"""STUDIO-04 — campaign authoring + content-addressed Ops hand-off."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from astro_mine.core.objective import ObjectiveDocument
from astro_mine.studio.campaign import (
    CampaignBundle,
    author_campaign,
    freeze_campaign,
    handoff,
    load_campaign,
)
from astro_mine.studio.designspace import run_trade_study
from astro_mine.studio.models import (
    AssetChoice,
    Campaign,
    CampaignPhase,
    ContingencyBranch,
    DecisionSpace,
    DesignCandidate,
    EvaluatedCandidate,
)
from astro_mine.studio.orchestrate import SiblingClients, evaluate_candidate
from astro_mine.studio.workspace import InMemoryWorkspace

_SPACE = DecisionSpace(assets=[AssetChoice(sadf_ref="rover", max_count=6)])
_PHASES = [CampaignPhase(id="deploy", name="Deploy"), CampaignPhase(id="prospect", name="Prospect")]
_CONTINGENCY = ContingencyBranch(
    id="comms-loss", trigger="link_lost", phases=[CampaignPhase(id="safe", name="Safe hold")]
)


@pytest.fixture
def chosen(objective_doc: ObjectiveDocument, clients: SiblingClients) -> EvaluatedCandidate:
    candidate = DesignCandidate(id="cand-a", swarm=[])
    return evaluate_candidate(candidate, objective_doc, clients=clients, seed=7)


def test_contingency_branch_requires_a_phase() -> None:
    with pytest.raises(ValidationError):
        ContingencyBranch(id="c", trigger="t", phases=[])


def test_author_campaign_records_lineage(
    objective_doc: ObjectiveDocument, clients: SiblingClients, chosen: EvaluatedCandidate
) -> None:
    study = run_trade_study(
        objective_doc, _SPACE, clients=clients, seeds=(7,), population=4, generations=1
    )
    campaign = author_campaign(
        objective_doc,
        chosen,
        name="Lunar ice",
        phases=_PHASES,
        contingencies=[_CONTINGENCY],
        trade_study=study,
    )
    assert isinstance(campaign, Campaign)
    assert campaign.trade_study_ref == study.id
    assert campaign.chosen is chosen
    assert campaign.contingencies[0].id == "comms-loss"
    # lineage: chosen digest + objective hash + trade-study digest
    assert len(campaign.provenance.input_hashes) == 3
    assert campaign.provenance.seed == 7


def test_author_campaign_without_trade_study(
    objective_doc: ObjectiveDocument, chosen: EvaluatedCandidate
) -> None:
    campaign = author_campaign(objective_doc, chosen, name="c", phases=_PHASES)
    assert campaign.trade_study_ref is None
    assert len(campaign.provenance.input_hashes) == 2  # chosen + objective only


def test_author_campaign_requires_a_phase(
    objective_doc: ObjectiveDocument, chosen: EvaluatedCandidate
) -> None:
    with pytest.raises(ValidationError):
        author_campaign(objective_doc, chosen, name="c", phases=[])


def test_handoff_roundtrips_with_no_translation(
    objective_doc: ObjectiveDocument, chosen: EvaluatedCandidate
) -> None:
    # the M1.1 exit artifact: chosen design -> validating Campaign -> content-addressed bundle
    campaign = author_campaign(
        objective_doc, chosen, name="Lunar ice", phases=_PHASES, contingencies=[_CONTINGENCY]
    )
    bundle = freeze_campaign(campaign)
    assert isinstance(bundle, CampaignBundle)
    assert bundle.digest == campaign.digest()  # frozen + content-addressed
    # a Phase-2 Ops consumer re-loads and re-validates with no translation
    assert load_campaign(bundle.payload()) == campaign


def test_handoff_persists_content_addressed_and_audited(
    objective_doc: ObjectiveDocument, chosen: EvaluatedCandidate
) -> None:
    campaign = author_campaign(objective_doc, chosen, name="c", phases=_PHASES)
    ws = InMemoryWorkspace()
    bundle = handoff(campaign, workspace=ws, author="designer")
    assert ws.has(bundle.digest)
    assert ws.get(bundle.digest) == bundle.payload()
    entry = ws.audit()[0]
    assert entry.artifact_type == "campaign" and entry.author == "designer"
    # the hand-off records the Core units-vocabulary version alongside the content hash
    assert entry.metadata == {
        "campaign_id": campaign.id,
        "units_schema_digest": bundle.schema_digest,
    }
    assert bundle.schema_digest.startswith("sha256:")


def test_campaign_records_the_world_it_was_inspected_against(
    objective_doc: ObjectiveDocument, chosen: EvaluatedCandidate
) -> None:
    """UC-F5's artifact half: a reviewer pulling the campaign can tell which terrain the design was
    inspected on, rather than inferring it. ``None`` when it never was — a legitimate state."""
    with_world = author_campaign(
        objective_doc,
        chosen,
        name="c",
        phases=_PHASES,
        world_ref="shackleton-de-gerlache:0.4.0",
    )
    assert with_world.world_ref == "shackleton-de-gerlache:0.4.0"

    without = author_campaign(objective_doc, chosen, name="c", phases=_PHASES)
    assert without.world_ref is None
