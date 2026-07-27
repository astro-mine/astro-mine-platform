"""Campaign authoring + Ops hand-off (RM-P1-STUDIO-04).

Author a chosen, scored ``EvaluatedCandidate`` into a ``Campaign`` (timeline + phases +
contingency branches) and freeze it as a content-addressed **hand-off bundle** that Ops
consumes **unchanged** in Phase 2 — hand-off, don't fork (studio.md §3, §5, §9). Studio
computes nothing here: it holds and structures state, runs no operations loop and no flight
path. No LLM on this path — a campaign is built only from validated candidate artifacts.
This is the M1.1 exit artifact: goal-in → scored-design-out → ``Campaign``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence

from pydantic import Field

from astro_mine.core.objective import ObjectiveDocument, to_wire

from .._base import FrozenStudioModel
from ..crs_schema import core_units_schema_digest
from ..hashing import canonical_json, content_hash
from ..models import (
    Campaign,
    CampaignPhase,
    ContingencyBranch,
    EvaluatedCandidate,
    TradeStudy,
)
from ..provenance import capture_provenance
from ..workspace import WorkspaceStore

__all__ = [
    "CampaignBundle",
    "author_campaign",
    "freeze_campaign",
    "handoff",
    "load_campaign",
]


class CampaignBundle(FrozenStudioModel):
    """The frozen, content-addressed artifact packaged for Ops (studio.md §5 lifecycle).
    ``payload`` is the canonical bytes Ops reads; ``digest`` is their content hash and the
    campaign's stable identity. ``schema_digest`` records the Core units-vocabulary version
    the campaign was authored against, *alongside* — never inside — the hashed payload, so it
    does not perturb ``digest`` yet lets a rehydrated campaign prove its vocabulary version
    (RM-P1-STUDIO-08, RFC-0007 §3)."""

    digest: str
    campaign: Campaign
    schema_digest: str = Field(default_factory=core_units_schema_digest)

    def payload(self) -> bytes:
        return canonical_json(self.campaign.model_dump(mode="json"))


def author_campaign(
    objective: ObjectiveDocument,
    chosen: EvaluatedCandidate,
    *,
    name: str,
    phases: Sequence[CampaignPhase],
    contingencies: Sequence[ContingencyBranch] = (),
    trade_study: TradeStudy | None = None,
    campaign_id: str = "campaign",
    world_ref: str | None = None,
    engine_versions: Mapping[str, str] | None = None,
) -> Campaign:
    """Author a validating campaign from a chosen candidate, stamping provenance + lineage
    (the trade study / candidate the design was chosen from) so it is reproducible from its
    lineage."""
    objective_hash = content_hash(to_wire(objective))
    lineage = [chosen.digest(), objective_hash]
    trade_study_ref: str | None = None
    if trade_study is not None:
        trade_study_ref = trade_study.id
        lineage.append(trade_study.digest())

    provenance = capture_provenance(
        input_hashes=sorted(lineage), seed=chosen.seed, engine_versions=engine_versions
    )
    return Campaign(
        id=campaign_id,
        name=name,
        objective_hash=objective_hash,
        chosen=chosen,
        phases=list(phases),
        contingencies=list(contingencies),
        trade_study_ref=trade_study_ref,
        # Carried forward, not merely referenced: a reviewer pulling this campaign by digest can see
        # what justified it without also fetching the trade study it came from.
        evaluator=trade_study.evaluator if trade_study is not None else None,
        world_ref=world_ref,
        provenance=provenance,
    )


def freeze_campaign(campaign: Campaign) -> CampaignBundle:
    """Freeze a (validated) campaign into a content-addressed hand-off bundle. Once frozen,
    the artifact's identity is its digest (studio.md §5 — drafts are mutable; a handed-off
    campaign is frozen and content-addressed)."""
    payload = canonical_json(campaign.model_dump(mode="json"))
    return CampaignBundle(
        digest=content_hash(payload),
        campaign=campaign,
        schema_digest=core_units_schema_digest(),
    )


def handoff(campaign: Campaign, *, workspace: WorkspaceStore, author: str) -> CampaignBundle:
    """Freeze and persist the campaign as the Ops hand-off bundle (content-addressed,
    fail-closed, audited)."""
    bundle = freeze_campaign(campaign)
    workspace.put(
        "campaign",
        bundle.digest,
        bundle.payload(),
        author=author,
        metadata={"campaign_id": campaign.id, "units_schema_digest": bundle.schema_digest},
    )
    return bundle


def load_campaign(payload: bytes) -> Campaign:
    """Re-load a campaign from a bundle's bytes and re-validate it — the Phase-2 Ops
    consumer reads the same Core-shaped artifact Studio produced, with **no translation
    layer and no re-derivation** (studio.md §9)."""
    return Campaign.model_validate(json.loads(payload))
