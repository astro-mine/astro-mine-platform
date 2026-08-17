# SPDX-License-Identifier: Apache-2.0
"""The seeded example: a full, pinned campaign so a fresh ``serve`` is never an empty screen (#32).

`serve` authors an example objective → study → campaign for the lunar polar water-ice anchor and
publishes it to the local registry **once** (idempotent: a re-run resolves the existing pin). It is
the same loop the user drives — goal in, design out — run headless, and it is **labelled as an
example** wherever it surfaces, never passed off as the user's own result (the G1.1 honesty rule).

The campaign is signed and content-addressed like any other artifact (CX-REPRO); this module holds
no crypto of its own — the injected publisher signs it.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from astro_mine.core.objective import ObjectiveDocument
from astro_mine.core.units.model import PlanetaryCRS

from .campaign import author_campaign, freeze_campaign
from .designspace import build_trade_study
from .intent.forms import build_objective
from .models import (
    AssetSelection,
    Campaign,
    CampaignPhase,
    DesignCandidate,
    GeoRegion,
    HardConstraint,
    IntentDraft,
    TargetProduct,
    TradeStudy,
)
from .orchestrate import (
    InMemoryJobStore,
    InMemoryResultCache,
    JobStatus,
    LocalDispatcher,
    local_clients,
    run_batch,
)

if TYPE_CHECKING:  # pragma: no cover - the publisher lives behind the [hub] extra
    from .hub import ArtifactPublisher

#: The example campaign's immutable ``name:version`` — the workspace opens on it, labelled.
SEED_NAME = "example-lunar-ice"
SEED_VERSION = "0.1.0"
SEED_REFERENCE = f"{SEED_NAME}:{SEED_VERSION}"


def _example_draft() -> IntentDraft:
    """The anchor scenario as a structured objective — the same one the form pre-fills."""
    return IntentDraft(
        id=SEED_NAME,
        name="Lunar polar water-ice (example)",
        author="astro-mine studio serve",
        region=GeoRegion(
            name="Shackleton rim",
            crs=PlanetaryCRS(body="MOON", body_fixed_frame="MOON_ME", reference_radius_m=1737400.0),
        ),
        products=[
            TargetProduct(
                criterion_id="water",
                metric="water_production_rate",
                unit="kg/day",
                target=40.0,
                tolerance=60.0,
                rate_window_s=86400.0,
                weight=1.0,
            )
        ],
        constraints=[
            HardConstraint(criterion_id="power", metric="power_margin", unit="W", threshold=0.0)
        ],
        inventory=[AssetSelection(sadf_ref="prospecting-rover:0.1.0", count=4)],
    )


async def example_study() -> tuple[ObjectiveDocument, TradeStudy]:
    """Run the example design loop and return its objective + trade study.

    Split out of :func:`_author_example` so the *pinned campaign* and the *committed UI fixture*
    (``scripts/regen_seed_fixture.py``) are built from one code path. They were built from two, and
    the fixture fell three days behind ``TradeStudy.evaluator`` becoming required — so every first
    open showed "comparison failed: evaluator: Field required" (#49).
    """
    objective = build_objective(_example_draft())
    clients = local_clients()
    candidates = [
        DesignCandidate(
            id="lean-2",
            swarm=[AssetSelection(sadf_ref="prospecting-rover:0.1.0", count=2)],
        ),
        DesignCandidate(
            id="balanced-4",
            swarm=[AssetSelection(sadf_ref="prospecting-rover:0.1.0", count=4)],
        ),
        DesignCandidate(
            id="heavy-6",
            swarm=[AssetSelection(sadf_ref="prospecting-rover:0.1.0", count=6)],
        ),
    ]
    cache = InMemoryResultCache()
    records = await run_batch(
        candidates,
        objective,
        dispatcher=LocalDispatcher(clients),
        seeds=(7,),
        store=InMemoryJobStore(),
        cache=cache,
    )
    evaluated = [
        cache.get(record.cache_key)
        for record in records
        if record.status is JobStatus.SUCCEEDED and cache.has(record.cache_key)
    ]
    study = build_trade_study(
        evaluated,
        objective,
        backend="batch",
        evaluator=clients.evaluator,
        seeds=[7],
        study_id=SEED_NAME,
    )
    return objective, study


async def _author_example() -> Campaign:
    objective, study = await example_study()
    chosen_id = study.pareto_front[0] if study.pareto_front else study.evaluated[0].candidate.id
    chosen = next(item for item in study.evaluated if item.candidate.id == chosen_id)
    return author_campaign(
        objective,
        chosen,
        name=SEED_NAME,
        phases=[CampaignPhase(id="prospect", name="Prospect")],
        trade_study=study,
        campaign_id=SEED_NAME,
    )


def _is_current(campaign: Campaign) -> bool:
    """Does a pinned campaign carry what today's model records, or is it a stale seed?

    Idempotence used to be **by reference**: if `SEED_REFERENCE` resolved, the pin was reused
    verbatim and never re-authored. `SEED_VERSION` has not moved since seeding was introduced, so a
    registry seeded before ``evaluator``/``world_ref`` existed served the pre-`evaluator` campaign
    forever — measured as ``{"evaluator": null, "world_ref": null}`` on a registry in that state
    (#49). Bumping the version would fix today's instance and leave the same trap for the next
    schema change, so the check is on **content**: fields the current authoring path always fills
    must be filled, or the seed is re-authored over the same reference.

    ``world_ref`` is deliberately *not* checked — a campaign authored without a resolved world has a
    legitimate ``None`` there (`Campaign.world_ref`), so demanding it would re-author on every
    start.
    """
    return bool(campaign.evaluator) and campaign.trade_study_ref is not None


def ensure_example_seeded(publisher: ArtifactPublisher) -> str:
    """Publish the example campaign unless a **current** one is already pinned; return its
    reference. Idempotent by content, not by reference — see :func:`_is_current`."""
    try:
        pinned = publisher.pull_campaign(SEED_REFERENCE)
    except Exception:
        pass  # not seeded, or unreadable — author it
    else:
        if _is_current(pinned):
            return SEED_REFERENCE
    campaign = asyncio.run(_author_example())
    bundle = freeze_campaign(campaign)
    published = publisher.publish_campaign(bundle, name=SEED_NAME, version=SEED_VERSION)
    return published.reference
