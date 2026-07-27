"""Seeding is idempotent by **content**, not by reference (#49).

`ensure_example_seeded` used to return early whenever `SEED_REFERENCE` resolved, and `SEED_VERSION`
has not moved since seeding was introduced. So a registry seeded before ``evaluator``/``world_ref``
existed on `Campaign` kept serving the pre-`evaluator` campaign forever — measured as
``{"evaluator": null, "world_ref": null}`` on a registry in that state.

Bumping `SEED_VERSION` would have fixed that instance and left the same trap for the next schema
change, so the check is on what the pinned artifact carries. These tests use a stub publisher rather
than a real registry: what is under test is the *decision* to re-author, not the publish path (which
`test_cli.py` exercises end to end).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from astro_mine.studio.campaign import CampaignBundle
from astro_mine.studio.hub.publish import PublishedArtifactRef
from astro_mine.studio.models import Campaign
from astro_mine.studio.seed import SEED_NAME, SEED_REFERENCE, SEED_VERSION, ensure_example_seeded


@dataclass
class _StubPublisher:
    """Holds one pinned campaign and counts how often it is re-authored."""

    pinned: Campaign | None = None
    published: list[Campaign] = field(default_factory=list)

    def pull_campaign(self, reference: str) -> Campaign:
        assert reference == SEED_REFERENCE
        if self.pinned is None:
            raise KeyError(reference)  # not seeded
        return self.pinned

    def publish_campaign(
        self, bundle: CampaignBundle, *, name: str, version: str
    ) -> PublishedArtifactRef:
        assert (name, version) == (SEED_NAME, SEED_VERSION)
        self.published.append(bundle.campaign)
        self.pinned = bundle.campaign
        return PublishedArtifactRef(
            reference=f"{name}:{version}",
            digest="sha256:" + "1" * 64,
            content_digest=bundle.campaign.digest(),
            kind="campaign",
        )


def test_an_empty_registry_is_seeded() -> None:
    publisher = _StubPublisher()
    assert ensure_example_seeded(publisher) == SEED_REFERENCE  # type: ignore[arg-type]
    assert len(publisher.published) == 1
    assert publisher.published[0].evaluator == "stand-in/0.1.0"


def test_a_current_pin_is_reused_and_nothing_is_re_authored() -> None:
    publisher = _StubPublisher()
    ensure_example_seeded(publisher)  # type: ignore[arg-type]
    ensure_example_seeded(publisher)  # type: ignore[arg-type]
    assert len(publisher.published) == 1, "a current pin must not be re-authored"


@pytest.mark.parametrize(
    ("update", "why"),
    [
        ({"evaluator": None}, "pinned before TradeStudy.evaluator existed"),
        ({"trade_study_ref": None}, "pinned with no study to justify it"),
    ],
)
def test_a_stale_pin_is_re_authored_over_the_same_reference(
    update: dict[str, None], why: str
) -> None:
    publisher = _StubPublisher()
    ensure_example_seeded(publisher)  # type: ignore[arg-type]
    assert publisher.pinned is not None
    # Rewind the pin to the shape a pre-2026-07-25 registry holds.
    publisher.pinned = publisher.pinned.model_copy(update=update)
    publisher.published.clear()

    assert ensure_example_seeded(publisher) == SEED_REFERENCE  # type: ignore[arg-type]
    assert len(publisher.published) == 1, f"a campaign {why} must be re-authored"
    assert publisher.published[0].evaluator == "stand-in/0.1.0"
    assert publisher.published[0].trade_study_ref is not None


def test_a_world_less_campaign_is_not_treated_as_stale() -> None:
    """`world_ref` is legitimately None when the design was never inspected on a world.

    Checking it would re-author on every start, which is the opposite failure.
    """
    publisher = _StubPublisher()
    ensure_example_seeded(publisher)  # type: ignore[arg-type]
    assert publisher.pinned is not None and publisher.pinned.world_ref is None
    publisher.published.clear()
    ensure_example_seeded(publisher)  # type: ignore[arg-type]
    assert publisher.published == []
