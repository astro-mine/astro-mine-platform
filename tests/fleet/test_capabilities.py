"""Applied capability taxonomy — Core-vocabulary validation + export-control gate (FLEET-10)."""

from __future__ import annotations

import pytest

from astro_mine.core.sadf.enums import CapabilityTag
from astro_mine.fleet.capabilities import (
    GATED_CAPABILITY_TAGS,
    CapabilityError,
    as_tags,
    assert_open_commons,
    gated_tags,
)


def test_as_tags_accepts_core_vocabulary_from_strings_and_enums() -> None:
    tags = as_tags(["mobility.wheeled", CapabilityTag.EXCAVATION_BUCKET, "isru.electrolysis"])
    assert tags == [
        CapabilityTag.MOBILITY_WHEELED,
        CapabilityTag.EXCAVATION_BUCKET,
        CapabilityTag.ISRU_ELECTROLYSIS,
    ]


def test_as_tags_rejects_a_tag_outside_cores_vocabulary() -> None:
    # A tag Core does not define is a Core RFC, never a Fleet-private extension (fleet.md §2.5).
    with pytest.raises(CapabilityError, match="Core RFC"):
        as_tags(["mobility.jetpack"])


def test_gated_tags_flags_only_reserved_tags() -> None:
    found = gated_tags(["mobility.wheeled", "operational_targeting", "power.storage"])
    assert found == [CapabilityTag.OPERATIONAL_TARGETING]


def test_assert_open_commons_passes_a_clean_tag_set() -> None:
    assert assert_open_commons(["mobility.wheeled", "power.storage", "sensing.imu"]) is None


@pytest.mark.parametrize("tag", sorted(GATED_CAPABILITY_TAGS, key=str))
def test_assert_open_commons_rejects_every_gated_tag(tag: CapabilityTag) -> None:
    with pytest.raises(CapabilityError, match="reserved/gated"):
        assert_open_commons(["mobility.wheeled", tag])
