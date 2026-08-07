"""The artifact-name rule (conventions.md §13), and the legacy set it does not yet apply to.

Three naming conventions grew up side by side, one per producing component, and all three were
visible in a single catalog listing (G3.6). §13 settled it: bare kebab-case, no component prefix,
version in the tag. This module is where that rule has teeth.

**Why a pinned inventory rather than a gate at publish.** The obvious enforcement point is
:meth:`HubClient.publish`, and it does not work yet. ``publish_asset`` passes ``name=identity.id``,
so a SADF asset's authored id *is* its registry name — and all six shipped Fleet library assets
carry legacy ids. A gate there would make ``astro-mine fleet publish`` refuse the platform's own
documented examples, before the migration that would fix them; and that migration is gated on the
public flip, because renaming published content means re-publishing under new digests while keeping
every existing scorecard resolvable (§13, "Artifact-name migration").

So the inventory below is the enforcement. It is exhaustive and exact — not a floor — which is what
makes it work in both directions:

* a **new** non-conforming name fails :func:`test_no_new_non_conforming_names`, so new artifacts are
  born conformant, which is the property §13 actually cares about;
* the **existing nine** are recorded rather than broken, so the published set keeps resolving;
* when the flip-time sweep runs, entries leave this set as they are migrated, and the day it empties
  the rule can move to ``HubClient.publish`` and this module becomes a gate.

A list that could only grow would rot into a permanent exemption. Pinning it exactly means the
migration cannot quietly stall, and a tenth legacy name cannot be added without saying so here.
"""

from __future__ import annotations

import json
import pathlib

import pytest
import yaml

from astro_mine.hub.registry import (
    ARTIFACT_NAME_PATTERN,
    InvalidArtifactName,
    is_valid_artifact_name,
    validate_artifact_name,
)

REPO = pathlib.Path(__file__).resolve().parents[2]

#: Every published artifact name that predates §13, and the name it becomes at the flip.
#:
#: Nine, which is exactly the count the issue that opened this named (astro-mine-platform#23). The
#: tenth pinned artifact, ``excavation-gns``, already conformed and is deliberately absent — it is
#: the proof that the rule describes something achievable rather than an aspiration.
LEGACY_NAMES = {
    "astro-mine.fleet.excavator": "excavator",
    "astro-mine.fleet.hauler": "hauler",
    "astro-mine.fleet.isru-plant": "isru-plant",
    "astro-mine.fleet.lander": "lander",
    "astro-mine.fleet.prospecting-rover": "prospecting-rover",
    "astro-mine.fleet.relay-orbiter": "relay-orbiter",
    "astro-mine.link.lunar-polar-relay-dsn": "lunar-polar-relay-dsn",
    "shackleton-de-gerlache-v1": "shackleton-de-gerlache",
    "shackleton_water_ice_v1": "shackleton-water-ice",
}


def _pinned_names() -> set[str]:
    """Every artifact name the scenario zoo pins, across all scenarios."""
    names: set[str] = set()
    for pins in (REPO / "src" / "astro_mine" / "bench" / "zoo").rglob("pins.json"):
        names |= set(json.loads(pins.read_text(encoding="utf-8")))
    return names


def _shipped_asset_ids() -> set[str]:
    """Every SADF asset id in the shipped Fleet library — each one a registry name at publish."""
    ids: set[str] = set()
    for path in (REPO / "src" / "astro_mine" / "fleet" / "library").rglob("*.sadf.yaml"):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        identity = document.get("asset", {}).get("identity", {})
        if "id" in identity:
            ids.add(str(identity["id"]))
    return ids


# --- the rule itself ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name", ["excavator", "prospecting-rover", "shackleton-de-gerlache", "excavation-gns", "gns2"]
)
def test_conforming_names_are_accepted(name: str) -> None:
    assert is_valid_artifact_name(name)
    assert validate_artifact_name(name) == name


@pytest.mark.parametrize(
    ("name", "why"),
    [
        ("astro-mine.fleet.excavator", "dots and a component prefix"),
        ("shackleton_water_ice", "underscores"),
        ("acme/no-entry", "a slash — a namespace is the `namespace=` argument"),
        ("Excavator", "uppercase"),
        ("2-rover", "does not start with a letter"),
        ("-rover", "leading hyphen"),
        ("rover-", "trailing hyphen"),
        ("rover--x", "an empty segment"),
        ("", "empty"),
        ("shackleton-de-gerlache-v1", "a version suffix beside a SemVer tag"),
        ("rover-v12", "a version suffix beside a SemVer tag"),
    ],
)
def test_non_conforming_names_are_rejected(name: str, why: str) -> None:
    assert not is_valid_artifact_name(name), why
    with pytest.raises(InvalidArtifactName):
        validate_artifact_name(name)


def test_the_version_suffix_rule_is_not_expressible_in_the_pattern() -> None:
    """Both halves of §13 are enforced, and the second is why the pattern alone is not enough.

    `shackleton-de-gerlache-v1` is *valid kebab-case* — it satisfies the regex and violates "the
    version lives in the tag". Checking only the pattern would leave one of the three legacy shapes
    able to be minted fresh, which is the failure this asserts against.
    """
    assert ARTIFACT_NAME_PATTERN.fullmatch("shackleton-de-gerlache-v1")
    assert not is_valid_artifact_name("shackleton-de-gerlache-v1")


def test_a_version_shaped_word_is_not_a_version_suffix() -> None:
    """The suffix rule must not swallow names that merely end in something v-ish."""
    for name in ("revision-vale", "carbo-v", "rover-mk2", "v1-rover"):
        assert is_valid_artifact_name(name), name


def test_the_error_says_what_to_write_instead() -> None:
    """A reader who has just been refused wants the correction, not the regex."""
    with pytest.raises(InvalidArtifactName, match="prospecting-rover"):
        validate_artifact_name("astro-mine.fleet.prospecting-rover")
    with pytest.raises(InvalidArtifactName, match="shackleton-water-ice"):
        validate_artifact_name("shackleton_water_ice_v1")
    # The version-suffix refusal explains itself rather than sending the reader hunting for a dot.
    with pytest.raises(InvalidArtifactName, match="version"):
        validate_artifact_name("shackleton-de-gerlache-v1")


# --- the legacy inventory -----------------------------------------------------------------------


def test_every_legacy_name_is_genuinely_non_conforming() -> None:
    """No entry may be parked here that the rule would have accepted anyway."""
    for name in LEGACY_NAMES:
        assert not is_valid_artifact_name(name), f"{name} conforms; it does not belong in the set"


def test_every_legacy_name_migrates_to_a_conforming_one() -> None:
    """The right-hand side is the flip-time worklist, so it has to be usable."""
    for old, new in LEGACY_NAMES.items():
        assert is_valid_artifact_name(new), f"{old} -> {new} is not conformant"
    assert len(set(LEGACY_NAMES.values())) == len(LEGACY_NAMES), "two names collide after migration"


def test_no_new_non_conforming_names() -> None:
    """**The gate.** Every pinned name and shipped asset id conforms, or is a recorded legacy.

    This is what makes the rule enforceable before the migration can run. Adding a new artifact with
    a dotted, snaked or version-suffixed name fails here, naming the file and the fix.
    """
    observed = _pinned_names() | _shipped_asset_ids()
    assert observed, "found no names to check — the discovery above has broken, not the content"

    offenders = {n for n in observed if not is_valid_artifact_name(n) and n not in LEGACY_NAMES}
    assert offenders == set(), (
        f"non-conforming artifact names outside the recorded legacy set: {sorted(offenders)}. "
        f"conventions.md §13 requires bare kebab-case with the version in the tag. If this is new "
        f"content, rename it; it must not join the migration backlog."
    )


def test_the_legacy_set_is_exact_and_still_present() -> None:
    """The set may not drift in either direction while the migration is outstanding.

    Shrinking it silently would mean content vanished; growing it would mean the rule quietly
    acquired an exemption. When the flip-time sweep runs, entries come out of `LEGACY_NAMES` in the
    same change that renames them, and this test is what forces the two to move together.
    """
    observed = _pinned_names() | _shipped_asset_ids()
    stale = set(LEGACY_NAMES) - observed
    assert stale == set(), (
        f"recorded as legacy but no longer used anywhere: {sorted(stale)}. If these were migrated, "
        f"remove them from LEGACY_NAMES in the same change."
    )


def test_the_already_conforming_anchor_is_not_recorded_as_legacy() -> None:
    """`excavation-gns` shipped conformant before the rule existed, and proves it is achievable."""
    assert "excavation-gns" in _pinned_names()
    assert is_valid_artifact_name("excavation-gns")
    assert "excavation-gns" not in LEGACY_NAMES
