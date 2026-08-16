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

**Discovery had a third blind spot, and closing it changed the shape of the problem.** The set said
nine because discovery looked in two places: zoo pins and shipped Fleet asset ids.
``shackleton_water_ice_pds_v1`` is published and non-conforming and was in neither — it is a
*prospect prior*, and a prior published under the registry key of its recipe, because
``publish_prior`` defaulted ``name`` to ``prior.provenance.recipe``. Nothing here could see it, so
the "exhaustive and exact" claim above was false. Found while building the registry inventory in
astro-mine-platform#41.

The fix was not to record a tenth legacy name. It was to stop a recipe key from *being* an artifact
name: a key is a Python-side identifier that names a callable and is snake_case for that reason,
while an artifact name is what published bytes are addressed by. They are now separate
(:func:`astro_mine.prospect.priors.default_artifact_name`), the publish default derives a conforming
name from the key, and :func:`_prospect_artifact_names` reads the *published* names. A prior added
tomorrow cannot mint a non-conforming name even by accident.

What remains is the already-published artifact still sitting in the registry under the old name.
That is a migration item rather than a code reference, so it is tracked in
``registry-inventory.json``, where the published set is recorded — not here, where tree references
are.
"""

from __future__ import annotations

import json
import pathlib
import tempfile

import pytest
import yaml

from astro_mine.core.registry import PluginKind, PluginManifest
from astro_mine.hub.client import HubClient
from astro_mine.hub.registry import (
    ARTIFACT_NAME_PATTERN,
    InvalidArtifactName,
    Registry,
    is_valid_artifact_name,
    validate_artifact_name,
)
from astro_mine.prospect.priors import list_artifact_names

REPO = pathlib.Path(__file__).resolve().parents[2]

#: Every published artifact name that predates §13, and the name it becomes at the flip.
#:
#: Nine — the count astro-mine-platform#23 opened with, and now for a checked reason rather than an
#: unchecked one: prior artifact names are discovered and conform, so nothing is hiding behind the
#: gap that hid ``shackleton_water_ice_pds_v1``. ``excavation-gns`` is pinned and deliberately
#: absent: it already conformed, and is the proof that the rule describes something achievable.
LEGACY_NAMES: dict[str, str] = {}


def _pinned_names() -> set[str]:
    """Every artifact name the scenario zoo pins, across all scenarios."""
    names: set[str] = set()
    for pins in (REPO / "src" / "astro_mine" / "bench" / "zoo").rglob("pins.json"):
        names |= set(json.loads(pins.read_text(encoding="utf-8")))
    return names


def _prospect_artifact_names() -> set[str]:
    """Every registered prior's **published** name — the third discovery source.

    Deliberately :func:`~astro_mine.prospect.priors.list_artifact_names` and not ``list_recipes``.
    The recipe key is a Python-side identifier and is allowed to stay snake_case; what has to
    satisfy §13 is the name the bytes are published under. Reading the live registry rather than a
    hand-kept list is what makes a *new* prior fail :func:`test_no_new_non_conforming_names` on the
    day it is added, the way a new Fleet asset or zoo pin already does.
    """
    return set(list_artifact_names())


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


def test_the_migration_is_complete() -> None:
    """The inventory is empty, and empty is the whole point of it having existed.

    While the migration was outstanding this set was the enforcement: exhaustive, exact, and
    checked in both directions so it could neither grow a quiet exemption nor shrink without
    someone noticing. It is empty now, which is the state it was built to reach rather than a
    weakening of it — the rule moved to :meth:`HubClient.publish`, where it is a property of
    publishing rather than a property of remembering.
    """
    assert LEGACY_NAMES == {}, (
        f"the legacy inventory is not empty: {sorted(LEGACY_NAMES)}. If a non-conforming name has "
        f"been published again, that is a defect in the publish gate, not a new exemption."
    )


def test_no_new_non_conforming_names() -> None:
    """**The gate.** Every pinned name and shipped asset id conforms, or is a recorded legacy.

    This is what makes the rule enforceable before the migration can run. Adding a new artifact with
    a dotted, snaked or version-suffixed name fails here, naming the file and the fix.
    """
    observed = _pinned_names() | _shipped_asset_ids() | _prospect_artifact_names()
    assert observed, "found no names to check — the discovery above has broken, not the content"

    offenders = {n for n in observed if not is_valid_artifact_name(n)}
    assert offenders == set(), (
        f"non-conforming artifact names: {sorted(offenders)}. conventions.md §13 requires bare "
        f"kebab-case with the version in the tag. There is no longer a legacy set to join — the "
        f"migration is complete and `HubClient.publish` refuses these outright."
    )


def test_the_rule_is_enforced_at_publish() -> None:
    """**The gate, at the endpoint §13 names.** Not a list someone has to remember to update.

    This is the assertion the whole module was building toward, and it is stronger than the
    inventory it replaces in a specific way: the inventory could only see names that had already
    been *published*, so it was structurally blind to a non-conforming name minted for the first
    time. The move surfaced three such cases that no list could have caught — the Fleet template
    factory building `astro-mine.fleet.<family>.<variant>`, `publish_prior` defaulting to a
    snake_case recipe key, and `recipe_reference_name` appending its suffix to that same key.
    """
    registry = Registry(tempfile.mkdtemp())
    client = HubClient(registry)
    manifest = PluginManifest(
        name="whatever", version="0.1.0", kind=PluginKind.RESOURCE_FIELD_BACKEND
    )
    with pytest.raises(InvalidArtifactName, match="excavator"):
        client.publish(
            name="astro-mine.fleet.excavator",
            version="0.1.0",
            kind="asset",
            manifest=manifest,
            private_key_pem=b"",
        )


def test_the_already_conforming_anchor_was_never_legacy() -> None:
    """`excavation-gns` shipped conformant before the rule existed, and proves it is achievable."""
    assert "excavation-gns" in _pinned_names()
    assert is_valid_artifact_name("excavation-gns")
    assert "excavation-gns" not in LEGACY_NAMES
