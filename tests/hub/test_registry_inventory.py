"""The committed registry inventory (astro-mine-platform#41), and the properties that keep it true.

``registry-inventory.json`` records every artifact this tree has published: its digest, whether
anything pins it, and whether a second copy exists. It exists because immutability is a property of
the *record*, not of the store — a pruned registry answers "nothing here", which is
indistinguishable from "never published", so a record kept only inside the registry evaporates
exactly when it matters.

**These tests never open a registry.** The record has to be verifiable on a clean clone with no
workspace store and no network; a test that read `files/hub-registry` would pass only on the one
machine whose prune caused the problem. So the assertions are internal consistency plus agreement
with what the tree pins.

The single-copy set below follows the same exact-set discipline as ``test_artifact_names.py``'s
legacy inventory: pinned exactly, so it cannot quietly grow, and entries leave it in the same change
that mirrors them.
"""

from __future__ import annotations

import json
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
INVENTORY = REPO / "registry-inventory.json"

_SHA256 = re.compile(r"^sha256:[a-f0-9]{64}$")
_DISPOSITIONS = {"published", "lost", "ephemeral"}

#: Artifacts that are real, pinned by something in the tree, and exist in exactly one place.
#:
#: This is the open half of astro-mine-platform#41. Both are absent from `ghcr.io/astro-mine`, which
#: holds exactly the nine anchor packages, so the workspace store is their only copy — the same
#: condition that made the 2026-08-08 prune unrecoverable. Pinned exactly rather than as a floor: a
#: *new* single-copy pinned artifact is a regression and fails here, and when these two are mirrored
#: they come out of this set in the same change that adds their `mirrored_to` entry.
SINGLE_COPY = {
    "excavation-gns:0.6.0",
    "shackleton_water_ice_pds_v1:1.0.0",
}


@pytest.fixture(scope="module")
def inventory() -> dict:
    return json.loads(INVENTORY.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def artifacts(inventory: dict) -> dict:
    return inventory["artifacts"]


# --- shape ----------------------------------------------------------------------------------


def test_every_entry_is_well_formed(artifacts: dict) -> None:
    for ref, record in artifacts.items():
        assert ref.count(":") == 1, f"{ref} is not a name:version reference"
        assert record["disposition"] in _DISPOSITIONS, f"{ref}: {record['disposition']}"
        for field in ("manifest_digest", "bundle_digest"):
            value = record.get(field)
            assert value is None or _SHA256.fullmatch(value), f"{ref}.{field} = {value!r}"
        assert isinstance(record.get("mirrored_to"), list), ref
        assert isinstance(record.get("pinned_by"), list), ref


def test_mirror_references_resolve(inventory: dict, artifacts: dict) -> None:
    """A `mirrored_to` naming a mirror that does not exist claims a copy that does not exist."""
    known = set(inventory["mirrors"])
    for ref, record in artifacts.items():
        unknown = set(record["mirrored_to"]) - known
        assert unknown == set(), f"{ref} names unknown mirrors {sorted(unknown)}"


def test_published_artifacts_carry_a_digest(artifacts: dict) -> None:
    """An entry claiming to be live that cannot say what it is records nothing at all."""
    for ref, record in artifacts.items():
        if record["disposition"] == "published":
            assert record["manifest_digest"] is not None, f"{ref} is published with no digest"


def test_lost_artifacts_are_not_mirrored(artifacts: dict) -> None:
    """`lost` means it resolves nowhere. A mirror would make it `published`."""
    for ref, record in artifacts.items():
        if record["disposition"] == "lost":
            assert record["mirrored_to"] == [], f"{ref} is recorded lost but claims a mirror"


def test_ephemeral_artifacts_are_pinned_by_nothing(artifacts: dict) -> None:
    """The disposition *is* the prune permission, so it may not contradict `pinned_by`."""
    for ref, record in artifacts.items():
        if record["disposition"] == "ephemeral":
            assert record["pinned_by"] == [], (
                f"{ref} is marked safe to prune but something pins it: {record['pinned_by']}. "
                f"One of the two is wrong, and pruning it would settle which."
            )


def test_every_pinned_by_path_exists_and_mentions_the_artifact(artifacts: dict) -> None:
    """A stale path is worse than none — it reads as evidence while pointing at nothing."""
    for ref, record in artifacts.items():
        name = ref.rsplit(":", 1)[0]
        for relative in record["pinned_by"]:
            path = REPO / relative
            assert path.exists(), f"{ref}: pinned_by names a missing file, {relative}"
            body = path.read_text(encoding="utf-8", errors="replace")
            assert name in body, f"{ref}: {relative} does not mention {name}"


# --- agreement with the tree ------------------------------------------------------------------


def test_scenario_content_hashes_match_the_inventory(artifacts: dict) -> None:
    """Every zoo `content_hash` resolves to an inventory entry with that manifest digest.

    This is the check that makes the record load-bearing rather than decorative: if a re-pin moved a
    scenario's digest and nobody updated the inventory, the guard would verify rebuilds against
    a stale value. Here the two are forced to move together.
    """
    by_digest = {
        record["manifest_digest"]: ref
        for ref, record in artifacts.items()
        if record["manifest_digest"] is not None
    }
    checked = 0
    for scenario in (REPO / "src" / "astro_mine" / "bench" / "zoo").rglob("scenario.json"):
        content = json.loads(scenario.read_text(encoding="utf-8")).get("content") or {}
        for slot, value in content.items():
            for pin in value if isinstance(value, list) else [value] if value else []:
                digest = pin.get("content_hash")
                if digest is None:
                    continue
                checked += 1
                assert digest in by_digest, (
                    f"{scenario.relative_to(REPO)} pins {pin.get('id')} at {digest} in slot "
                    f"'{slot}', which no registry-inventory.json entry records. Publishing content "
                    f"the zoo pins means adding its entry in the same change."
                )
                assert by_digest[digest].startswith(f"{pin['id']}:"), (
                    f"{digest} is recorded as {by_digest[digest]} but pinned as {pin['id']}"
                )
    assert checked, "found no scenario content pins — the discovery has broken, not the content"


def test_every_zoo_pinned_name_is_recorded(artifacts: dict) -> None:
    """`pins.json` names the artifacts a scenario was built against, digest or not."""
    recorded = {ref.rsplit(":", 1)[0] for ref in artifacts}
    for pins in (REPO / "src" / "astro_mine" / "bench" / "zoo").rglob("pins.json"):
        for name in json.loads(pins.read_text(encoding="utf-8")):
            assert name in recorded, (
                f"{pins.relative_to(REPO)} pins {name}, which registry-inventory.json does not "
                f"record. An artifact the zoo depends on and the tree has no record of is the "
                f"condition astro-mine-platform#41 exists to prevent."
            )


# --- the open gap -------------------------------------------------------------------------------


def test_the_single_copy_set_is_exact(artifacts: dict) -> None:
    """Real, pinned, one copy. The set may not drift in either direction.

    Growing means a new artifact was published into the same trap. Shrinking without removing the
    entry here means the mirror happened and this set was not updated with it.
    """
    observed = {
        ref
        for ref, record in artifacts.items()
        if record["disposition"] == "published"
        and record["pinned_by"]
        and not record["mirrored_to"]
    }
    assert observed == SINGLE_COPY, (
        f"the set of pinned, unmirrored artifacts changed: {sorted(observed)}. If one was "
        f"mirrored, remove it from SINGLE_COPY in the same change; if one was added, it is a new "
        f"single-point-of-failure and astro-mine-platform#41 is the reason not to."
    )


def test_the_anchor_nine_are_mirrored(artifacts: dict) -> None:
    """The claim the workspace convention actually makes, asserted rather than assumed.

    "The store is a convenience, no longer the only source" is true of exactly these nine. Recording
    which entries it is *false* of is the point of the inventory.
    """
    mirrored = {ref for ref, record in artifacts.items() if "ghcr" in record["mirrored_to"]}
    assert len(mirrored) == 9, f"expected the nine anchor packages, found {sorted(mirrored)}"
