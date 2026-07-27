"""RM-P1-STUDIO-08 — pin the content-addressed PlanetaryCRS to Core's units schema.

Studio content-addresses the CRS that rides inside an ``IntentDraft`` (``GeoRegion.crs``),
carried into the ``Campaign`` handed to Ops unchanged. These tests prove the CRS dict that
enters ``content_hash_json`` validates against Core's canonical ``units.schema.json``, that a
Moon CRS with an Earth (WGS84) marker is rejected at intent (rule 6), and that recording the
schema digest never moves an existing content hash (RFC-0007 §1a/§3; conventions.md §5).
"""

from __future__ import annotations

import pytest
from jsonschema import Draft202012Validator

from astro_mine.core import schema_registry
from astro_mine.core.schemas import core_schema
from astro_mine.core.units import UnitsValidationError
from astro_mine.core.units.model import PlanetaryCRS
from astro_mine.studio.campaign import author_campaign, freeze_campaign, handoff
from astro_mine.studio.crs_schema import (
    _CRS_PROBE_SCHEMA,
    CrsSchemaError,
    core_units_schema_digest,
    validate_crs_schema,
)
from astro_mine.studio.hashing import content_hash_json
from astro_mine.studio.intent import capture_intent
from astro_mine.studio.intent.forms import build_objective
from astro_mine.studio.models import GeoRegion, IntentDraft, TargetProduct
from astro_mine.studio.workspace import InMemoryWorkspace

_MOON = PlanetaryCRS(body="MOON", body_fixed_frame="MOON_ME", reference_radius_m=1737400.0)


def _draft(crs: PlanetaryCRS) -> IntentDraft:
    return IntentDraft(
        id="o",
        name="n",
        author="a",
        region=GeoRegion(name="r", crs=crs),
        products=[
            TargetProduct(criterion_id="c", metric="m", unit="kg", target=1.0, tolerance=0.1)
        ],
    )


# --- criterion 1: content_hash_json inputs validate against the canonical schema ---------- #


def test_content_hash_json_crs_input_validates_against_canonical_schema(
    lunar_draft: IntentDraft,
) -> None:
    """The exact CRS dict fed into ``content_hash_json`` (nested in the draft dump) is a valid
    ``PlanetaryCRS`` per Core's ``units.schema.json`` — validated with an independent validator
    built straight from Core's public ``schema_registry`` (RFC-0009 §2), not Studio's helper."""
    dumped = lunar_draft.model_dump(mode="json")  # the object content_hash_json hashes
    crs_dict = dumped["region"]["crs"]

    validator = Draft202012Validator(_CRS_PROBE_SCHEMA, registry=schema_registry(_CRS_PROBE_SCHEMA))
    assert list(validator.iter_errors(crs_dict)) == []  # canonical schema accepts it

    # and the Studio guard agrees, returning the same dict it content-addresses
    assert validate_crs_schema(lunar_draft.region.crs) == crs_dict


def test_validate_crs_schema_rejects_a_malformed_crs() -> None:
    with pytest.raises(CrsSchemaError, match=r"units\.schema\.json"):
        validate_crs_schema(
            {"body": "MOON", "body_fixed_frame": "MOON_ME", "reference_radius_m": -1.0}
        )


def test_probe_schema_owns_its_id_and_refs_core_by_absolute_id() -> None:
    """RFC-0009 §1: ``$id`` namespaces are owned, and Core is ``$ref``d by its absolute ``$id``.

    The probe used to declare its ``$id`` *inside Core's namespace* so that a bare relative ref
    would land on a path-shaped URI Core kept privately — a URI astro-mine-core#54 retired,
    which is what latently broke this module. Both halves are asserted here so neither can
    return: Studio names only its own namespace, and it reaches Core by the one public name.
    """
    assert _CRS_PROBE_SCHEMA["$id"].startswith("https://schemas.astro-mine.org/studio/")
    units_id = core_schema("astro_mine.core.units", "units.schema.json")["$id"]
    ref = _CRS_PROBE_SCHEMA["$ref"]
    assert ref == f"{units_id}#/$defs/PlanetaryCRS"
    # the retired path-shaped URI (kept resolvable only by the deprecated migration alias)
    assert "core/units/schema/units.schema.json" not in ref


def test_capture_intent_validates_crs_before_content_addressing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The content-addressing path calls the schema guard on the CRS it is about to hash."""
    seen: list[object] = []
    real = validate_crs_schema

    def _spy(crs: object) -> object:
        seen.append(crs)
        return real(crs)

    monkeypatch.setattr("astro_mine.studio.intent.validate_crs_schema", _spy)
    draft = _draft(_MOON)
    capture_intent(draft, workspace=InMemoryWorkspace())
    assert seen == [draft.region.crs]  # the guard ran on the artifact's CRS


# --- criterion 2: recording the schema digest does not move existing content hashes ------- #


def test_schema_digest_is_recorded_without_moving_the_campaign_hash(
    objective_doc: object, clients: object
) -> None:
    from astro_mine.studio.models import DesignCandidate
    from astro_mine.studio.orchestrate import evaluate_candidate

    chosen = evaluate_candidate(
        DesignCandidate(id="c", swarm=[]), objective_doc, clients=clients, seed=3
    )
    campaign = author_campaign(objective_doc, chosen, name="c", phases=[_phase()], trade_study=None)
    bundle = freeze_campaign(campaign)

    # the sidecar digest is present and content-addressed...
    assert bundle.schema_digest == core_units_schema_digest()
    assert bundle.schema_digest.startswith("sha256:")
    # ...yet the campaign's content hash is exactly its own model dump, digest-free
    assert bundle.digest == content_hash_json(campaign.model_dump(mode="json"))
    assert bundle.digest == campaign.digest()
    assert "schema_digest" not in campaign.model_dump(mode="json")
    assert "schema_digest" not in bundle.payload().decode()


def test_handoff_metadata_carries_schema_digest_for_rehydration(
    objective_doc: object, clients: object
) -> None:
    from astro_mine.studio.models import DesignCandidate
    from astro_mine.studio.orchestrate import evaluate_candidate

    chosen = evaluate_candidate(
        DesignCandidate(id="c", swarm=[]), objective_doc, clients=clients, seed=3
    )
    campaign = author_campaign(objective_doc, chosen, name="c", phases=[_phase()])
    ws = InMemoryWorkspace()
    bundle = handoff(campaign, workspace=ws, author="designer")
    assert ws.audit()[0].metadata["units_schema_digest"] == bundle.schema_digest


def _phase() -> object:
    from astro_mine.studio.models import CampaignPhase

    return CampaignPhase(id="p", name="P")


# --- criterion 3: a Moon CRS with a WGS84 datum is rejected at intent --------------------- #


def test_moon_crs_with_wgs84_datum_is_rejected_at_intent() -> None:
    """Rule 6 (RM-P1-CORE-08): an Earth datum marker on a non-Earth body is a defaulting bug.
    The GeoRegion still *constructs* (the guard is not a model constraint), but the intent path
    (``build_objective`` -> ``require_crs``, and ``capture_intent``) rejects it fail-loud."""
    moon_wgs84 = PlanetaryCRS(
        body="MOON", body_fixed_frame="MOON_ME", reference_radius_m=1737400.0, datum="WGS84"
    )
    draft = _draft(moon_wgs84)

    with pytest.raises(UnitsValidationError, match="Earth"):
        build_objective(draft)

    ws = InMemoryWorkspace()
    with pytest.raises(UnitsValidationError, match="Earth"):
        capture_intent(draft, workspace=ws)
    assert ws.audit() == ()  # never persisted


def test_moon_crs_with_epsg_projection_is_rejected_at_intent() -> None:
    moon_epsg = PlanetaryCRS(
        body="MOON",
        body_fixed_frame="MOON_ME",
        reference_radius_m=1737400.0,
        projection="EPSG:4326",
    )
    with pytest.raises(UnitsValidationError, match="Earth"):
        build_objective(_draft(moon_epsg))


def test_earth_crs_with_wgs84_datum_is_accepted_at_intent() -> None:
    """The mirror of rule 6: an Earth datum on ``body=EARTH`` is legitimate (Earth analogs)."""
    earth = PlanetaryCRS(
        body="EARTH", body_fixed_frame="ITRF93", reference_radius_m=6378137.0, datum="WGS84"
    )
    doc = build_objective(_draft(earth))
    assert doc.objective.labels["region.body"] == "EARTH"
    validate_crs_schema(earth)  # and it is schema-valid


def test_lunar_fixtures_do_not_trip_the_earth_marker_rule(lunar_draft: IntentDraft) -> None:
    """The anchor lunar fixtures carry no Earth marker, so rule 6 never fires on them."""
    captured = capture_intent(lunar_draft, workspace=InMemoryWorkspace())
    assert captured.document.objective.labels["region.body"] == "MOON"
