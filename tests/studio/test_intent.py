"""STUDIO-01 — intent capture: forms projection, validation boundary, persistence."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from astro_mine.core.objective import ObjectiveDocument, WindowKind, from_wire, to_wire
from astro_mine.core.registry import PluginKind
from astro_mine.core.registry.model import PluginManifest
from astro_mine.core.units.model import PlanetaryCRS
from astro_mine.studio.intent import (
    CapturedObjective,
    MetricVocabulary,
    ObjectiveGateError,
    capture_intent,
)
from astro_mine.studio.intent.forms import build_objective
from astro_mine.studio.intent.validate import validate_objective_document
from astro_mine.studio.models import GeoRegion, IntentDraft, TargetProduct
from astro_mine.studio.workspace import InMemoryWorkspace

_MOON = PlanetaryCRS(body="MOON", body_fixed_frame="MOON_ME", reference_radius_m=1737400.0)


# ---- forms ---------------------------------------------------------------- #


def test_build_objective_maps_products_and_constraints(lunar_draft: IntentDraft) -> None:
    doc = build_objective(lunar_draft)
    assert isinstance(doc, ObjectiveDocument)
    criteria = {c.id: c for c in doc.objective.success_criteria}
    # rate product -> rolling window; target + tolerance preserved
    water = criteria["water"].binding
    assert water.target == 40.0 and water.tolerance == 60.0
    assert water.evaluation_window is not None
    assert water.evaluation_window.kind is WindowKind.ROLLING
    # hard constraint -> required criterion with a threshold
    power = criteria["power"]
    assert power.required and power.binding.threshold == 0.0
    assert doc.objective.labels["region.body"] == "MOON"


def test_build_objective_without_rate_has_no_window() -> None:
    draft = IntentDraft(
        id="o",
        name="n",
        author="a",
        region=GeoRegion(name="r", crs=_MOON),
        products=[
            TargetProduct(criterion_id="c", metric="m", unit="kg", target=1.0, tolerance=0.1)
        ],
    )
    binding = build_objective(draft).objective.success_criteria[0].binding
    assert binding.evaluation_window is None


def test_build_objective_rejects_empty_criteria() -> None:
    draft = IntentDraft(id="o", name="n", author="a", region=GeoRegion(name="r", crs=_MOON))
    with pytest.raises(ValidationError):  # Core ObjectiveSpec requires >= 1 criterion
        build_objective(draft)


# ---- validation boundary -------------------------------------------------- #


def test_metric_vocabulary_check() -> None:
    vocab = MetricVocabulary(metrics={"m": "kg", "free": ""})
    vocab.check("m", "kg")  # ok
    vocab.check("free", "anything")  # unconstrained unit ok
    with pytest.raises(ObjectiveGateError, match="not in the declared vocabulary"):
        vocab.check("absent", "kg")
    with pytest.raises(ObjectiveGateError, match="expects unit"):
        vocab.check("m", "lb")


def test_metric_vocabulary_from_manifest() -> None:
    manifest = PluginManifest(
        name="bench-metrics",
        version="0.1.0",
        kind=PluginKind.METRIC,
        core_interfaces={},
        inputs=[],
        outputs=["water_production_rate", "power_margin"],
        capability_tags=[],
        regimes=[],
    )
    vocab = MetricVocabulary.from_manifest(manifest)
    assert set(vocab.metrics) == {"water_production_rate", "power_margin"}


def test_metric_vocabulary_from_manifest_rejects_non_metric() -> None:
    manifest = PluginManifest(
        name="a-rover",
        version="0.1.0",
        kind=PluginKind.ASSET,
        core_interfaces={},
        inputs=[],
        outputs=[],
        capability_tags=[],
        regimes=[],
    )
    with pytest.raises(ObjectiveGateError, match="expected a 'metric' plugin"):
        MetricVocabulary.from_manifest(manifest)


def test_validate_rejects_unknown_metric(objective_doc: ObjectiveDocument) -> None:
    with pytest.raises(ObjectiveGateError, match="not in the declared vocabulary"):
        validate_objective_document(
            objective_doc, vocabulary=MetricVocabulary(metrics={"only": ""})
        )


def test_validate_rejects_semantically_invalid_objective() -> None:
    from astro_mine.core.objective import (
        MetricBinding,
        MetricDirection,
        ObjectiveDocument,
        ObjectiveError,
        ObjectiveSpec,
        SuccessCriterion,
    )

    def _crit(cid: str) -> SuccessCriterion:
        return SuccessCriterion(
            id=cid,
            binding=MetricBinding(
                metric="m",
                unit="kg",
                direction=MetricDirection.HIGHER_BETTER,
                target=1.0,
                tolerance=0.1,
            ),
        )

    dup = ObjectiveDocument(
        objective_version="0.1",
        objective=ObjectiveSpec(id="o", name="n", success_criteria=[_crit("dup"), _crit("dup")]),
    )
    with pytest.raises(ObjectiveError):  # Core semantic check: criterion ids must be unique
        validate_objective_document(dup)


def test_validate_detects_non_bytestable_wire(
    objective_doc: ObjectiveDocument, monkeypatch: pytest.MonkeyPatch
) -> None:
    other = objective_doc.model_copy(deep=True)
    other.objective.name = "tampered"
    monkeypatch.setattr("astro_mine.studio.intent.validate.from_wire", lambda _data: other)
    with pytest.raises(ObjectiveGateError, match="wire form is not byte-stable"):
        validate_objective_document(objective_doc)


# ---- capture orchestration ------------------------------------------------ #


def test_capture_persists_with_provenance_and_authorship(
    lunar_draft: IntentDraft, vocabulary: MetricVocabulary
) -> None:
    ws = InMemoryWorkspace()
    captured = capture_intent(lunar_draft, workspace=ws, vocabulary=vocabulary)
    assert isinstance(captured, CapturedObjective)
    assert ws.has(captured.digest)
    assert ws.get(captured.digest) == to_wire(captured.document)
    audit = ws.audit()[0]
    assert audit.author == "designer@astro-mine" and audit.model is None
    assert captured.provenance.core_interface_versions["objective"] == "0.1.0"


def test_capture_is_deterministic(lunar_draft: IntentDraft, vocabulary: MetricVocabulary) -> None:
    a = capture_intent(lunar_draft, workspace=InMemoryWorkspace(), vocabulary=vocabulary)
    b = capture_intent(lunar_draft, workspace=InMemoryWorkspace(), vocabulary=vocabulary)
    assert a.digest == b.digest


def test_capture_records_llm_model(lunar_draft: IntentDraft, vocabulary: MetricVocabulary) -> None:
    ws = InMemoryWorkspace()
    capture_intent(lunar_draft, workspace=ws, vocabulary=vocabulary, model="claude-opus-4-8")
    assert ws.audit()[0].model == "claude-opus-4-8"


def test_capture_rejects_and_does_not_persist(
    lunar_draft: IntentDraft, vocabulary: MetricVocabulary
) -> None:
    bad = lunar_draft.model_copy(
        update={
            "products": [
                TargetProduct(
                    criterion_id="x", metric="unknown", unit="kg", target=1.0, tolerance=0.1
                )
            ],
            "constraints": [],
        }
    )
    ws = InMemoryWorkspace()
    with pytest.raises(ObjectiveGateError):
        capture_intent(bad, workspace=ws, vocabulary=vocabulary)
    assert ws.audit() == ()  # never persisted as valid


# ---- property: every produced spec serialises and re-validates ------------ #


@given(
    specs=st.lists(
        st.tuples(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=10),
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
            st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=5,
    )
)
def test_produced_objective_roundtrips(specs: list[tuple[str, float, float]]) -> None:
    draft = IntentDraft(
        id="obj",
        name="n",
        author="a",
        region=GeoRegion(name="r", crs=_MOON),
        products=[
            TargetProduct(
                criterion_id=f"c{i}", metric=metric, unit="kg", target=target, tolerance=tol
            )
            for i, (metric, target, tol) in enumerate(specs)
        ],
    )
    doc = build_objective(draft)
    validate_objective_document(doc)  # Core validation + byte-stable wire round-trip
    assert from_wire(to_wire(doc)) == doc
