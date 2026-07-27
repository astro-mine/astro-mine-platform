"""Learned illumination-field surrogate adapter (RM-P1-WORLDS-10; surrogate.md §3, §6, §9).

The adapter loads a **published** field surrogate — a self-contained ONNX graph + its calibrated
``ErrorReport`` + the Core manifest attributes — and serves illumination from it, reading admission
**entirely from the Core manifest** (validated against the *vendored* JSON Schemas) and NEVER
importing ``astro_mine.surrogate``. These tests use a tiny fixture surrogate (``tests/fixtures/
illumination/``, built by ``scripts/gen_illumination_surrogate_fixture.py``) that lights a cell iff
its easting is positive — deliberately distinct from the horizon/ray-cast reference so "the
surrogate served this" is distinguishable from "the reference served this".

They assert: the ``ErrorReport`` is present and schema-valid; admission is read from the manifest
(domain / trust region / budget / digest) and fails closed on a bad schema, wrong domain, or a
digest mismatch; the trust region gates in-domain serving vs OOD escalation; and the
``illumination_field`` domain maps to a registrable ``FIELD_MODEL`` manifest — with no Core change.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
import rasterio.transform

from astro_mine.core.hashing import content_hash_json
from astro_mine.core.registry import PluginKind, PluginManifest, PluginRegistry
from astro_mine.worlds.illumination import (
    IlluminationError,
    IlluminationModel,
    PsrEpochSemantics,
    PsrResult,
    SurrogateAdmissionError,
    SurrogateIlluminationModel,
    build_illumination_model,
    field_model_kind_for_domain,
)
from astro_mine.worlds.illumination._registry import surrogate_field_model
from astro_mine.worlds.illumination._surrogate import _vendored_schema
from astro_mine.worlds.terrain import ingest_dem

_FIXTURES = Path(__file__).parent / "fixtures" / "illumination"


def _artifacts() -> tuple[bytes, dict, dict]:
    onnx = (_FIXTURES / "model.onnx").read_bytes()
    attrs = json.loads((_FIXTURES / "surrogate_attributes.json").read_text())
    report = json.loads((_FIXTURES / "error_report.json").read_text())
    return onnx, attrs, report


@pytest.fixture
def artifacts() -> tuple[bytes, dict, dict]:
    return _artifacts()


@pytest.fixture
def product(synthetic_dem, tmp_path):
    return ingest_dem(synthetic_dem, tmp_path / "terrain", resolution_m=2000.0)


def _build(product, onnx, attrs, report, **kwargs) -> SurrogateIlluminationModel:
    return SurrogateIlluminationModel(
        product,
        onnx_model=onnx,
        manifest_attributes=attrs,
        error_report=report,
        name="fixture",
        n_azimuth=16,
        max_radius_m=8000.0,
        abcorr="NONE",
        **kwargs,
    )


def _easting_grid(model: SurrogateIlluminationModel) -> np.ndarray:
    rows, cols = np.mgrid[0 : model.height, 0 : model.width]
    xs, _ys = rasterio.transform.xy(model.transform, rows.ravel().tolist(), cols.ravel().tolist())
    return np.asarray(xs, dtype=np.float64).reshape(model.height, model.width)


# --- ErrorReport presence + admission --------------------------------------------


def test_vendored_schemas_declare_ids_in_worlds_own_namespace() -> None:
    """RFC-0009 §1: a package publishes ``$id``\\s only under its own namespace.

    These two schemas are byte-copies of Surrogate's published contract, and they used to
    carry **Surrogate's** ``$id``\\s — two packages publishing one name, which is a silent
    wrong-schema resolution the moment anything resolves by ``$id``. Worlds cannot ``$ref``
    Surrogate's originals instead: it must not depend on ``astro-mine-surrogate`` (the seam
    exists to carry the contract across the Core waist *without* that dependency), so the
    copies stay under names Worlds owns.
    """
    for name in ("error_report.schema.json", "surrogate_attributes.schema.json"):
        schema_id = _vendored_schema(name)["$id"]
        assert schema_id.startswith("https://schemas.astro-mine.org/worlds/"), name
        assert "/surrogate/" not in schema_id, name


def test_error_report_is_present_and_schema_valid(artifacts) -> None:
    _onnx, attrs, report = artifacts
    # The surrogate ships its calibrated ErrorReport; it validates against the vendored schema.
    jsonschema.validate(report, _vendored_schema("error_report.schema.json"))
    jsonschema.validate(attrs, _vendored_schema("surrogate_attributes.schema.json"))
    assert report["domain"] == "illumination_field"
    assert report["substitution_policy"]["recommended_error_budget"]  # a published bound


def test_admission_from_manifest_and_backend(product, artifacts) -> None:
    onnx, attrs, report = artifacts
    model = _build(product, onnx, attrs, report)
    assert model.error_report["domain"] == "illumination_field"
    assert model.recommended_error_budget == attrs["recommended_error_budget"]
    assert model.backend == "surrogate:fixture"
    assert model.to_manifest()["params"]["backend"] == "surrogate:fixture"
    # The horizon LOS map is still built (the always-present Link product), API unchanged.
    assert model.horizon.shape == (model.height, model.width, 16)


def test_domain_maps_to_registrable_field_model(artifacts) -> None:
    _onnx, attrs, _report = artifacts
    assert field_model_kind_for_domain(attrs["domain"]) == PluginKind.FIELD_MODEL
    manifest = PluginManifest(
        name="illumination-surrogate",
        version="0.1.0",
        kind=field_model_kind_for_domain(attrs["domain"]),
        core_interfaces={"world_provider": "0.1.0"},
        license="Apache-2.0",
        attributes=attrs,
    )
    assert manifest.kind == PluginKind.FIELD_MODEL
    PluginRegistry(require_signature=False).register(manifest)  # a plugin, no Core change


def test_non_field_domain_is_not_a_field_model() -> None:
    with pytest.raises(SurrogateAdmissionError, match="not a Worlds field-model domain"):
        field_model_kind_for_domain("granular_excavation")


# --- in-domain serving vs OOD escalation -----------------------------------------


def test_serves_in_domain_via_onnx(product, artifacts, synthetic_spice) -> None:
    onnx, attrs, report = artifacts
    model = _build(product, onnx, attrs, report)
    easting = _easting_grid(model)
    # All cells lie inside the (wide) trust region, so the surrogate serves every cell: its learned
    # rule (lit iff easting > 0) is reproduced exactly — proving ONNX Runtime, not the reference.
    mask = model.illuminated_mask(synthetic_spice.epoch)
    np.testing.assert_array_equal(mask, easting > 0.0)
    # Point queries agree with the same rule.
    pos = np.unravel_index(int(np.argmax(easting)), easting.shape)
    neg = np.unravel_index(int(np.argmin(easting)), easting.shape)
    px, py = rasterio.transform.xy(model.transform, int(pos[0]), int(pos[1]))
    nx, ny = rasterio.transform.xy(model.transform, int(neg[0]), int(neg[1]))
    assert model.sun_visible(float(px), float(py), synthetic_spice.epoch) is True
    assert model.illumination_at(float(nx), float(ny), synthetic_spice.epoch)[0] is False


def test_out_of_domain_escalates_to_reference(product, artifacts, synthetic_spice) -> None:
    onnx, attrs, report = artifacts
    ood = copy.deepcopy(attrs)
    ood["trust_region"]["bounds"]["easting_m"] = {
        "low": 1.0e6,
        "high": 2.0e6,
    }  # no grid cell inside
    model = _build(product, onnx, ood, report)
    reference = IlluminationModel(product, n_azimuth=16, max_radius_m=8000.0, abcorr="NONE")
    # Every cell is OOD, so the surrogate escalates to the horizon reference for the whole raster.
    np.testing.assert_array_equal(
        model.illuminated_mask(synthetic_spice.epoch),
        reference.illuminated_mask(synthetic_spice.epoch),
    )
    x, y = rasterio.transform.xy(model.transform, model.height // 2, model.width // 2)
    assert (
        model.illumination_at(float(x), float(y), synthetic_spice.epoch)[0]
        == (reference.illumination_at(float(x), float(y), synthetic_spice.epoch)[0])
    )


def test_escalation_disabled_always_serves_surrogate(product, artifacts, synthetic_spice) -> None:
    onnx, attrs, report = artifacts
    # A report that does not ask to escalate on OOD: the surrogate serves every cell even out of the
    # (here narrowed) trust region. The digest is recomputed so admission still passes fail-closed.
    no_escalate = copy.deepcopy(report)
    no_escalate["substitution_policy"]["escalate_on_ood"] = False
    attrs2 = copy.deepcopy(attrs)
    attrs2["trust_region"]["bounds"]["easting_m"] = {"low": 1.0e6, "high": 2.0e6}
    attrs2["error_report_digest"] = content_hash_json(no_escalate)
    model = _build(product, onnx, attrs2, no_escalate)
    easting = _easting_grid(model)
    np.testing.assert_array_equal(model.illuminated_mask(synthetic_spice.epoch), easting > 0.0)


# --- fail-closed admission --------------------------------------------------------


def test_admission_rejects_wrong_domain(product, artifacts) -> None:
    onnx, attrs, report = artifacts
    bad = copy.deepcopy(attrs)
    bad["domain"] = "thermal_field"  # schema-valid but not an illumination field model
    with pytest.raises(SurrogateAdmissionError, match="not"):
        _build(product, onnx, bad, report)


def test_admission_rejects_digest_mismatch(product, artifacts) -> None:
    onnx, attrs, report = artifacts
    bad = copy.deepcopy(attrs)
    bad["error_report_digest"] = "sha256:" + "0" * 64
    with pytest.raises(SurrogateAdmissionError, match="does not match"):
        _build(product, onnx, bad, report)


def test_admission_rejects_error_report_domain_mismatch(product, artifacts) -> None:
    onnx, attrs, report = artifacts
    bad_report = copy.deepcopy(report)
    bad_report["domain"] = "thermal_field"  # manifest says illumination_field; the report disagrees
    with pytest.raises(SurrogateAdmissionError, match="ErrorReport domain"):
        _build(product, onnx, attrs, bad_report)


def test_admission_rejects_schema_invalid_attributes(product, artifacts) -> None:
    onnx, attrs, report = artifacts
    bad = copy.deepcopy(attrs)
    del bad["trust_region"]  # a required SurrogateAttributes field
    with pytest.raises(SurrogateAdmissionError, match="schema"):
        _build(product, onnx, bad, report)


def test_admission_rejects_unsupported_input_channel(product, artifacts) -> None:
    onnx, attrs, report = artifacts
    bad = copy.deepcopy(attrs)
    bad["input_channels"] = [*attrs["input_channels"], "unmodelled_channel"]
    with pytest.raises(SurrogateAdmissionError, match="cannot build"):
        _build(product, onnx, bad, report)


# --- public API / PSR semantics preserved ----------------------------------------


def test_surrogate_preserves_public_api(product, artifacts, synthetic_spice) -> None:
    onnx, attrs, report = artifacts
    model = _build(product, onnx, attrs, report)
    with pytest.raises(IlluminationError, match="outside the terrain grid"):
        model.sun_visible(1.0e9, 1.0e9, synthetic_spice.epoch)
    result = model.psr_mask(
        synthetic_spice.window, 6.0 * 3600.0, semantics=PsrEpochSemantics.MISSION
    )
    assert isinstance(result, PsrResult)
    from astro_mine.spice import epoch_range

    ever = np.zeros((model.height, model.width), dtype=np.bool_)
    for epoch in epoch_range(synthetic_spice.window, 6.0 * 3600.0):
        ever |= model.illuminated_mask(epoch)
    np.testing.assert_array_equal(result.mask, ~ever)


# --- factory / entry-point wiring -------------------------------------------------


def _surrogate_kwargs(artifacts) -> dict:
    onnx, attrs, report = artifacts
    return {
        "onnx_model": onnx,
        "manifest_attributes": attrs,
        "error_report": report,
        "name": "fixture",
    }


def test_surrogate_via_factory_and_entry_point(product, artifacts) -> None:
    surrogate = _surrogate_kwargs(artifacts)
    model = build_illumination_model(
        product,
        backend="surrogate:fixture",
        surrogate=surrogate,
        n_azimuth=16,
        max_radius_m=8000.0,
        abcorr="NONE",
    )
    assert isinstance(model, SurrogateIlluminationModel)
    assert model.backend == "surrogate:fixture"
    via_entry_point = surrogate_field_model(
        product, surrogate=surrogate, n_azimuth=16, max_radius_m=8000.0, abcorr="NONE"
    )
    assert isinstance(via_entry_point, SurrogateIlluminationModel)


def test_surrogate_backend_requires_artifacts(product) -> None:
    with pytest.raises(IlluminationError, match="published surrogate artifacts"):
        build_illumination_model(product, backend="surrogate:x", abcorr="NONE")
