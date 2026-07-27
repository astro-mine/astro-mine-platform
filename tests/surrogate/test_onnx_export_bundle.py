"""ONNX export + content-addressed bundle (RM-P1-SURR-04).

The offline `serve.export` step: a trained surrogate becomes a self-contained ONNX graph packaged
as a deterministic, content-addressed :class:`OnnxBundle` carrying its featurization metadata and
calibrated ErrorReport — the artifact a manifest pins by digest (surrogate.md §4, §5).
"""

from __future__ import annotations

from astro_mine.surrogate.enums import ServedBackend
from astro_mine.surrogate.manifest import build_surrogate_manifest
from astro_mine.surrogate.serve import OnnxBundle


def test_export_produces_a_bundle_with_graph_meta_and_report(served_bundle) -> None:
    assert isinstance(served_bundle.onnx_model, bytes) and served_bundle.onnx_model
    # Featurization now lives in-graph; serve_meta carries only the query field vocabulary + layout.
    meta = served_bundle.serve_meta
    assert meta["domain"] == "granular_excavation"
    assert meta["ensemble_size"] >= 1
    assert meta["input_fields"] == ["position", "velocity", "tool_x", "config"]
    assert meta["output_field_layout"] == [["position", 2], ["velocity", 2]]
    # The calibrated report travels with the artifact (self-describing).
    assert served_bundle.error_report.domain.value == "granular_excavation"


def test_bundle_serialization_is_deterministic_and_content_addressed(served_bundle) -> None:
    first = served_bundle.serialize()
    second = served_bundle.serialize()
    assert first == second  # zeroed timestamps + fixed entry order -> byte-reproducible
    assert (
        served_bundle.content_hash() == f"sha256:{__import__('hashlib').sha256(first).hexdigest()}"
    )


def test_bundle_round_trips_through_parse(served_bundle) -> None:
    reparsed = OnnxBundle.parse(served_bundle.serialize())
    assert reparsed.content_hash() == served_bundle.content_hash()
    assert reparsed.serve_meta == served_bundle.serve_meta
    assert reparsed.error_report == served_bundle.error_report


def test_parse_rejects_an_unknown_format_version(served_bundle) -> None:
    import io
    import json
    import zipfile

    # Rebuild the archive with a bumped format_version -> parse must fail loudly.
    data = served_bundle.serialize()
    buffer = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(data)) as src, zipfile.ZipFile(buffer, "w") as dst:
        for name in src.namelist():
            payload = src.read(name)
            if name == "serve_meta.json":
                meta = json.loads(payload)
                meta["format_version"] = 999
                payload = json.dumps(meta).encode()
            dst.writestr(name, payload)
    try:
        OnnxBundle.parse(buffer.getvalue())
    except ValueError as exc:
        assert "format_version" in str(exc)
    else:  # pragma: no cover - the parse must raise
        raise AssertionError("expected a ValueError on an unknown format_version")


def test_manifest_pins_the_bundle_and_records_onnx_backend(served_bundle) -> None:
    manifest = build_surrogate_manifest(
        name="excavation-gns",
        version="0.1.0",
        report=served_bundle.error_report,
        artifact_digest=served_bundle.content_hash(),
        served_backend=ServedBackend.ONNX,
        native_graph_fallback=False,
    )
    assert manifest.provenance.digest == served_bundle.content_hash()
    assert manifest.attributes["served_backend"] == "onnx"
    assert manifest.attributes["native_graph_fallback"] is False
    # The admission budget rides in the manifest for Sim's scheduler to read directly.
    assert manifest.attributes["recommended_error_budget"] == dict(
        served_bundle.error_report.substitution_policy.recommended_error_budget
    )
