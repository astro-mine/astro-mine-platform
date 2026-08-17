# SPDX-License-Identifier: Apache-2.0
"""``OnnxBundle`` — the content-addressed served-surrogate artifact (RM-P1-SURR-04).

The one payload a trained surrogate ships to Sim: the **self-contained ONNX graph** (network →
ensemble mean/std → conformal half-width → trust-region box → OOD inflation, so ONNX Runtime emits
``next_state`` *and* calibrated ``uncertainty``/``in_domain`` with no post-processing), a small
``serve_meta`` of the numpy-side featurization params a consumer needs to build the graph inputs,
and the surrogate's :class:`~astro_mine.surrogate.report.ErrorReport` so the artifact is
self-describing. It is **content-addressed**: :meth:`content_hash` over a *deterministic* archive
(sorted entries, zeroed timestamps, no compression) so the same bundle hashes identically across
machines — the ``artifact_digest`` a :class:`~astro_mine.core.registry.PluginManifest` pins and a
signature binds to (surrogate.md §4, §5; the immutable-artifact rule).
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from typing import Any

from astro_mine.core.hashing import content_hash
from astro_mine.surrogate.report import ErrorReport

__all__ = ["ONNX_BUNDLE_MEDIA_TYPE", "SERVE_META_FORMAT_VERSION", "OnnxBundle"]

#: Media type of the served bundle (the OCI layer a Hub artifact carries).
ONNX_BUNDLE_MEDIA_TYPE = "application/vnd.astro-mine.surrogate.onnx-bundle.v1+zip"
#: Bumped only on a breaking change to the ``serve_meta`` / archive layout. v2 is the raw-state
#: graph (featurization moved in-graph): ``serve_meta`` dropped the normalizer/cutoff params.
SERVE_META_FORMAT_VERSION = 2

_MODEL_ENTRY = "model.onnx"
_META_ENTRY = "serve_meta.json"
_REPORT_ENTRY = "error_report.json"
# A fixed DOS timestamp (1980-01-01) so the archive bytes carry no wall-clock — the bundle is
# content-addressed and must be byte-reproducible from the same inputs.
_FIXED_DATE_TIME = (1980, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class OnnxBundle:
    """A served surrogate: the ONNX graph, its featurization metadata, and its ErrorReport.

    ``onnx_model`` is the serialized self-contained graph; ``serve_meta`` carries the numpy-side
    featurization params (normalizer stats, radius-graph cutoff, bed width, channel layout) a
    consumer applies to build the graph's inputs; ``error_report`` is the calibrated bound the
    manifest commits to by digest. Immutable and content-addressed via :meth:`serialize`.
    """

    onnx_model: bytes
    serve_meta: dict[str, Any]
    error_report: ErrorReport

    def serialize(self) -> bytes:
        """The deterministic archive bytes — the bundle's content-addressable form.

        A ``ZIP_STORED`` archive with entries written in a **fixed order** and a **zeroed**
        timestamp, so two bundles built from identical inputs serialize byte-for-byte identically
        (the content-address requirement, surrogate.md §5).
        """
        buffer = io.BytesIO()
        # Canonical, sorted-key JSON for the metadata/report so their bytes are stable too.
        parts = {
            _MODEL_ENTRY: self.onnx_model,
            _META_ENTRY: json.dumps(
                self.serve_meta, sort_keys=True, separators=(",", ":")
            ).encode(),
            _REPORT_ENTRY: json.dumps(
                self.error_report.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode(),
        }
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
            for name in sorted(parts):
                info = zipfile.ZipInfo(name, date_time=_FIXED_DATE_TIME)
                archive.writestr(info, parts[name])
        return buffer.getvalue()

    def content_hash(self) -> str:
        """The ``sha256:<hex>`` content address of this bundle (its ``artifact_digest``)."""
        return content_hash(self.serialize())

    @classmethod
    def parse(cls, data: bytes) -> OnnxBundle:
        """Reconstruct an :class:`OnnxBundle` from its :meth:`serialize` bytes.

        Validates the ``serve_meta`` format version and re-hydrates the ``ErrorReport`` through its
        Pydantic model (so a tampered/malformed report fails loudly at the boundary, core.md
        principle 7).
        """
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            onnx_model = archive.read(_MODEL_ENTRY)
            meta = json.loads(archive.read(_META_ENTRY))
            report = ErrorReport.model_validate_json(archive.read(_REPORT_ENTRY))
        version = meta.get("format_version")
        if version != SERVE_META_FORMAT_VERSION:
            raise ValueError(
                f"unsupported serve_meta format_version {version!r}; "
                f"this build reads {SERVE_META_FORMAT_VERSION}"
            )
        return cls(onnx_model=onnx_model, serve_meta=meta, error_report=report)
