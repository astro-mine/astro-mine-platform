# SPDX-License-Identifier: Apache-2.0
"""Learned illumination-field surrogate adapter — a Worlds field-model backend (RM-P1-WORLDS-10).

worlds.md §11's open question — *could a learned surrogate replace ray casting for very large
swarm-scale queries, with tracked error?* — co-designed with [Surrogate](surrogate.md). A field
surrogate publishes itself as a Core ``field_model`` behind the ``world_provider`` interface
(surrogate.md §3: a ``PhysicsDomain.ILLUMINATION_FIELD`` surrogate reuses the **existing**
``PluginKind.FIELD_MODEL`` — no Core change). This module is the Worlds side of that seam: it loads
such a published surrogate (the self-contained ONNX graph + its calibrated ``ErrorReport`` + the
signed Core manifest) and serves the illumination Sun-visibility surface from it, gating on the
declared **trust region** so an out-of-distribution query escalates to the reference model rather
than extrapolating (surrogate.md principle 3, "Out-of-distribution silence is forbidden").

**Worlds never imports ``astro_mine.surrogate``.** Admission is read entirely from the Core
:class:`~astro_mine.core.registry.PluginManifest` ``attributes`` (``domain`` /
``recommended_error_budget`` / ``trust_region`` / ``error_report_digest``), duck-typed as a plain
mapping and validated against the **vendored** JSON Schemas in ``illumination/schema/`` — the same
contract Sim's scheduler consumes, honoured across the Core waist without a package dependency
(conventions.md §1.1, §1.7; surrogate.md §6). The surrogate is measured against Worlds' own ray-cast
oracle, so the *error is the product* it carries: no calibrated ``ErrorReport``, no admission.

Those vendored schemas declare ``$id``\\s under **Worlds'** namespace, not Surrogate's (RFC-0009 §1:
a package publishes ``$id``\\s only under its own, and two packages must never publish the same
one). They previously carried Surrogate's ``$id``\\s verbatim — two packages publishing one name, a
silent wrong-schema resolution waiting for anything that resolves by ``$id``. ``$ref``\\ing
Surrogate's own schema instead would mean *depending on Surrogate*, which is precisely what the
paragraph above forbids: the whole point of this seam is that the contract crosses the Core waist
without the dependency. So the copies stay, under names Worlds owns. They are the admission gate's
read of Surrogate's published contract; Surrogate remains its author (surrogate.md §6).
"""

from __future__ import annotations

import functools
import importlib.resources
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import rasterio.transform
from numpy.typing import NDArray

from astro_mine.core.hashing import content_hash_json
from astro_mine.core.registry import PluginKind
from astro_mine.core.units import Epoch
from astro_mine.worlds.illumination import IlluminationError, IlluminationModel

__all__ = [
    "ILLUMINATION_FIELD_DOMAIN",
    "SurrogateAdmissionError",
    "SurrogateIlluminationModel",
    "field_model_kind_for_domain",
]

#: The ``PhysicsDomain`` value (surrogate.md) a learned illumination surrogate declares.
ILLUMINATION_FIELD_DOMAIN = "illumination_field"
#: Surrogate domains that map to a Worlds ``field_model`` (behind the ``world_provider`` interface).
_FIELD_DOMAINS = frozenset({ILLUMINATION_FIELD_DOMAIN, "thermal_field"})
#: The input-feature channels this adapter can build for a query. A surrogate whose declared
#: ``input_channels`` are a subset of these is servable; any other channel fails admission (a
#: surrogate the adapter cannot feed must not be silently mis-served).
_SUPPORTED_CHANNELS = ("easting_m", "northing_m", "epoch_s", "sun_elevation_deg")


class SurrogateAdmissionError(IlluminationError):
    """A learned surrogate failed the load-time admission gate (schema / domain / integrity).

    A subclass of :class:`IlluminationError` so existing callers catch it, but distinct so a
    consumer can tell a *rejected surrogate* from an ordinary out-of-bounds query. Fail-closed: the
    surrogate model is never constructed when this fires (surrogate.md §9).
    """


def field_model_kind_for_domain(domain: str) -> PluginKind:
    """The Core :class:`PluginKind` a surrogate ``domain`` publishes as, for a Worlds field model.

    Mirrors surrogate.md's domain→kind mapping for the **field** family (``illumination_field`` /
    ``thermal_field`` → :attr:`PluginKind.FIELD_MODEL`) without importing ``astro_mine.surrogate`` —
    the "reuse an existing closed-vocabulary member, no new Core kind" rule (surrogate.md §3). A
    non-field domain is not a Worlds field model and is rejected.
    """
    if domain in _FIELD_DOMAINS:
        return PluginKind.FIELD_MODEL
    raise SurrogateAdmissionError(
        f"domain {domain!r} is not a Worlds field-model domain {sorted(_FIELD_DOMAINS)}"
    )


@functools.cache
def _vendored_schema(name: str) -> dict[str, Any]:
    """Load a vendored surrogate JSON Schema from ``illumination/schema/`` (cached)."""
    resource = importlib.resources.files("astro_mine.worlds.illumination").joinpath("schema", name)
    schema: dict[str, Any] = json.loads(resource.read_text(encoding="utf-8"))
    return schema


def _validate(instance: Mapping[str, Any], schema_name: str, *, what: str) -> None:
    """Validate ``instance`` against a vendored schema; re-raise as a fail-closed admission."""
    import jsonschema

    try:
        jsonschema.validate(dict(instance), _vendored_schema(schema_name))
    except jsonschema.ValidationError as exc:
        raise SurrogateAdmissionError(f"{what} failed schema {schema_name}: {exc.message}") from exc


class SurrogateIlluminationModel(IlluminationModel):
    """Illumination served by a learned field surrogate, with the horizon map as the LOS product.

    Subclasses :class:`~astro_mine.worlds.illumination.IlluminationModel` so it still builds the
    per-azimuth horizon map (the Link line-of-sight product and the reference the adapter escalates
    to on an out-of-trust-region query) and inherits the public API; :meth:`illumination_at` and
    :meth:`illuminated_mask` are overridden to query the ONNX graph inside the trust region and fall
    back to the horizon reference outside it. Admission (schema-valid manifest attributes, the
    ``illumination_field`` domain, and the ``ErrorReport`` whose content hash matches the manifest's
    ``error_report_digest``) is checked in the constructor and fails closed.
    """

    def __init__(
        self,
        terrain: Any,
        *,
        onnx_model: bytes,
        manifest_attributes: Mapping[str, Any],
        error_report: Mapping[str, Any],
        name: str,
        backend: str | None = None,
        **kwargs: Any,
    ) -> None:
        attrs = self._admit(manifest_attributes, error_report)
        super().__init__(terrain, backend=backend or f"surrogate:{name}", **kwargs)
        self.surrogate_name = name
        self._error_report: dict[str, Any] = dict(error_report)
        self._trust_region: dict[str, dict[str, float]] = attrs["trust_region"]["bounds"]
        self._input_channels: tuple[str, ...] = tuple(attrs["input_channels"])
        self.recommended_error_budget: dict[str, float] = dict(attrs["recommended_error_budget"])
        self._escalate_on_ood = bool(
            error_report.get("substitution_policy", {}).get("escalate_on_ood", True)
        )
        unsupported = set(self._input_channels) - set(_SUPPORTED_CHANNELS)
        if unsupported:
            raise SurrogateAdmissionError(
                f"surrogate {name!r} needs input channels this adapter cannot build: "
                f"{sorted(unsupported)} (supported: {list(_SUPPORTED_CHANNELS)})"
            )
        self._session = self._build_session(onnx_model)

    @staticmethod
    def _admit(
        manifest_attributes: Mapping[str, Any], error_report: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Run the fail-closed admission gate and return the validated manifest attributes."""
        _validate(
            manifest_attributes, "surrogate_attributes.schema.json", what="manifest attributes"
        )
        _validate(error_report, "error_report.schema.json", what="error report")
        if manifest_attributes.get("domain") != ILLUMINATION_FIELD_DOMAIN:
            raise SurrogateAdmissionError(
                f"surrogate domain {manifest_attributes.get('domain')!r} is not "
                f"{ILLUMINATION_FIELD_DOMAIN!r}; not an illumination field model"
            )
        if error_report.get("domain") != ILLUMINATION_FIELD_DOMAIN:
            raise SurrogateAdmissionError("ErrorReport domain does not match the manifest domain")
        declared = manifest_attributes.get("error_report_digest")
        actual = content_hash_json(dict(error_report))
        if actual != declared:
            raise SurrogateAdmissionError(
                f"ErrorReport content hash {actual} does not match the manifest's "
                f"error_report_digest {declared!r} — the served bound is not the signed one"
            )
        # `field_model_kind_for_domain` also gates the domain → FIELD_MODEL projection.
        field_model_kind_for_domain(str(manifest_attributes["domain"]))
        return manifest_attributes

    @staticmethod
    def _build_session(onnx_model: bytes) -> Any:
        """Build the ONNX Runtime CPU session (onnxruntime is the ``[surrogate]`` extra)."""
        import onnxruntime as ort

        return ort.InferenceSession(onnx_model, providers=["CPUExecutionProvider"])

    @property
    def error_report(self) -> dict[str, Any]:
        """The calibrated :class:`ErrorReport` (as a plain mapping) the surrogate carries."""
        return dict(self._error_report)

    # --- inference -----------------------------------------------------------------

    def _predict_lit(self, features: NDArray[np.float32]) -> NDArray[np.bool_]:
        """Run the ONNX graph on ``(N, C)`` features and threshold the lit score at zero."""
        input_name = self._session.get_inputs()[0].name
        output_name = self._session.get_outputs()[0].name
        (scores,) = self._session.run([output_name], {input_name: features})
        return np.asarray(scores, dtype=np.float64).reshape(features.shape[0]) > 0.0

    def _in_domain(self, values: Mapping[str, float]) -> bool:
        """Whether every input channel of a single query lies inside the trust region."""
        for channel in self._input_channels:
            bound = self._trust_region[channel]
            if not (bound["low"] <= values[channel] <= bound["high"]):
                return False
        return True

    def _in_domain_grid(self, grids: Mapping[str, NDArray[np.float64]]) -> NDArray[np.bool_]:
        """Per-cell trust-region membership across every input channel."""
        inside = np.ones(next(iter(grids.values())).shape, dtype=np.bool_)
        for channel in self._input_channels:
            bound = self._trust_region[channel]
            inside &= (grids[channel] >= bound["low"]) & (grids[channel] <= bound["high"])
        return inside

    def _point_features(
        self, x: float, y: float, epoch: Epoch, sun_elevation_deg: float
    ) -> dict[str, float]:
        return {
            "easting_m": x,
            "northing_m": y,
            "epoch_s": epoch.tdb_seconds,
            "sun_elevation_deg": sun_elevation_deg,
        }

    def _cell_feature_grids(
        self, epoch: Epoch, sun_elevation_deg: float
    ) -> dict[str, NDArray[np.float64]]:
        rows, cols = np.mgrid[0 : self.height, 0 : self.width]
        xs, ys = rasterio.transform.xy(self.transform, rows.ravel().tolist(), cols.ravel().tolist())
        easting = np.asarray(xs, dtype=np.float64).reshape(self.height, self.width)
        northing = np.asarray(ys, dtype=np.float64).reshape(self.height, self.width)
        return {
            "easting_m": easting,
            "northing_m": northing,
            "epoch_s": np.full((self.height, self.width), epoch.tdb_seconds, dtype=np.float64),
            "sun_elevation_deg": np.full(
                (self.height, self.width), sun_elevation_deg, dtype=np.float64
            ),
        }

    # --- Sun-visibility overrides --------------------------------------------------

    def illumination_at(self, x: float, y: float, epoch: Epoch) -> tuple[bool, float]:
        """``(sun_visible, sun_elevation_deg)`` served by the surrogate, or escalated when OOD."""
        row, col = rasterio.transform.rowcol(self.transform, x, y)
        row, col = int(row), int(col)
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise IlluminationError(f"({x}, {y}) is outside the terrain grid")
        elevation_deg, _azimuth = self._sun(x, y, epoch)
        values = self._point_features(x, y, epoch, elevation_deg)
        if self._escalate_on_ood and not self._in_domain(values):
            return super().illumination_at(x, y, epoch)[0], elevation_deg
        features = np.array(
            [[values[channel] for channel in self._input_channels]], dtype=np.float32
        )
        return bool(self._predict_lit(features)[0]), elevation_deg

    def illuminated_mask(self, epoch: Epoch) -> NDArray[np.bool_]:
        """Swarm-scale lit raster from the surrogate; OOD cells take the horizon reference."""
        cx, cy = rasterio.transform.xy(self.transform, self.height // 2, self.width // 2)
        elevation_deg, _azimuth = self._sun(float(cx), float(cy), epoch)
        grids = self._cell_feature_grids(epoch, elevation_deg)
        features = np.stack(
            [grids[channel].ravel() for channel in self._input_channels], axis=1
        ).astype(np.float32)
        predicted = self._predict_lit(features).reshape(self.height, self.width)
        if not self._escalate_on_ood:
            return predicted
        reference = super().illuminated_mask(epoch)
        return np.where(self._in_domain_grid(grids), predicted, reference)
