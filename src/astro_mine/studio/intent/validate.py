"""Intent validation boundary — fail early and loudly (studio.md §9).

Every produced/captured spec is validated against Core schemas at the boundary before
it is allowed to flow downstream; a spec that does not validate is **rejected and
surfaced for correction — it never flows downstream and is never persisted**. This is
the guarantee the (later, optional) LLM adapter is tested against (RM-P1-STUDIO-05):
the same gate rejects malformed model output.

The metric vocabulary is negotiated against a **declared capability** — a Bench
``metric``-kind Core plugin manifest — not by importing Bench (studio.md §2 principle 4;
the narrow-waist "no sibling-package imports" constraint).
"""

from __future__ import annotations

from astro_mine.core.objective import (
    ObjectiveDocument,
    from_wire,
    to_wire,
    validate_objective,
)
from astro_mine.core.registry.model import PluginManifest

from .._base import StudioModel


class ObjectiveGateError(ValueError):
    """A produced ``ObjectiveSpec`` failed the validation boundary. Surfaced for human
    correction; the caller must not persist or forward the spec."""


class MetricVocabulary(StudioModel):
    """The declared metric vocabulary an objective's bindings are checked against.

    ``metrics`` maps a Bench metric key to its expected SI unit; an empty string means
    the unit is unconstrained by the vocabulary. Built from a Bench ``metric`` plugin
    manifest (``from_manifest``) in production, or directly in tests/local use."""

    metrics: dict[str, str]

    @classmethod
    def from_manifest(cls, manifest: PluginManifest) -> MetricVocabulary:
        """A Bench ``metric``-kind plugin declares its metric keys as manifest outputs
        (unit unconstrained here — Core manifests carry keys, not per-key units)."""
        if manifest.kind.value != "metric":
            raise ObjectiveGateError(
                f"expected a 'metric' plugin manifest, got '{manifest.kind.value}'"
            )
        return cls(metrics={key: "" for key in manifest.outputs})

    def check(self, metric: str, unit: str) -> None:
        if metric not in self.metrics:
            raise ObjectiveGateError(f"metric '{metric}' is not in the declared vocabulary")
        expected = self.metrics[metric]
        if expected and expected != unit:
            raise ObjectiveGateError(f"metric '{metric}' expects unit '{expected}', got '{unit}'")


def validate_objective_document(
    doc: ObjectiveDocument, *, vocabulary: MetricVocabulary | None = None
) -> None:
    """Validate a produced objective at the boundary. Raises :class:`ObjectiveGateError`
    (or a Core ``ObjectiveError``) on any failure; returns ``None`` when the spec is
    safe to persist and forward.

    Three checks: (1) Core structural + semantic validation of the in-memory document
    (unique criterion ids, evaluation-window rules — ``validate_objective`` checks the
    document directly, without a text/YAML re-parse, so it is robust to values like
    subnormal floats that the text loader mishandles); (2) its Protobuf wire form is
    byte-stable (reproducibility — the digest is stable); (3) every metric key is in the
    declared vocabulary."""
    validate_objective(doc)
    if from_wire(to_wire(doc)) != doc:
        raise ObjectiveGateError("objective wire form is not byte-stable")
    if vocabulary is not None:
        for criterion in doc.objective.success_criteria:
            vocabulary.check(criterion.binding.metric, criterion.binding.unit)
