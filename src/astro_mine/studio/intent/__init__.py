"""Intent capture (RM-P1-STUDIO-01): NL/forms → a Core-validated ``ObjectiveSpec``.

Phase-1 ships the deterministic, always-available forms path (:mod:`.forms`) behind the
validation boundary (:mod:`.validate`). The optional provider-abstracted LLM adapter
(``intent/llm``, RM-P1-STUDIO-05) layers on later and reuses this exact boundary; it is
never required. :func:`capture_intent` is the front-door orchestration: build → validate
→ content-address → persist with provenance + authorship. Anything that fails the gate
raises **before** persistence, so a malformed spec is never stored as valid.
"""

from __future__ import annotations

from collections.abc import Sequence

from astro_mine.core.objective import ObjectiveDocument, to_wire

from .._base import FrozenStudioModel
from ..crs_schema import validate_crs_schema
from ..hashing import content_hash, content_hash_json
from ..models import IntentDraft
from ..provenance import ArtifactProvenance, capture_provenance
from ..workspace import WorkspaceStore
from .forms import build_objective
from .validate import MetricVocabulary, ObjectiveGateError, validate_objective_document

__all__ = [
    "CapturedObjective",
    "MetricVocabulary",
    "ObjectiveGateError",
    "build_objective",
    "capture_intent",
    "persist_objective",
    "validate_objective_document",
]


class CapturedObjective(FrozenStudioModel):
    """The persisted result of intent capture: the validated document, its content
    address, and the provenance that reproduces it."""

    document: ObjectiveDocument
    digest: str
    provenance: ArtifactProvenance


def persist_objective(
    document: ObjectiveDocument,
    *,
    workspace: WorkspaceStore,
    author: str,
    model: str | None = None,
    input_hashes: Sequence[str] = (),
) -> CapturedObjective:
    """Content-address a **validated** objective, persist it with provenance + authorship,
    and record the drafting model (``None`` for the forms path) in the audit log.

    The caller must have already passed ``document`` through
    :func:`validate_objective_document` — this is the shared persistence step behind both
    the forms path (:func:`capture_intent`) and the LLM path (``intent.llm.accept_draft``),
    so a spec only ever enters the workspace after clearing the boundary."""
    wire = to_wire(document)
    digest = content_hash(wire)
    provenance = capture_provenance(input_hashes=list(input_hashes))
    workspace.put(
        "objective",
        digest,
        wire,
        author=author,
        model=model,
        metadata={"objective_id": document.objective.id},
    )
    return CapturedObjective(document=document, digest=digest, provenance=provenance)


def capture_intent(
    draft: IntentDraft,
    *,
    workspace: WorkspaceStore,
    vocabulary: MetricVocabulary | None = None,
    model: str | None = None,
) -> CapturedObjective:
    """Capture an ``IntentDraft`` into a validated, content-addressed, persisted
    ``ObjectiveSpec``. Raises (without persisting) if the draft yields an invalid
    objective."""
    document = build_objective(draft)
    validate_objective_document(document, vocabulary=vocabulary)
    # Pin the CRS that rides inside the content-addressed IntentDraft to Core's canonical
    # units schema before it enters the reproducibility chain — not only at the forms
    # boundary (``build_objective`` -> ``require_crs``) (RM-P1-STUDIO-08, RFC-0007 §1a/§3).
    validate_crs_schema(draft.region.crs)
    return persist_objective(
        document,
        workspace=workspace,
        author=draft.author,
        model=model,
        input_hashes=[content_hash_json(draft.model_dump(mode="json"))],
    )
