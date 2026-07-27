"""Foundation: hashing, provenance, models, workspace."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from astro_mine.studio.hashing import canonical_json, content_hash, content_hash_json
from astro_mine.studio.models import (
    AssetSelection,
    CandidateScore,
    DesignCandidate,
    EvaluatedCandidate,
    TargetProduct,
)
from astro_mine.studio.provenance import ArtifactProvenance, capture_provenance
from astro_mine.studio.workspace import (
    AuditEntry,
    InMemoryWorkspace,
    WorkspaceError,
    WorkspaceStore,
)

# ---- hashing -------------------------------------------------------------- #


def test_content_hash_is_prefixed_and_deterministic() -> None:
    assert content_hash(b"abc").startswith("sha256:")
    assert content_hash(b"abc") == content_hash(b"abc")
    assert content_hash(b"abc") != content_hash(b"abd")


def test_canonical_json_is_key_order_independent() -> None:
    assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})
    assert content_hash_json({"x": [1, 2]}) == content_hash_json({"x": [1, 2]})


# ---- provenance ----------------------------------------------------------- #


def test_capture_provenance_snapshots_core_versions() -> None:
    prov = capture_provenance(
        input_hashes=["sha256:aa"], seed=7, engine_versions={"sim": "1.2"}, env_lockfile="lock"
    )
    assert prov.input_hashes == ["sha256:aa"]
    assert prov.seed == 7
    assert prov.engine_versions == {"sim": "1.2"}
    assert prov.env_lockfile == "lock"
    assert prov.core_interface_versions["objective"] == "0.1.0"
    assert prov.code_version is not None
    assert prov.toolchain_version == "python3.12"


def test_provenance_is_deterministic_and_frozen() -> None:
    a = capture_provenance(input_hashes=["sha256:aa"], seed=1)
    b = capture_provenance(input_hashes=["sha256:aa"], seed=1)
    assert a == b  # no wall-clock -> content-addressable
    with pytest.raises(ValidationError):
        ArtifactProvenance(seed=1).seed = 2  # type: ignore[misc]


# ---- models --------------------------------------------------------------- #


def test_asset_selection_requires_positive_count() -> None:
    with pytest.raises(ValidationError):
        AssetSelection(sadf_ref="x", count=0)


def test_target_product_rejects_negative_tolerance() -> None:
    with pytest.raises(ValidationError):
        TargetProduct(criterion_id="c", metric="m", unit="u", target=1.0, tolerance=-1.0)


def test_design_candidate_digest_is_stable() -> None:
    a = DesignCandidate(id="c", swarm=[AssetSelection(sadf_ref="r", count=2)])
    b = DesignCandidate(id="c", swarm=[AssetSelection(sadf_ref="r", count=2)])
    assert a.digest() == b.digest()
    assert a.digest().startswith("sha256:")


def test_evaluated_candidate_digest() -> None:
    candidate = DesignCandidate(id="c", swarm=[AssetSelection(sadf_ref="r", count=1)])
    evaluated = EvaluatedCandidate(
        candidate=candidate,
        score=CandidateScore(
            objective_hash="sha256:o", metric_scores={"m": 1.0}, aggregate=1.0, passed=True
        ),
        seed=0,
        world_ref="sha256:w",
        provenance=capture_provenance(input_hashes=[], seed=0),
    )
    assert evaluated.digest().startswith("sha256:")


# ---- workspace ------------------------------------------------------------ #


def test_workspace_stores_and_retrieves_content_addressed() -> None:
    ws = InMemoryWorkspace()
    assert isinstance(ws, WorkspaceStore)
    payload = b"hello"
    digest = content_hash(payload)
    assert not ws.has(digest)
    ws.put("blob", digest, payload, author="me", metadata={"k": "v"})
    assert ws.has(digest)
    assert ws.get(digest) == payload
    entry = ws.audit()[0]
    assert isinstance(entry, AuditEntry)
    assert entry.author == "me" and entry.seq == 0 and entry.metadata == {"k": "v"}


def test_workspace_rejects_digest_mismatch() -> None:
    ws = InMemoryWorkspace()
    with pytest.raises(WorkspaceError, match="digest mismatch"):
        ws.put("blob", "sha256:wrong", b"hello", author="me")
    assert ws.audit() == ()  # fail-closed: nothing recorded


def test_workspace_missing_get_raises() -> None:
    with pytest.raises(WorkspaceError, match="no artifact"):
        InMemoryWorkspace().get("sha256:absent")


def test_workspace_records_llm_model_in_audit() -> None:
    ws = InMemoryWorkspace()
    payload = b"spec"
    ws.put("objective", content_hash(payload), payload, author="a", model="claude-opus-4-8")
    assert ws.audit()[0].model == "claude-opus-4-8"
