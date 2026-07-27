"""STUDIO-02 — design ↔ decision-variable encoding (lossless)."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import ValidationError

from astro_mine.studio.designspace import decode, encode
from astro_mine.studio.models import AssetChoice, DecisionSpace, DesignCandidate

_SPACE = DecisionSpace(
    assets=[AssetChoice(sadf_ref="rover", max_count=6), AssetChoice(sadf_ref="relay", max_count=3)],
    infrastructure=["sha256:orbiter"],
    policies={"planner": "sha256:policy"},
)


def test_asset_choice_rejects_inverted_bounds() -> None:
    with pytest.raises(ValidationError):
        AssetChoice(sadf_ref="x", max_count=2, min_count=5)


def test_decision_space_dimension_and_bounds() -> None:
    assert _SPACE.dimension() == 2
    assert _SPACE.bounds() == [(0, 6), (0, 3)]


def test_decode_materializes_canonical_candidate() -> None:
    candidate = decode((4, 0), _SPACE)
    assert isinstance(candidate, DesignCandidate)
    # zero-count kinds are dropped from the swarm but kept in the decision vector
    assert [s.sadf_ref for s in candidate.swarm] == ["rover"]
    assert candidate.decision_vector == {"rover": 4.0, "relay": 0.0}
    assert candidate.infrastructure == ["sha256:orbiter"]
    assert candidate.policy_refs == {"planner": "sha256:policy"}
    assert candidate.id.startswith("cand-")


def test_decode_rejects_wrong_length() -> None:
    with pytest.raises(ValueError, match="vector length"):
        decode((1, 2, 3), _SPACE)


def test_decode_is_deterministic() -> None:
    assert decode((2, 1), _SPACE).id == decode((2, 1), _SPACE).id
    assert decode((2, 1), _SPACE).id != decode((1, 2), _SPACE).id


@given(
    vector=st.tuples(st.integers(min_value=0, max_value=6), st.integers(min_value=0, max_value=3))
)
def test_encode_decode_roundtrip_is_lossless(vector: tuple[int, int]) -> None:
    assert encode(decode(vector, _SPACE), _SPACE) == vector
