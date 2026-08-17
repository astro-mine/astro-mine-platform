# SPDX-License-Identifier: Apache-2.0
"""Design ↔ decision-variable encoding (studio.md §3 ``designspace/encode``, §11).

A heterogeneous, variable-cardinality swarm is encoded as one integer per asset kind —
its count — in ``DecisionSpace`` order. The codec is **lossless** against the decision
vector: ``encode(decode(v)) == v`` for any in-bounds ``v`` (the property STUDIO-02
asserts with Hypothesis). Decode is total and deterministic — the candidate id is the
content hash of ``(vector, space)``, so the same point always yields the same candidate
id (and therefore the same content-addressed evaluation).
"""

from __future__ import annotations

from collections.abc import Sequence

from ..hashing import content_hash_json
from ..models import AssetSelection, DecisionSpace, DesignCandidate


def encode(candidate: DesignCandidate, space: DecisionSpace) -> tuple[int, ...]:
    """Project a candidate's swarm onto the decision vector (count per asset kind)."""
    counts = {selection.sadf_ref: selection.count for selection in candidate.swarm}
    return tuple(counts.get(choice.sadf_ref, 0) for choice in space.assets)


def decode(vector: Sequence[int], space: DecisionSpace) -> DesignCandidate:
    """Materialize the canonical ``DesignCandidate`` at a decision vector.

    The swarm carries only positive-count kinds (``AssetSelection`` requires count ≥ 1),
    while the full vector — zeros included — is preserved on ``decision_vector`` so the
    encode round-trip is lossless. Infrastructure and the policy stack come from the
    (fixed) space."""
    if len(vector) != space.dimension():
        raise ValueError(f"vector length {len(vector)} != space dimension {space.dimension()}")

    swarm: list[AssetSelection] = []
    decision_vector: dict[str, float] = {}
    for choice, raw in zip(space.assets, vector, strict=True):
        count = int(raw)
        decision_vector[choice.sadf_ref] = float(count)
        if count > 0:
            swarm.append(AssetSelection(sadf_ref=choice.sadf_ref, count=count))

    identity = content_hash_json(
        {"vector": [int(v) for v in vector], "space": space.model_dump(mode="json")}
    )
    return DesignCandidate(
        id=f"cand-{identity[7:19]}",  # drop the 'sha256:' prefix, keep 12 hex chars
        swarm=swarm,
        infrastructure=list(space.infrastructure),
        policy_refs=dict(space.policies),
        decision_vector=decision_vector,
    )
