"""Golden / determinism gate for the compiler (RM-P1-GUARD-01 acceptance).

"The property enforced is exactly the property reviewed": the compiled IR of the anchor spec
is pinned byte-for-byte to a checked-in golden, and two compiles produce an identical content
hash. CI fails on non-reproducibility.

To refresh the golden after an intentional change: run ``scripts/gen_golden.py``.
"""

from __future__ import annotations

from pathlib import Path

from astro_mine.core.hashing import canonical_json
from astro_mine.guard.spec import CompiledSafetyModel, SafetyDocument, compile_spec

GOLDEN = Path(__file__).resolve().parent / "golden" / "anchor.compiled.json"


def test_compiled_matches_golden_bytes(anchor_compiled: CompiledSafetyModel) -> None:
    produced = canonical_json(anchor_compiled.model_dump(mode="json"))
    expected = GOLDEN.read_bytes()
    assert produced == expected, (
        "compiled IR drifted from the golden. If this change is intentional, "
        "regenerate with scripts/gen_golden.py and review the diff as a safety artifact."
    )


def test_recompile_is_byte_identical(anchor_document: SafetyDocument) -> None:
    a = compile_spec(anchor_document)
    b = compile_spec(anchor_document)
    assert canonical_json(a.model_dump(mode="json")) == canonical_json(b.model_dump(mode="json"))
    assert a.content_hash() == b.content_hash()


def test_golden_hash_is_stable(anchor_compiled: CompiledSafetyModel) -> None:
    # The golden file's own content hash matches a freshly compiled model's.
    from astro_mine.guard.spec.ir import CompiledSafetyModel as _CSM

    golden_model = _CSM.model_validate_json(GOLDEN.read_bytes())
    assert golden_model.content_hash() == anchor_compiled.content_hash()
