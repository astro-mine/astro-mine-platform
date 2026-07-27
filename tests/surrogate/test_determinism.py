"""Golden determinism references (RM-P1-SURR-01; conventions.md §11).

Seeded/fixed content reproduces stored references, or CI fails. The ErrorReport content
address and the wire-form digest are the reproducibility anchors — a surrogate's
ErrorReport *is* its performance claim, and it must be reproducible (surrogate.md §8). If
a change here is intentional, update the constants; a silent drift is a bug.
"""

from __future__ import annotations

import hashlib

from astro_mine.surrogate.wire import error_report_to_wire
from tests.surrogate.factories import granular_report, illumination_report

_GRANULAR_CONTENT_HASH = "sha256:697025ebada6fe5017e938d3fd938d3e3c5e355193d97b2774826d93441d0d7f"
_ILLUMINATION_CONTENT_HASH = (
    "sha256:4a4e77cd91fd122387b411d1157f206f7e1c3b0c7663823ff699e2b111cfb22b"
)
_GRANULAR_WIRE_SHA = "sha256:db9157e2bc9b6c824c004abe22ef2e9db576e934bf0d4d7ce47c6401e19db00e"


def test_granular_report_content_hash_is_stable() -> None:
    assert granular_report().content_hash() == _GRANULAR_CONTENT_HASH


def test_illumination_report_content_hash_is_stable() -> None:
    assert illumination_report().content_hash() == _ILLUMINATION_CONTENT_HASH


def test_granular_wire_form_digest_is_stable() -> None:
    digest = "sha256:" + hashlib.sha256(error_report_to_wire(granular_report())).hexdigest()
    assert digest == _GRANULAR_WIRE_SHA
