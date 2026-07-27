"""ImageRef: digest pinning, parsing, canonicalization, strictness."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from astro_mine.cloud.packaging import ImageRef

DIGEST = "sha256:" + "ab" * 32
REPO = "ghcr.io/astro-mine/astro-mine-sim"


def test_reference_and_str() -> None:
    ref = ImageRef(repository=REPO, digest=DIGEST, tag="0.1.0")
    assert ref.reference == f"{REPO}@{DIGEST}"
    assert str(ref) == ref.reference
    assert ref.tag == "0.1.0"


def test_parse_round_trips_a_pinned_reference() -> None:
    ref = ImageRef.parse(f"{REPO}@{DIGEST}", tag="latest")
    assert ref.repository == REPO
    assert ref.digest == DIGEST
    assert ref.tag == "latest"


def test_parse_rejects_unpinned_reference() -> None:
    with pytest.raises(ValueError, match="unpinned"):
        ImageRef.parse(f"{REPO}:0.1.0")


def test_bare_hex_digest_is_canonicalized() -> None:
    # Sim's bare-hex form is accepted and normalized to the canonical sha256: form.
    ref = ImageRef(repository=REPO, digest="ab" * 32)
    assert ref.digest == DIGEST


@pytest.mark.parametrize("bad", ["sha512:" + "0" * 128, "sha256:xyz", "not-a-digest"])
def test_malformed_digest_is_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        ImageRef(repository=REPO, digest=bad)


@pytest.mark.parametrize("bad", ["", "  ", "has space", f"{REPO}@{DIGEST}"])
def test_bad_repository_is_rejected(bad: str) -> None:
    with pytest.raises(ValidationError):
        ImageRef(repository=bad, digest=DIGEST)


def test_extra_fields_rejected_and_frozen() -> None:
    with pytest.raises(ValidationError):
        ImageRef(repository=REPO, digest=DIGEST, oops="no")  # type: ignore[call-arg]
    ref = ImageRef(repository=REPO, digest=DIGEST)
    with pytest.raises(ValidationError):
        ref.repository = "mutated"  # type: ignore[misc]
