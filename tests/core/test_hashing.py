"""Tests for the canonical content-hash helper (issue #19)."""

from __future__ import annotations

import hashlib

import pytest

from astro_mine.core.hashing import HASH_ALGORITHM, canonical_json, content_hash, content_hash_json

# sha256 of the empty byte string — a fixed, externally-verifiable vector.
_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def test_content_hash_format() -> None:
    digest = content_hash(b"astro-mine")
    algo, _, hexpart = digest.partition(":")
    assert algo == HASH_ALGORITHM == "sha256"
    assert len(hexpart) == 64
    assert int(hexpart, 16) >= 0  # valid hex


def test_content_hash_known_vector() -> None:
    assert content_hash(b"") == f"sha256:{_EMPTY_SHA256}"


def test_content_hash_matches_hashlib_reference() -> None:
    data = b"the content hash is the task identity"
    assert content_hash(data) == f"sha256:{hashlib.sha256(data).hexdigest()}"


def test_content_hash_is_stable_and_input_sensitive() -> None:
    assert content_hash(b"a") == content_hash(b"a")
    assert content_hash(b"a") != content_hash(b"b")


def test_content_hash_rejects_str() -> None:
    # Fail loud: hashing must be over bytes, never an implicitly-encoded str.
    with pytest.raises(TypeError):
        content_hash("not-bytes")  # type: ignore[arg-type]


def test_canonical_json_exact_bytes() -> None:
    assert canonical_json({"b": 2, "a": 1}) == b'{"a":1,"b":2}'


def test_canonical_json_is_key_order_invariant() -> None:
    assert canonical_json({"a": 1, "b": 2}) == canonical_json({"b": 2, "a": 1})


def test_canonical_json_preserves_non_ascii() -> None:
    # ensure_ascii=False: characters are UTF-8 bytes, not \\uXXXX escapes.
    assert canonical_json({"site": "café"}) == '{"site":"café"}'.encode()


def test_canonical_json_rejects_non_serializable() -> None:
    with pytest.raises(TypeError):
        canonical_json({"bad": {1, 2, 3}})  # a set is not JSON-serializable


def test_content_hash_json_composes_helpers() -> None:
    obj = {"scenario": "lunar-polar-ice", "seed": 7}
    assert content_hash_json(obj) == content_hash(canonical_json(obj))


def test_content_hash_json_is_key_order_invariant() -> None:
    assert content_hash_json({"a": 1, "b": 2}) == content_hash_json({"b": 2, "a": 1})
