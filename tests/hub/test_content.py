"""Content-address primitive tests (mirrors the platform canonical form)."""

from __future__ import annotations

from astro_mine.hub._content import canonical_json, content_hash, content_hash_json


def test_canonical_json_is_sorted_and_compact() -> None:
    assert canonical_json({"b": 1, "a": 2}) == b'{"a":2,"b":1}'


def test_content_hash_is_prefixed_and_stable() -> None:
    assert content_hash(b"x").startswith("sha256:")
    assert content_hash(b"x") == content_hash(b"x")
    assert content_hash(b"x") != content_hash(b"y")


def test_content_hash_json_composes() -> None:
    assert content_hash_json({"a": 1}) == content_hash(canonical_json({"a": 1}))
