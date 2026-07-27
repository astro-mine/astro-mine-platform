"""Content-addressing convention: sha256, canonical JSON, ``sha256:<hex>`` form."""

from __future__ import annotations

import hashlib
import json

import pytest

from astro_mine.cloud.artifacts import addressing


def test_bytes_are_hashed_raw() -> None:
    expected = "sha256:" + hashlib.sha256(b"hello").hexdigest()
    assert addressing.content_address(b"hello") == expected
    assert addressing.content_address(bytearray(b"hello")) == expected


def test_structured_payload_uses_canonical_json() -> None:
    canonical = json.dumps({"a": 1, "b": 2}, sort_keys=True, separators=(",", ":")).encode()
    expected = "sha256:" + hashlib.sha256(canonical).hexdigest()
    assert addressing.content_address({"b": 2, "a": 1}) == expected


def test_mapping_key_order_is_irrelevant() -> None:
    assert addressing.content_address({"x": 1, "y": 2}) == addressing.content_address(
        {"y": 2, "x": 1}
    )


def test_matches_sims_bare_hex_convention() -> None:
    # Sim's content_digest returns bare hex of the same canonical JSON; Cloud just
    # prefixes it. Stripping the prefix must recover Sim's value exactly.
    payload = {"scenario": "lunar", "seed": 7}
    sim_hex = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()
    assert addressing.hex_of(addressing.content_address(payload)) == sim_hex


def test_format_and_parse_round_trip() -> None:
    hexdigest = hashlib.sha256(b"x").hexdigest()
    address = addressing.format_address(hexdigest)
    assert address == f"sha256:{hexdigest}"
    assert addressing.parse_address(address) == ("sha256", hexdigest)


def test_parse_accepts_bare_hex() -> None:
    hexdigest = hashlib.sha256(b"x").hexdigest()
    assert addressing.parse_address(hexdigest) == ("sha256", hexdigest)
    assert addressing.hex_of(hexdigest) == hexdigest


def test_parse_rejects_unknown_algorithm() -> None:
    hexdigest = hashlib.sha256(b"x").hexdigest()
    with pytest.raises(ValueError, match="unsupported digest algorithm"):
        addressing.parse_address(f"md5:{hexdigest}")


@pytest.mark.parametrize("bad", ["sha256:deadbeef", "sha256:" + "z" * 64, "nothex"])
def test_parse_rejects_malformed_digest(bad: str) -> None:
    with pytest.raises(ValueError, match=r"malformed|unsupported"):
        addressing.parse_address(bad)


def test_format_rejects_malformed_hex() -> None:
    with pytest.raises(ValueError, match="malformed"):
        addressing.format_address("tooshort")
