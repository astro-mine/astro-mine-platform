"""The trust root: whose signature counts, and how that answer changes safely (platform#22).

Seal could always say "this signature is intact and bound to this artifact". It could not say "a
party we trust made it" -- ``verify_signature`` took one optional key, so the answer was either
*anybody* (omit it) or *exactly one key, forever* (supply it). Neither is a policy a production
deployment can hold, and the second is why rotation was not a procedure anyone could run.

These assert the properties the decision in conventions.md §9 turns on, and the rotation test is the
one that justifies the whole design: a set makes the successor and the predecessor valid at once, so
there is no instant at which correctly signed artifacts stop verifying.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from astro_mine.seal import (
    TRUST_ROOT_ENV,
    SignatureError,
    TrustedKey,
    TrustRoot,
    TrustRootError,
    default_trust_root,
    generate_keypair,
    load_trust_root,
    resolve_trust_root,
    same_key,
    sign_digest,
    trust_root_from_env,
    verify_signature,
)

DIGEST = "sha256:" + "ab" * 32


@pytest.fixture(scope="module")
def alice() -> tuple[bytes, bytes]:
    return generate_keypair()


@pytest.fixture(scope="module")
def bob() -> tuple[bytes, bytes]:
    return generate_keypair()


# --- key identity -------------------------------------------------------------------------------


def test_the_same_key_formatted_differently_is_the_same_key(alice: tuple[bytes, bytes]) -> None:
    """Identity is canonical DER, not text -- a re-wrapped PEM must not read as an attack."""
    _, public = alice
    rewrapped = b"\n".join(line.strip() for line in public.strip().splitlines()) + b"\n\n\n"
    assert same_key(public, rewrapped)
    assert TrustRoot.single(public).accepts(rewrapped)


def test_unloadable_material_is_never_equal_to_anything() -> None:
    """A malformed key must not be the reason a signature is trusted -- including against itself."""
    assert not same_key(b"-----BEGIN PUBLIC KEY-----\nnope\n-----END PUBLIC KEY-----", b"nope")
    assert not same_key(b"garbage", b"garbage")


# --- the set, and what it is for ----------------------------------------------------------------


def test_an_untrusted_signer_is_refused(
    alice: tuple[bytes, bytes], bob: tuple[bytes, bytes]
) -> None:
    signature = sign_digest(DIGEST, bob[0])
    with pytest.raises(SignatureError, match="not in the trust root"):
        verify_signature(signature, DIGEST, trust_root=TrustRoot.single(alice[1], identity="alice"))


def test_an_empty_root_trusts_nobody(alice: tuple[bytes, bytes]) -> None:
    """Empty is a real state, not the same as "do not check" -- conflating them opens a door."""
    signature = sign_digest(DIGEST, alice[0])
    with pytest.raises(SignatureError):
        verify_signature(signature, DIGEST, trust_root=TrustRoot())
    # ...whereas passing no root at all still accepts any intact signature, unchanged.
    verify_signature(signature, DIGEST)


def test_passing_both_a_root_and_a_key_is_refused(alice: tuple[bytes, bytes]) -> None:
    """Two answers to "whose signature counts" is a bug, not a configuration."""
    signature = sign_digest(DIGEST, alice[0])
    with pytest.raises(SignatureError, match="not both"):
        verify_signature(
            signature,
            DIGEST,
            trusted_public_key_pem=alice[1],
            trust_root=TrustRoot.single(alice[1]),
        )


def test_a_single_key_still_behaves_exactly_as_before(alice: tuple[bytes, bytes], bob) -> None:
    """The one-key case is preserved, so existing callers keep their semantics."""
    verify_signature(sign_digest(DIGEST, alice[0]), DIGEST, trusted_public_key_pem=alice[1])
    with pytest.raises(SignatureError):
        verify_signature(sign_digest(DIGEST, bob[0]), DIGEST, trusted_public_key_pem=alice[1])


# --- rotation: the reason a root is a set -------------------------------------------------------


def test_rotation_is_an_overlap_not_a_flag_day(alice, bob) -> None:
    """The property the whole design exists for.

    Old and new are both valid through the overlap window, so nothing signed by either stops
    verifying at any instant. With a single key there is a moment where every artifact signed by the
    predecessor becomes untrusted -- which is why rotation was previously not safely runnable.
    """
    cutover = datetime(2026, 6, 1, tzinfo=UTC)
    overlap_ends = cutover + timedelta(days=30)
    root = TrustRoot.of(
        TrustedKey(identity="old", public_key_pem=alice[1], not_after=overlap_ends),
        TrustedKey(identity="new", public_key_pem=bob[1], not_before=cutover),
    )
    old_sig, new_sig = sign_digest(DIGEST, alice[0]), sign_digest(DIGEST, bob[0])

    during = cutover + timedelta(days=1)
    verify_signature(old_sig, DIGEST, trust_root=root, at=during)  # predecessor still honoured
    verify_signature(new_sig, DIGEST, trust_root=root, at=during)  # successor already honoured

    # Before the cutover the successor is not yet trusted...
    with pytest.raises(SignatureError, match="validity window"):
        verify_signature(new_sig, DIGEST, trust_root=root, at=cutover - timedelta(days=1))
    # ...and after the overlap the predecessor is retired, which is what revocation looks like.
    with pytest.raises(SignatureError, match="validity window"):
        verify_signature(old_sig, DIGEST, trust_root=root, at=overlap_ends + timedelta(days=1))


def test_a_key_that_is_never_valid_is_refused_at_construction(alice) -> None:
    with pytest.raises(TrustRootError, match="never valid"):
        TrustedKey(
            identity="backwards",
            public_key_pem=alice[1],
            not_before=datetime(2027, 1, 1, tzinfo=UTC),
            not_after=datetime(2026, 1, 1, tzinfo=UTC),
        )


def test_a_key_needs_an_identity(alice) -> None:
    """An anonymous root cannot be audited, and an audit trail is the point of naming signers."""
    with pytest.raises(TrustRootError, match="identity"):
        TrustedKey(identity="  ", public_key_pem=alice[1])


# --- per-kind scoping ---------------------------------------------------------------------------


def test_a_scoped_key_signs_only_its_kinds(alice) -> None:
    root = TrustRoot.of(
        TrustedKey(identity="worlds-only", public_key_pem=alice[1], kinds=frozenset({"world"}))
    )
    signature = sign_digest(DIGEST, alice[0])
    verify_signature(signature, DIGEST, trust_root=root, kind="world")
    with pytest.raises(SignatureError, match="may sign only"):
        verify_signature(signature, DIGEST, trust_root=root, kind="policy")


def test_an_unscoped_key_covers_every_kind(alice) -> None:
    root = TrustRoot.single(alice[1], identity="org")
    signature = sign_digest(DIGEST, alice[0])
    for kind in ("world", "policy", "asset", None):
        verify_signature(signature, DIGEST, trust_root=root, kind=kind)


# --- the refusal has to be actionable -----------------------------------------------------------


def test_the_rejection_says_which_of_the_three_reasons_it_was(alice, bob) -> None:
    """ "Not trusted" sends a reader hunting; the specific reason sends them to the fix."""
    expired = TrustRoot.of(
        TrustedKey(
            identity="old", public_key_pem=alice[1], not_after=datetime(2020, 1, 1, tzinfo=UTC)
        )
    )
    scoped = TrustRoot.of(
        TrustedKey(identity="w", public_key_pem=alice[1], kinds=frozenset({"world"}))
    )
    unknown = TrustRoot.single(bob[1], identity="somebody-else")

    assert "validity window" in expired.why_rejected(alice[1])
    assert "may sign only" in scoped.why_rejected(alice[1], kind="policy")
    assert "not in the trust root" in unknown.why_rejected(alice[1])
    # ...and it names who *is* trusted, so the reader can tell a misconfiguration from an intrusion.
    assert "somebody-else" in unknown.why_rejected(alice[1])


def test_match_returns_the_key_so_an_audit_record_can_name_the_signer(alice) -> None:
    root = TrustRoot.single(alice[1], identity="astro-mine-release-2026")
    matched = root.match(alice[1])
    assert matched is not None and matched.identity == "astro-mine-release-2026"


# --- the document, and where a root comes from --------------------------------------------------


def test_a_document_round_trips(alice, bob) -> None:
    from astro_mine.seal._trust import to_document

    root = TrustRoot.of(
        TrustedKey(
            identity="a", public_key_pem=alice[1], not_after=datetime(2027, 1, 1, tzinfo=UTC)
        ),
        TrustedKey(identity="b", public_key_pem=bob[1], kinds=frozenset({"world", "policy"})),
    )
    restored = load_trust_root(to_document(root))
    assert restored.identities() == ("a", "b")
    assert restored.accepts(alice[1], at=datetime(2026, 1, 1, tzinfo=UTC))
    assert restored.accepts(bob[1], kind="world")
    assert not restored.accepts(bob[1], kind="asset")


def test_an_unknown_field_is_refused_not_ignored(alice) -> None:
    """Same rule as Seal's unknown-`require`-token: a typo must not quietly disable a check."""
    with pytest.raises(TrustRootError, match="unknown trust-root field"):
        load_trust_root(json.dumps({"trust_root_version": "0.1", "keys": [], "requre": []}))
    with pytest.raises(TrustRootError, match="unknown field"):
        load_trust_root(
            json.dumps(
                {"keys": [{"identity": "a", "public_key_pem": alice[1].decode(), "kindz": []}]}
            )
        )


def test_a_naive_timestamp_is_read_as_utc_rather_than_exploding_in_a_gate(alice) -> None:
    """A naive/aware comparison would raise inside a gate, on an artifact that may be fine."""
    root = load_trust_root(
        json.dumps(
            {
                "keys": [
                    {
                        "identity": "a",
                        "public_key_pem": alice[1].decode(),
                        "not_after": "2030-01-01T00:00:00",
                    }
                ]
            }
        )
    )
    assert root.accepts(alice[1])


def test_malformed_documents_fail_closed() -> None:
    for document in ("not json", json.dumps([1, 2]), json.dumps({"keys": "nope"})):
        with pytest.raises(TrustRootError):
            load_trust_root(document)


def test_the_packaged_default_is_empty_and_therefore_fails_closed() -> None:
    """The wheel ships a root so the offline tier has one (CX-LOCAL); it grants nothing yet.

    An empty packaged root can never be the reason an artifact is accepted, which is the safe
    default for a file that is about to be populated by a rotation procedure rather than a release.
    """
    assert default_trust_root().keys == ()


def test_the_environment_overrides_the_packaged_root(tmp_path, alice) -> None:
    from astro_mine.seal._trust import to_document

    path = tmp_path / "trust.json"
    path.write_text(
        to_document(TrustRoot.single(alice[1], identity="deployment")), encoding="utf-8"
    )
    root = trust_root_from_env({TRUST_ROOT_ENV: str(path)})
    assert root.identities() == ("deployment",)


def test_an_unreadable_env_root_fails_loudly(tmp_path) -> None:
    """Silently falling back to the packaged root would hide a deployment's misconfiguration."""
    with pytest.raises(TrustRootError, match="cannot read trust root"):
        trust_root_from_env({TRUST_ROOT_ENV: str(tmp_path / "absent.json")})


def test_resolution_precedence(tmp_path, alice, bob) -> None:
    """explicit → single key → environment → packaged (conventions.md §9)."""
    from astro_mine.seal._trust import to_document

    path = tmp_path / "trust.json"
    path.write_text(to_document(TrustRoot.single(bob[1], identity="env")), encoding="utf-8")
    env = {TRUST_ROOT_ENV: str(path)}

    explicit = TrustRoot.single(alice[1], identity="explicit")
    assert resolve_trust_root(explicit, env=env).identities() == ("explicit",)
    assert resolve_trust_root(None, trusted_public_key_pem=alice[1], env=env).identities() == (
        "unnamed",
    )
    assert resolve_trust_root(None, env=env).identities() == ("env",)
    # Nothing configured anywhere → None, which preserves "do not check identity" for callers
    # that have not opted in. It is not an empty root, which would mean "trust nobody".
    assert resolve_trust_root(None, env={}) is None
