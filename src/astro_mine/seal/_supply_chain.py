# SPDX-License-Identifier: Apache-2.0
"""Verify-twice orchestration — the required-evidence policy and the fail-closed check.

The consumer half of the supply-chain trust boundary (guard.md §9.5; hub.md §9; conventions.md §9):
*which* evidence a verified artifact must carry, *what makes each attestation well-shaped*, and the
:func:`verify` orchestrator that enforces both. It is the **same check at admission (publish) and at
pull** — the "verify twice" of hub.md §9 — so a compromised registry cannot make a consumer accept
tampered bytes.

:func:`verify` is **registry-agnostic**: it drives the
:class:`~astro_mine.seal._attest.AttestationStore` port (``str`` / ``bytes`` only), never a concrete
registry type, so Seal keeps its Core-only dependency (seal.md §2.1) and the policy lives in exactly
one place for every consumer — Hub's pull gate, Bench's submission check, Guard's load gate
(RM-P1-SEAL-03; RFC-0005 §Sequencing).

**Everything fails closed.** :func:`verify` returns ``None`` *only* when every required piece of
evidence is present, intact, and valid. A tampered artifact, a tampered attestation, a missing or
bad signature, an untrusted signing key, absent provenance, an absent SBOM, an unparseable document,
an unknown policy token, or *any* error raised by the store all raise :class:`SupplyChainError`.
There is no permissive default and no soft "maybe": an error is never read as "no evidence,
therefore fine" (seal.md §2.4, §9).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from astro_mine.core.registry import Signature
from astro_mine.seal._attest import (
    ARTIFACT_TYPE_SBOM,
    ARTIFACT_TYPE_SIGNATURE,
    ARTIFACT_TYPE_SLSA,
    SLSA_PREDICATE_TYPE,
    AttestationStore,
)
from astro_mine.seal._signing import verify_signature
from astro_mine.seal._trust import TrustRoot

__all__ = [
    "DEFAULT_REQUIRED",
    "AttestationError",
    "SupplyChainError",
    "verify",
    "verify_sbom_document",
    "verify_slsa_document",
]

#: The attestations a verified/curated artifact must carry (hub.md §9; `LUNAR-SR-002`).
DEFAULT_REQUIRED: tuple[str, ...] = ("signature", "slsa", "sbom")

#: Every evidence kind :func:`verify` knows how to check. A ``require`` token outside this set names
#: a policy the caller believes is enforced but that nothing would check — so it is refused rather
#: than ignored: a silent typo must never become a silent bypass.
_KNOWN_KINDS: frozenset[str] = frozenset(DEFAULT_REQUIRED)


class AttestationError(Exception):
    """An attestation document is missing or wrong-shaped (always fail closed)."""


class SupplyChainError(Exception):
    """An artifact fails a supply-chain check at admission or pull (always fail closed)."""


def verify_slsa_document(doc: Mapping[str, Any]) -> None:
    """Check a parsed SLSA provenance ``doc`` is well-shaped — raise :class:`AttestationError` else.

    The pure predicate a verifier applies to the SLSA attestation payload once its integrity is
    re-established. Fail-closed on a bad ``predicateType``.
    """
    if doc.get("predicateType") != SLSA_PREDICATE_TYPE:
        raise AttestationError("SLSA provenance has the wrong predicateType")


def verify_sbom_document(doc: Mapping[str, Any]) -> None:
    """Check a parsed SBOM ``doc`` is a CycloneDX bill of materials — raise on any other format.

    The pure predicate a verifier applies to the SBOM attestation payload once its integrity is
    re-established. Fail-closed on a non-CycloneDX ``bomFormat``.
    """
    if doc.get("bomFormat") != "CycloneDX":
        raise AttestationError("SBOM is not CycloneDX")


def _digests(
    store: AttestationStore, subject: str, artifact_type: str, label: str
) -> Sequence[str]:
    """``subject``'s attestation digests of ``artifact_type`` — a store failure is a *refusal*."""
    try:
        return store.attestation_digests(subject, artifact_type=artifact_type)
    except Exception as exc:  # any store failure fails closed — never "no evidence, therefore fine"
        raise SupplyChainError(f"could not read {label} attestations: {exc}") from exc


def _payload(store: AttestationStore, digest: str, label: str) -> bytes:
    """The attestation payload at ``digest``. The store re-hashes it: a tampered read raises."""
    try:
        return store.read_attestation(digest)
    except Exception as exc:  # includes the store's integrity error on a tampered attestation
        raise SupplyChainError(f"{label} integrity failed: {exc}") from exc


def _document(store: AttestationStore, digest: str, label: str) -> Mapping[str, Any]:
    """A parsed attestation payload — raise unless it is intact *and* a JSON object."""
    payload = _payload(store, digest, label)
    try:
        doc = json.loads(payload)
    except ValueError as exc:
        raise SupplyChainError(f"{label} is not valid JSON: {exc}") from exc
    if not isinstance(doc, Mapping):
        raise SupplyChainError(f"{label} is not a JSON object")
    return doc


def _check_signatures(
    store: AttestationStore,
    subject: str,
    trusted_public_key_pem: bytes | None,
    trust_root: TrustRoot | None = None,
    kind: str | None = None,
) -> None:
    """*Every* attached signature must be intact and verify over ``subject`` — else refuse.

    Checking every signature (rather than accepting the first that passes) is what stops an attacker
    from *appending* one: a bad signature cannot hide behind a good one.
    """
    digests = _digests(store, subject, ARTIFACT_TYPE_SIGNATURE, "signature")
    if not digests:
        raise SupplyChainError("no cosign signature attached")
    for digest in digests:
        payload = _payload(store, digest, "signature")
        try:
            signature = Signature.model_validate_json(payload)
            verify_signature(
                signature,
                subject,
                trusted_public_key_pem=trusted_public_key_pem,
                trust_root=trust_root,
                kind=kind,
            )
        except Exception as exc:  # SignatureError, a malformed envelope, an untrusted key, ...
            raise SupplyChainError(f"signature verification failed: {exc}") from exc


def _check_documents(
    store: AttestationStore,
    subject: str,
    *,
    artifact_type: str,
    label: str,
    check: Callable[[Mapping[str, Any]], None],
) -> None:
    """``subject`` must carry >=1 attestation of ``artifact_type``, and all must be valid."""
    digests = _digests(store, subject, artifact_type, label)
    if not digests:
        raise SupplyChainError(f"no {label} attached")
    for digest in digests:
        doc = _document(store, digest, label)
        try:
            check(doc)
        except AttestationError as exc:
            raise SupplyChainError(str(exc)) from exc


def verify(
    store: AttestationStore,
    subject: str,
    *,
    trusted_public_key_pem: bytes | None = None,
    trust_root: TrustRoot | None = None,
    kind: str | None = None,
    require: Sequence[str] = DEFAULT_REQUIRED,
) -> None:
    """Re-verify ``subject``'s integrity and required attestations — **raise on any failure**.

    The one verify-twice check, run identically at admission and at pull (hub.md §2.3, §9). In
    order: ``subject``'s own bytes hash to their addresses; every attached cosign signature is
    intact and verifies over ``subject`` (pinned to ``trusted_public_key_pem`` when given); SLSA
    provenance is present, intact, and well-shaped; an SBOM is present, intact, and CycloneDX.

    ``require`` is the required-evidence policy — :data:`DEFAULT_REQUIRED` by default. A token it
    does not know is **refused**, not ignored, so a typo can never quietly disable a check.

    ``trust_root`` decides *whose* signature counts — a :class:`~astro_mine.seal.TrustRoot` is a
    **set** of signers with validity windows, so a rotation is an overlap rather than a flag day
    (conventions.md §9). ``trusted_public_key_pem`` is the one-key case: a root of one.
    ``kind`` selects among per-kind-scoped keys.

    Omit both and a signature must still be present, intact, and bound to ``subject`` — but *any*
    key satisfies it, which proves integrity and self-consistency, **not** that a trusted party
    signed. A gate that decides trust (a Hub pull, a Bench submission, a Guard load) MUST pass one.

    Returns ``None`` only when every required check passes; otherwise raises
    :class:`SupplyChainError` — the single outcome for a tampered artifact, a tampered or missing
    attestation, a bad signature, an untrusted key, an unparseable document, or a failing store.
    """
    unknown = sorted(set(require) - _KNOWN_KINDS)
    if unknown:
        raise SupplyChainError(f"unknown required-evidence kind(s): {', '.join(unknown)}")

    try:
        store.verify_integrity(subject)
    except Exception as exc:  # a missing subject and a tampered blob both land here
        raise SupplyChainError(f"artifact integrity check failed: {exc}") from exc

    if "signature" in require:
        _check_signatures(store, subject, trusted_public_key_pem, trust_root, kind)
    if "slsa" in require:
        _check_documents(
            store,
            subject,
            artifact_type=ARTIFACT_TYPE_SLSA,
            label="SLSA provenance",
            check=verify_slsa_document,
        )
    if "sbom" in require:
        _check_documents(
            store,
            subject,
            artifact_type=ARTIFACT_TYPE_SBOM,
            label="SBOM",
            check=verify_sbom_document,
        )
