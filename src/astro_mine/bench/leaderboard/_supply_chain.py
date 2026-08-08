"""Fail-closed supply-chain verification of a Hub submission (bench#29; bench.md §9).

bench.md §9: *"Submission artifacts (ONNX/OCI) are verified via **Sigstore/cosign** signatures and
**SLSA** provenance; SBOMs (Syft/CycloneDX) recorded."* Until bench#29, Hub-digest intake verified
**content hashes only** — it proved the bytes had not been *corrupted*, but nothing about who
produced them or how. A content hash is not a signature: an attacker who can write to the registry
can publish a coherent, self-consistent artifact whose every blob hashes correctly.

This module closes that gap, and it does so by **reusing the platform's existing verify primitives
rather than re-implementing signing** (RFC-0005 — Seal is the single home for `cryptography`, and
the one artifact-integrity implementation):

- :func:`astro_mine.hub.supply_chain.verify` — Hub's registry-plane *verify-twice* orchestration,
  which is itself built on :mod:`astro_mine.seal` (``verify_signature`` / ``verify_slsa_document`` /
  ``verify_sbom_document``, cosign ECDSA P-256). Bench adds **no crypto of its own** — it has no
  ``cryptography`` dependency and no signature code to get wrong.

The check runs **before the submission executes** — before the policy reference is ever handed to
the sandbox — and it is required, not optional: :data:`REQUIRED_EVIDENCE` is Seal's
``DEFAULT_REQUIRED`` (``signature``, ``slsa``, ``sbom``), and a submission missing *any* of them,
or carrying one that does not verify, is **rejected** (bench#29 AC3: "verification failure fails
closed — submission rejected, not silently accepted").

``trusted_public_key_pem`` pins the signer: with it, only artifacts signed by that key are admitted.
The key is **public** material loaded from a path (:data:`TRUSTED_KEY_ENV`), never a secret in the
image or the repo (conventions.md §9).

The local/offline tier does not use Hub intake at all, so none of this touches the account-free
``run(spec, policy)`` path (CX-LOCAL).

Backlog: bench#29 — astro-mine-bench#29
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict

__all__ = [
    "REQUIRED_EVIDENCE",
    "TRUSTED_KEY_ENV",
    "AttestationPolicy",
    "AttestationVerdict",
    "SupplyChainRejected",
    "attestation_policy_from_env",
    "verify_submission_attestations",
]

#: Env var naming a **file** holding the trusted cosign *public* key (PEM). Unset ⇒ any intact,
#: self-consistent cosign signature is accepted (the artifact is still required to *have* one, with
#: SLSA provenance and an SBOM); set ⇒ only that signer's artifacts are admitted.
TRUSTED_KEY_ENV = "ASTRO_MINE_BENCH_TRUSTED_KEY"

#: The evidence a submission must carry, per bench.md §9. This is Seal's ``DEFAULT_REQUIRED``
#: (``signature``, ``slsa``, ``sbom``) — Bench does not weaken the platform's required set.
REQUIRED_EVIDENCE: tuple[str, ...] = ("signature", "slsa", "sbom")


class SupplyChainRejected(Exception):
    """A submission's attestations are missing or invalid — it is rejected, never executed."""


@dataclass(frozen=True, slots=True)
class AttestationPolicy:
    """What a submission must prove about itself before the evaluator will run it.

    ``required`` is the evidence set (cosign signature + SLSA provenance + SBOM by default).
    ``trusted_public_key_pem`` pins the acceptable signer; ``None`` accepts any intact cosign
    signature — weaker, but still fail-closed on a *missing* or *broken* one.
    """

    required: tuple[str, ...] = REQUIRED_EVIDENCE
    trusted_public_key_pem: bytes | None = None

    def __post_init__(self) -> None:
        if not self.required:
            raise ValueError(
                "AttestationPolicy.required must not be empty — an unverified submission is "
                "arbitrary code from an unknown author (bench.md §9)"
            )


class AttestationVerdict(BaseModel):
    """The outcome of verifying one submission's attestations — an audit-log record (bench#29 AC4).

    Recorded for **every** Hub submission, verified or rejected, so a dispute over an entry can be
    settled from the trail: what was checked, against which signer, and what the answer was.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    #: The Hub image-manifest digest the evidence is bound to.
    subject: str
    required: tuple[str, ...]
    verified: bool
    #: Whether the signature was pinned to a trusted key, or merely checked for integrity.
    signer_pinned: bool = False
    #: Why it failed (``None`` when it verified).
    detail: str | None = None


def verify_submission_attestations(
    registry: Any, subject: str, policy: AttestationPolicy
) -> AttestationVerdict:
    """Verify a Hub submission's cosign signature, SLSA provenance, and SBOM — or reject it.

    Delegates to :func:`astro_mine.hub.supply_chain.verify`, the platform's verify-twice
    orchestration over Seal's primitives (RFC-0005): it re-checks that the artifact's bytes hash to
    their content addresses, that a **cosign** signature over ``subject`` exists and verifies (to
    to ``policy.trusted_public_key_pem`` when given), that **SLSA** provenance is attached and
    well-shaped, and that an **SBOM** is present.

    Raises :class:`SupplyChainRejected` on any missing or invalid evidence — the submission is not
    executed. Returns the :class:`AttestationVerdict` to be audit-logged on success.
    """
    from astro_mine.hub.supply_chain import SupplyChainError
    from astro_mine.hub.supply_chain import verify as hub_verify

    signer_pinned = policy.trusted_public_key_pem is not None
    try:
        # Hub types this on its concrete Registry; Bench holds the structural HubRegistry protocol
        # (bench.md §2.2 — Bench never reaches into a private Hub schema), so the cast is the seam.
        hub_verify(
            cast(Any, registry),
            subject,
            trusted_public_key_pem=policy.trusted_public_key_pem,
            require=list(policy.required),
        )
    except SupplyChainError as exc:
        raise SupplyChainRejected(
            f"submission {subject} failed supply-chain verification "
            f"(required: {', '.join(policy.required)}): {exc}"
        ) from exc
    except Exception as exc:
        raise SupplyChainRejected(
            f"submission {subject} could not be supply-chain verified: {type(exc).__name__}: {exc}"
        ) from exc

    return AttestationVerdict(
        subject=subject,
        required=policy.required,
        verified=True,
        signer_pinned=signer_pinned,
        detail=None,
    )


def attestation_policy_from_env(
    env: Mapping[str, str] | None = None, *, required: Sequence[str] = REQUIRED_EVIDENCE
) -> AttestationPolicy:
    """Build the deployment's attestation policy, loading the trusted **public** key from its path.

    Note what is *not* configurable: whether to verify at all. A deployment can pin a signer, but it
    cannot switch the requirement off — an unverified community artifact is arbitrary code from an
    unknown author (bench.md §9).
    """
    environment = os.environ if env is None else env
    key_path = environment.get(TRUSTED_KEY_ENV)
    trusted: bytes | None = None
    if key_path:
        path = Path(key_path)
        if not path.is_file():
            raise SupplyChainRejected(
                f"{TRUSTED_KEY_ENV} points at {key_path!r}, which is not a readable file; "
                "refusing to start with an unresolvable trust root"
            )
        trusted = path.read_bytes()
    return AttestationPolicy(required=tuple(required), trusted_public_key_pem=trusted)
