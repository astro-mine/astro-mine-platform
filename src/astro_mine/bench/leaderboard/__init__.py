# SPDX-License-Identifier: Apache-2.0
"""Public leaderboard — submit-policy-we-run + Hub-digest intake (RM-P1-BENCH-10; bench.md §9).

A submit-policy-we-run leaderboard: a submitted policy is executed server-side on the scenario's
embargoed **held-out seeds** and scored, a fraction is **re-executed from its provenance bundle** to
verify integrity, and the content-addressed result is ranked — the anti-gaming integrity baseline
and the seed of the academic flywheel. Two intake paths share one catalog and one harness: a local
importable ``policy_ref`` (RM-P0-BENCH-06) and a community submission resolved from **Hub by
digest**
(RM-P1-BENCH-10). The service is a deployment of the local scoring library
(:func:`~astro_mine.bench.baseline.run`); it composes, it does not re-score.

The dependency-clean core (models, stores, evaluation, ranking, Hub-digest intake, provenance
bundle, and the :class:`LeaderboardService` orchestration) imports only ``core + pydantic``, so
``import astro_mine.bench.leaderboard`` works without the service deps; the concrete Hub
:class:`~astro_mine.hub.registry.Registry` (:func:`open_registry`), the FastAPI app, and the
SQLAlchemy store need the ``[leaderboard]`` extra and are imported lazily.

Backlog: RM-P1-BENCH-10 -- astro-mine-bench#18
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_mine.bench._hub_payload import PayloadRetrievalError, pull_verified_layer
from astro_mine.bench.leaderboard._audit import (
    AuditDecision,
    AuditEvent,
    AuditLog,
    InMemoryAuditLog,
    audit_event,
)
from astro_mine.bench.leaderboard._auth import (
    AuthenticationError,
    OidcTokenVerifier,
    Principal,
    TokenVerifier,
    bearer_token,
    oidc_verifier_from_env,
)
from astro_mine.bench.leaderboard._authz import (
    DEFAULT_QUOTAS,
    DEFAULT_ROLE_ACTIONS,
    Action,
    AuthorizationError,
    AuthorizationRequest,
    Decision,
    OpaPolicyEngine,
    PolicyEngine,
    RbacPolicyEngine,
    Role,
    policy_engine_from_env,
)
from astro_mine.bench.leaderboard._eval import (
    EMBARGO_ROOT,
    EMBARGO_ROOT_ENV,
    PolicyReferenceError,
    build_submission,
    evaluate,
    load_heldout_seeds,
    rank,
    resolve_embargo_root,
    resolve_policy,
    validate_policy_ref,
)
from astro_mine.bench.leaderboard._hub import (
    HubRegistry,
    HubResolutionError,
    ManifestInterfaceError,
    PolicyLoader,
    ResolvedSubmission,
    open_registry,
    reference_policy_loader,
    resolve_submission,
    submission_policy_ref,
    validate_submission_manifest,
)
from astro_mine.bench.leaderboard._jobs import (
    InMemoryJobQueue,
    InMemoryJobStore,
    InMemoryRateLimiter,
    JobQueue,
    JobRecord,
    JobStore,
    RateLimiter,
    RateLimitError,
    SubmissionEnvelope,
    SubmissionStatus,
)
from astro_mine.bench.leaderboard._models import (
    HubSubmissionRequest,
    Integrity,
    LeaderboardEntry,
    MetricScore,
    Submission,
    SubmissionRequest,
)
from astro_mine.bench.leaderboard._objects import (
    FileObjectStore,
    InMemoryObjectStore,
    ObjectIntegrityError,
    ObjectStore,
)
from astro_mine.bench.leaderboard._provenance import (
    ProvenanceBundle,
    SeedRecord,
    build_provenance_bundle,
    resample_from_bundle,
)
from astro_mine.bench.leaderboard._service import LeaderboardService, SubmissionRejected
from astro_mine.bench.leaderboard._store import InMemoryStore, LeaderboardStore
from astro_mine.bench.leaderboard._supply_chain import (
    REQUIRED_EVIDENCE,
    AttestationPolicy,
    AttestationVerdict,
    SupplyChainRejected,
    attestation_policy_from_env,
    verify_submission_attestations,
)

if TYPE_CHECKING:
    from fastapi import FastAPI

    from astro_mine.bench.zoo import ScenarioCatalog

__all__ = [
    "DEFAULT_QUOTAS",
    "DEFAULT_ROLE_ACTIONS",
    "EMBARGO_ROOT",
    "EMBARGO_ROOT_ENV",
    "REQUIRED_EVIDENCE",
    "Action",
    "AttestationPolicy",
    "AttestationVerdict",
    "AuditDecision",
    "AuditEvent",
    "AuditLog",
    "AuthenticationError",
    "AuthorizationError",
    "AuthorizationRequest",
    "Decision",
    "FileObjectStore",
    "HubRegistry",
    "HubResolutionError",
    "HubSubmissionRequest",
    "InMemoryAuditLog",
    "InMemoryJobQueue",
    "InMemoryJobStore",
    "InMemoryObjectStore",
    "InMemoryRateLimiter",
    "InMemoryStore",
    "Integrity",
    "JobQueue",
    "JobRecord",
    "JobStore",
    "LeaderboardEntry",
    "LeaderboardService",
    "LeaderboardStore",
    "ManifestInterfaceError",
    "MetricScore",
    "ObjectIntegrityError",
    "ObjectStore",
    "OidcTokenVerifier",
    "OpaPolicyEngine",
    "PayloadRetrievalError",
    "PolicyEngine",
    "PolicyLoader",
    "PolicyReferenceError",
    "Principal",
    "ProvenanceBundle",
    "RateLimitError",
    "RateLimiter",
    "RbacPolicyEngine",
    "ResolvedSubmission",
    "Role",
    "SeedRecord",
    "Submission",
    "SubmissionEnvelope",
    "SubmissionRejected",
    "SubmissionRequest",
    "SubmissionStatus",
    "SupplyChainRejected",
    "TokenVerifier",
    "attestation_policy_from_env",
    "audit_event",
    "bearer_token",
    "build_provenance_bundle",
    "build_submission",
    "create_app",
    "evaluate",
    "load_heldout_seeds",
    "oidc_verifier_from_env",
    "open_registry",
    "policy_engine_from_env",
    "pull_verified_layer",
    "rank",
    "reference_policy_loader",
    "resample_from_bundle",
    "resolve_embargo_root",
    "resolve_policy",
    "resolve_submission",
    "submission_policy_ref",
    "validate_policy_ref",
    "validate_submission_manifest",
    "verify_submission_attestations",
]


def create_app(
    store: LeaderboardStore | None = None,
    *,
    service: LeaderboardService | None = None,
    catalog: ScenarioCatalog | None = None,
) -> FastAPI:
    """Construct the leaderboard FastAPI app (requires the ``[leaderboard]`` extra).

    Thin lazy wrapper so the base package imports without FastAPI; see
    :func:`astro_mine.bench.leaderboard._app.create_app`.
    """
    # astro-mine-platform does not ship the REST route module (`_app`); the
    # leaderboard *library* (service, store, auth, eval, ...) is fully present.
    raise ImportError(
        "the leaderboard REST surface (astro_mine.bench.leaderboard._app) is "
        "not included in astro-mine-platform; use the astro-mine-bench "
        "distribution to serve the leaderboard API"
    )
