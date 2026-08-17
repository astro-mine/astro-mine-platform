# SPDX-License-Identifier: Apache-2.0
"""The optional distributed field service (RM-P1-PROSPECT-11; prospect.md §3, §7, §9).

A thin gRPC service fronting the importable library: it serves large shared belief fields and
**streams belief updates** so a distributed swarm sim shares **one consistent, replayable
posterior** (prospect.md §3, §5, §7). The server (:class:`FieldServicer` / :func:`serve`) is the
single writer per field; the client (:class:`FieldServiceClient`) reconstructs the posterior from
the shipped prior + the streamed observation log and verifies its content hash against the
server's — fail-closed.

**It is an authenticated service.** Every RPC is authenticated and authorized before the servicer is
reached (:mod:`~astro_mine.prospect.service._auth`): TLS (optionally mTLS) transport, an OIDC-issued
bearer token, RBAC scopes, and — on the ground-truth-adjacent write path — the Core
``GROUND_TRUTH_ACCESS`` capability grant (prospect.md §9; ``LUNAR-SR-001``, ``LUNAR-SR-005``).
:func:`serve` **requires** an ``auth`` posture: a :class:`ServiceAuth` for any real deployment, or
the explicitly-opt-in, loopback-only :class:`InsecureDevAuth` for local development. There is no
unauthenticated default.

``grpcio`` is an optional dependency (the ``service`` extra:
``pip install astro-mine-platform[prospect-service]``); importing this subpackage requires it.
The core
library (fields, belief, infogain, priors, publish) never imports it.
"""

from __future__ import annotations

from astro_mine.prospect.service._auth import (
    DEFAULT_POLICY,
    INSECURE_DEV_ENV_VAR,
    READ_SCOPE,
    WRITE_SCOPE,
    AuthenticationError,
    AuthInterceptor,
    AuthorizationError,
    InsecureDevAuth,
    JwtVerifier,
    MethodPolicy,
    Principal,
    ServerTls,
    ServiceAuth,
    TokenVerifier,
    bearer_metadata,
    insecure_dev_enabled,
    secure_channel,
)
from astro_mine.prospect.service.client import (
    FieldServiceClient,
    FieldState,
    PosteriorConsistencyError,
)
from astro_mine.prospect.service.server import FieldServicer, serve

__all__ = [
    "DEFAULT_POLICY",
    "INSECURE_DEV_ENV_VAR",
    "READ_SCOPE",
    "WRITE_SCOPE",
    "AuthInterceptor",
    "AuthenticationError",
    "AuthorizationError",
    "FieldServiceClient",
    "FieldServicer",
    "FieldState",
    "InsecureDevAuth",
    "JwtVerifier",
    "MethodPolicy",
    "PosteriorConsistencyError",
    "Principal",
    "ServerTls",
    "ServiceAuth",
    "TokenVerifier",
    "bearer_metadata",
    "insecure_dev_enabled",
    "secure_channel",
    "serve",
]
