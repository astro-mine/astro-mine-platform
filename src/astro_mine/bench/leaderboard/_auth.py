"""OIDC bearer-token authentication for the hosted leaderboard (bench#29; bench.md §9).

bench.md §9 and conventions.md §9 both open with the same requirement: **AuthN is OIDC** (Keycloak
or a cloud IdP). Until bench#29 the hosted leaderboard had *no authentication of any kind* — anyone
who could reach the port could submit, and the only thing standing between the public internet and
the evaluator was an in-memory rate limiter keyed on a **client-supplied** ``identity`` string that
a submitter could simply change to reset their own quota.

This module is the authentication half of the fix:

- :class:`Principal` — the authenticated caller: a subject, the issuer that vouched for it, and the
  roles/scopes the authorization layer (:mod:`._authz`) decides on. Every quota, embargo, and
  authoring decision is keyed on ``principal.subject`` — a value the IdP asserts and the submitter
  cannot forge — never on a field of the request body.
- :class:`OidcTokenVerifier` — a fail-closed JWT verifier: it checks the **signature** against the
  issuer's JWKS, and the ``iss``, ``aud``, ``exp`` and ``nbf`` claims. A token that is unsigned,
  signed by an unknown key, expired, or issued for another audience is rejected; ``alg: none`` and
  symmetric algorithms are impossible because only asymmetric algorithms are accepted.

**No secrets live here.** The verifier holds public keys only, and is configured from the
environment by *reference* (issuer URL, audience, JWKS URL) — never by embedding key material or a
shared secret in the image or the repo (conventions.md §9: *"No secrets in images or repos"*).

The **local, offline tier is untouched**: ``astro-mine bench score``, ``run(spec, policy)``, and the
leaderboard's read paths need no account and no token (CX-LOCAL; bench#29 AC5). Authentication gates
the hosted *write* surface only.

Requires the ``[leaderboard]`` extra (``pyjwt[crypto]``); imported lazily by the app.

Backlog: bench#29 — https://github.com/astro-mine/astro-mine-bench/issues/29
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AUDIENCE_ENV",
    "ISSUER_ENV",
    "JWKS_URL_ENV",
    "ROLES_CLAIM_ENV",
    "AuthenticationError",
    "OidcTokenVerifier",
    "Principal",
    "TokenVerifier",
    "bearer_token",
    "oidc_verifier_from_env",
]

#: Env vars configuring the hosted deployment's IdP. All are *references*, never key material:
#: the issuer and audience the token must carry, and where to fetch the issuer's **public** JWKS.
ISSUER_ENV = "ASTRO_MINE_BENCH_OIDC_ISSUER"
AUDIENCE_ENV = "ASTRO_MINE_BENCH_OIDC_AUDIENCE"
JWKS_URL_ENV = "ASTRO_MINE_BENCH_OIDC_JWKS_URL"
ROLES_CLAIM_ENV = "ASTRO_MINE_BENCH_OIDC_ROLES_CLAIM"

#: Only asymmetric signatures are accepted. A shared secret (``HS*``) would let anyone holding the
#: verification key *mint* tokens, and ``none`` is the classic JWT forgery — neither is admissible.
DEFAULT_ALGORITHMS: tuple[str, ...] = ("RS256", "RS384", "RS512", "ES256", "ES384", "PS256")

#: Clock-skew tolerance on ``exp``/``nbf``, in seconds.
DEFAULT_LEEWAY_SECONDS = 30


class AuthenticationError(Exception):
    """The caller could not be authenticated — the request is rejected with 401 (fail-closed)."""


class Principal(BaseModel):
    """An authenticated caller: who the IdP says they are, and what they are entitled to be.

    ``subject`` is the IdP's stable subject claim (``sub``) — the identity every quota and audit
    record is keyed on. ``roles``/``scopes`` feed the policy engine (:mod:`._authz`); they come from
    the token, so a submitter cannot grant themselves ``admin`` by editing a request body.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    subject: str = Field(min_length=1)
    issuer: str = Field(min_length=1)
    roles: tuple[str, ...] = ()
    scopes: tuple[str, ...] = ()
    email: str | None = None

    @property
    def identity(self) -> str:
        """The rate-limit / quota key: the issuer-qualified subject, unforgeable by the client."""
        return f"{self.issuer}#{self.subject}"


class TokenVerifier(Protocol):
    """Verifies a bearer token and returns the :class:`Principal` it authenticates.

    Implementations MUST fail closed: any doubt about a token's signature, issuer, audience, or
    validity window raises :class:`AuthenticationError` rather than returning a partial principal.
    """

    def verify(self, token: str) -> Principal:
        """Authenticate ``token``; raise :class:`AuthenticationError` if it cannot be trusted."""
        ...


def bearer_token(header: str | None) -> str:
    """Extract the token from an ``Authorization: Bearer <token>`` header, fail-closed.

    A missing, malformed, or non-``Bearer`` header is an authentication failure — never an
    anonymous pass-through.
    """
    if not header:
        raise AuthenticationError("missing Authorization header; a bearer token is required")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthenticationError("Authorization header must be 'Bearer <token>'")
    return token.strip()


class OidcTokenVerifier:
    """Verify an OIDC bearer JWT against the issuer's JWKS — signature, issuer, audience, expiry.

    ``jwks`` supplies the issuer's public key set directly (a deployment that mounts it, and the
    tests). ``jwks_url`` fetches it instead, through an injected HTTP client; the fetched set is
    cached, and a ``kid`` that misses the cache triggers exactly one refetch (so an IdP key rotation
    is picked up without a restart, and an unknown ``kid`` cannot become an unbounded fetch loop).

    Every check is delegated to PyJWT with verification **on**: an unsigned token, a token signed
    with an unknown key, an expired token, and a token minted for another audience are all rejected.
    """

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        jwks: Mapping[str, Any] | None = None,
        jwks_url: str | None = None,
        http: Any = None,
        algorithms: Sequence[str] = DEFAULT_ALGORITHMS,
        roles_claim: str = "roles",
        leeway: int = DEFAULT_LEEWAY_SECONDS,
    ) -> None:
        if not issuer or not audience:
            raise ValueError("OidcTokenVerifier needs both an issuer and an audience")
        if jwks is None and jwks_url is None:
            raise ValueError("OidcTokenVerifier needs a jwks mapping or a jwks_url to fetch one")
        self._issuer = issuer
        self._audience = audience
        self._jwks_url = jwks_url
        self._http = http
        self._algorithms = tuple(algorithms)
        self._roles_claim = roles_claim
        self._leeway = leeway
        self._jwks: dict[str, Any] | None = dict(jwks) if jwks is not None else None

    @property
    def issuer(self) -> str:
        """The IdP whose tokens this verifier accepts."""
        return self._issuer

    def _fetch_jwks(self) -> dict[str, Any]:
        """Fetch the issuer's public key set over HTTP (never any private material)."""
        if self._jwks_url is None:
            raise AuthenticationError("no JWKS is configured for this issuer")
        import httpx

        client = self._http if self._http is not None else httpx.Client(timeout=5.0)
        try:
            response = client.get(self._jwks_url)
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
        except Exception as exc:
            raise AuthenticationError(f"could not fetch the issuer JWKS: {exc}") from exc
        finally:
            if self._http is None:
                client.close()
        self._jwks = payload
        return payload

    @staticmethod
    def _match_key(jwks: Mapping[str, Any], kid: str | None) -> Any:
        """The key in ``jwks`` bearing ``kid``, or ``None`` if the set carries no such key.

        An **empty** key set is "no match", not an error: it is what an IdP briefly serves mid
        rotation, and it must fall through to the single refetch rather than short-circuiting.
        """
        import jwt

        if not jwks.get("keys"):
            return None
        try:
            key_set = jwt.PyJWKSet.from_dict(dict(jwks))
        except jwt.PyJWTError as exc:
            raise AuthenticationError(f"malformed issuer JWKS: {exc}") from exc
        for key in key_set.keys:
            if kid is None or key.key_id == kid:
                return key.key
        return None

    def _signing_key(self, token: str) -> Any:
        """Resolve the token's ``kid`` to a public key from the JWKS, refetching once on a miss."""
        import jwt

        try:
            kid = jwt.get_unverified_header(token).get("kid")
        except jwt.PyJWTError as exc:
            raise AuthenticationError(f"malformed bearer token: {exc}") from exc

        for attempt in (0, 1):
            jwks = self._jwks if self._jwks is not None else self._fetch_jwks()
            key = self._match_key(jwks, kid)
            if key is not None:
                return key
            # A rotated IdP key: refetch exactly **once**, then give up rather than trusting the
            # token — an unknown kid must not become an unbounded fetch loop an attacker can drive.
            if attempt == 0 and self._jwks_url is not None:
                self._jwks = None
            else:
                break
        raise AuthenticationError(f"no JWKS key matches the token's kid {kid!r}")

    def verify(self, token: str) -> Principal:
        """Authenticate ``token`` and build its :class:`Principal`; raise on any doubt."""
        import jwt

        key = self._signing_key(token)
        try:
            claims: dict[str, Any] = jwt.decode(
                token,
                key=key,
                algorithms=list(self._algorithms),
                audience=self._audience,
                issuer=self._issuer,
                leeway=self._leeway,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_nbf": True,
                    "verify_aud": True,
                    "verify_iss": True,
                    "require": ["exp", "iss", "sub"],
                },
            )
        except jwt.PyJWTError as exc:
            raise AuthenticationError(f"bearer token rejected: {exc}") from exc

        subject = str(claims.get("sub") or "")
        if not subject:
            raise AuthenticationError("bearer token carries no subject claim")
        return Principal(
            subject=subject,
            issuer=str(claims.get("iss") or self._issuer),
            roles=_claim_tuple(claims.get(self._roles_claim)),
            scopes=_scope_tuple(claims.get("scope") or claims.get("scp")),
            email=(str(claims["email"]) if claims.get("email") else None),
        )


def _claim_tuple(value: object) -> tuple[str, ...]:
    """Normalize a roles claim, which IdPs render as a list or a space-delimited string."""
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(part for part in value.split() if part)
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    return ()


def _scope_tuple(value: object) -> tuple[str, ...]:
    """Normalize the OAuth ``scope``/``scp`` claim to a tuple."""
    return _claim_tuple(value)


def oidc_verifier_from_env(
    env: Mapping[str, str] | None = None, *, http: Any = None
) -> OidcTokenVerifier | None:
    """Build the deployment's verifier from the environment, or ``None`` if OIDC is unconfigured.

    Returning ``None`` does **not** mean "allow anonymous writes": the app treats an unconfigured
    verifier as *this deployment cannot authenticate anyone* and refuses every write route with
    503, while the account-free read/score paths keep working (bench#29 AC5). Enabling
    authentication is a deployment act; disabling it is not a supported one.
    """
    environment = os.environ if env is None else env
    issuer = environment.get(ISSUER_ENV)
    audience = environment.get(AUDIENCE_ENV)
    if not issuer or not audience:
        return None
    default_jwks = f"{issuer.rstrip('/')}/protocol/openid-connect/certs"
    jwks_url = environment.get(JWKS_URL_ENV) or default_jwks
    return OidcTokenVerifier(
        issuer=issuer,
        audience=audience,
        jwks_url=jwks_url,
        http=http,
        roles_claim=environment.get(ROLES_CLAIM_ENV, "roles"),
    )
