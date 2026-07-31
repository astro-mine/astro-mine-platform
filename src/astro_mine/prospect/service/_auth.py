"""AuthN/AuthZ for the distributed field service — TLS + OIDC tokens + capability-gated RBAC.

prospect.md §9 requires **OIDC authentication and RBAC authorization** on the field service, and
conventions.md §9 requires **mTLS** between services and forbids secrets in images or repos. This
module is that enforcement point, and it is **fail-closed at every branch**: no token, an
unverifiable token, an expired token, a token for another audience or issuer, an unknown RPC, an
unknown capability string, or a caller lacking the required grant — every one of them is a
rejection, never a pass-through. There is no configuration in which the service authenticates
*nobody* and serves *everybody*, except the one that must be asked for by name
(:class:`InsecureDevAuth`, below).

**Why the field service in particular.** It is the network path to the belief posterior that the
sealed ground truth feeds. Only privileged, ``GROUND_TRUTH_ACCESS``-holding code (Sim's sensor
model) may draw observations from the sealed field, so only such a caller has any business
*submitting* them into the one shared posterior. An unauthenticated writer could poison the swarm's
belief with fabricated readings — the same class of defect as leaking the truth outright
(prospect.md §9; ``LUNAR-SR-001``, ``LUNAR-SR-005``). :data:`DEFAULT_POLICY` therefore gates
``SubmitObservations`` — the ground-truth-adjacent RPC — on the Core ``GROUND_TRUTH_ACCESS`` grant,
over and above the write scope, while the read RPCs need only a read scope.

**What a caller presents.** A bearer token in the ``authorization`` gRPC metadata header, minted by
the deployment's OIDC provider (conventions.md §9: Keycloak self-host or a cloud IdP). It is
verified **offline** against the IdP's JWKS — signature, ``iss``, ``aud``, ``exp``/``nbf`` — so the
service takes no network dependency on the token path, and an IdP outage cannot fail a request
*open*. The ``scope``/``scp`` claim carries the RBAC scopes and the ``capabilities`` claim the Core
capability grants; the capability vocabulary is Core's, and a claim outside it is a **rejection**,
not a claim quietly ignored.

**No secrets live here.** This module holds no keys, no default token, and no default credential. A
deployment supplies the JWKS and the TLS material (External Secrets / Vault / KMS, conventions.md
§9); the tests mint throwaway ones at runtime.

**Local dev.** ``pip install``-and-run must still work, so :class:`InsecureDevAuth` exists — but it
is opt-in three times over (an explicit argument, an environment variable, and a loopback bind),
never a default, and it is what :func:`~astro_mine.prospect.service.server.serve` refuses to assume.

Backlog: prospect.md §9; conventions.md §9; LUNAR-SR-001, LUNAR-SR-005 —
https://github.com/astro-mine/astro-mine-prospect/issues/32
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import grpc
import jwt
from jwt import PyJWKClient

from astro_mine.core.sadf.enums import CapabilityTag

__all__ = [
    "DEFAULT_POLICY",
    "INSECURE_DEV_ENV_VAR",
    "READ_SCOPE",
    "WRITE_SCOPE",
    "AuthInterceptor",
    "AuthenticationError",
    "AuthorizationError",
    "InsecureDevAuth",
    "JwtVerifier",
    "MethodPolicy",
    "Principal",
    "ServerTls",
    "ServiceAuth",
    "TokenVerifier",
    "bearer_metadata",
    "insecure_dev_enabled",
    "is_loopback",
    "secure_channel",
]

#: The RBAC scopes the built-in :data:`DEFAULT_POLICY` recognizes. A deployment's OPA/IdP maps its
#: own roles onto these (conventions.md §9); they are the vocabulary, not the policy engine.
READ_SCOPE = "prospect.field.read"
WRITE_SCOPE = "prospect.field.write"

#: The environment variable that must be set (truthy) before the insecure local-dev mode is
#: *available at all*. Even then the caller must still pass :class:`InsecureDevAuth` explicitly and
#: bind loopback — three independent opt-ins, so no deployment falls into cleartext by accident.
INSECURE_DEV_ENV_VAR = "ASTRO_MINE_PROSPECT_INSECURE_DEV"

_BEARER_PREFIX = "bearer "
_AUTHORIZATION = "authorization"
_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "[::1]"})


class AuthenticationError(Exception):
    """The caller could not be authenticated — no token, or a token that does not verify.

    Maps to gRPC ``UNAUTHENTICATED``. Deliberately carries no detail about *why* a token failed
    beyond a coarse reason, so the service is not an oracle for forging one.
    """


class AuthorizationError(Exception):
    """The caller is authenticated but lacks the scope / capability the RPC requires.

    Maps to gRPC ``PERMISSION_DENIED``. Raised — never warned — because the ground-truth-adjacent
    RPCs are exactly where a silent pass-through would be a security-class defect (prospect.md §9).
    """


@dataclass(frozen=True)
class Principal:
    """The verified identity of a caller: who they are, and what they are entitled to do.

    :attr:`subject` is the token's ``sub``; :attr:`scopes` the RBAC scopes it carries; and
    :attr:`capabilities` the **Core** capability grants it carries — the same
    :class:`~astro_mine.core.sadf.enums.CapabilityTag` vocabulary that gates the sealed field
    in-process (:mod:`astro_mine.prospect.isolation`), so the network gate and the in-process gate
    are provably the *same* gate, expressed twice.
    """

    subject: str
    scopes: frozenset[str] = frozenset()
    capabilities: frozenset[CapabilityTag] = frozenset()


@dataclass(frozen=True)
class MethodPolicy:
    """What one RPC requires of a caller: RBAC scopes and Core capability grants (**all** of each).

    Both sets are conjunctive — a caller must hold *every* listed scope and *every* listed
    capability. An empty set is not an escape hatch: a method with no policy entry at all is denied
    outright (:meth:`AuthInterceptor._authorize`), so adding an RPC and forgetting its policy fails
    closed rather than exposing it.
    """

    scopes: frozenset[str] = frozenset()
    capabilities: frozenset[CapabilityTag] = frozenset()

    def check(self, principal: Principal, method: str) -> None:
        """Raise :class:`AuthorizationError` unless ``principal`` satisfies this policy."""
        missing_scopes = self.scopes - principal.scopes
        if missing_scopes:
            raise AuthorizationError(
                f"caller {principal.subject!r} may not call {method}: missing scope(s) "
                f"{', '.join(sorted(missing_scopes))}"
            )
        missing_caps = self.capabilities - principal.capabilities
        if missing_caps:
            raise AuthorizationError(
                f"caller {principal.subject!r} may not call {method}: missing capability grant(s) "
                f"{', '.join(sorted(c.value for c in missing_caps))} — this RPC is "
                "ground-truth-adjacent and is gated on the Core capability (prospect.md §9)"
            )


_SERVICE = "/astro_mine.prospect.service.v1.FieldService/"

#: The service's RBAC policy — the authoritative mapping from RPC to requirement.
#:
#: ``SubmitObservations`` is the **ground-truth-adjacent** RPC. The observations it appends to the
#: one shared posterior are, in a simulation, drawn from the sealed ``GroundTruthField``, which only
#: a ``GROUND_TRUTH_ACCESS``-holding caller (Sim's sensor model) may read — so the write path is
#: gated on that Core capability grant *in addition to* the write scope: a caller that could not
#: have obtained the truth may not claim to report it. The read RPCs serve only the public belief
#: prior and the agent-facing posterior, and need a read scope alone.
DEFAULT_POLICY: Mapping[str, MethodPolicy] = {
    f"{_SERVICE}GetField": MethodPolicy(scopes=frozenset({READ_SCOPE})),
    f"{_SERVICE}StreamBeliefUpdates": MethodPolicy(scopes=frozenset({READ_SCOPE})),
    f"{_SERVICE}SubmitObservations": MethodPolicy(
        scopes=frozenset({WRITE_SCOPE}),
        capabilities=frozenset({CapabilityTag.GROUND_TRUTH_ACCESS}),
    ),
}


class TokenVerifier(Protocol):
    """Verifies a bearer token and returns the :class:`Principal` it attests, or raises.

    The seam a deployment plugs its own IdP into (or an OPA sidecar's decision, if authorization is
    externalized). It MUST raise :class:`AuthenticationError` on any token it cannot fully verify:
    returning an empty/anonymous principal instead would be a fail-open, which this service does not
    permit.
    """

    def verify(self, token: str) -> Principal: ...


@dataclass(frozen=True)
class JwtVerifier:
    """An **offline** OIDC-compatible JWT verifier (signature, ``iss``, ``aud``, ``exp``/``nbf``).

    Construct it from the IdP's JWKS: :meth:`from_jwks` for a key set already in hand, or
    :meth:`from_jwks_uri` to let PyJWT fetch and cache it from the provider's discovery document at
    startup. Verification itself never touches the network, so the request path has no IdP
    dependency and an IdP outage cannot fail a request open.

    Scopes are read from the standard ``scope`` (space-delimited) or ``scp`` (list) claim, and Core
    capability grants from :attr:`capabilities_claim`. An unrecognized capability string is a
    **rejection**: a token asserting a grant this platform does not know is malformed, and treating
    it as "no grant" would let a typo'd or forged claim slide through unremarked.
    """

    issuer: str
    audience: str
    key_resolver: Callable[[str], Any]
    algorithms: Sequence[str] = ("RS256", "ES256")
    capabilities_claim: str = "capabilities"
    leeway_s: float = 0.0

    @classmethod
    def from_jwks(
        cls, jwks: Mapping[str, Any], *, issuer: str, audience: str, **kwargs: Any
    ) -> JwtVerifier:
        """Build a verifier over an in-memory ``kid -> key`` map (the IdP's published JWKS)."""

        def resolve(token: str) -> Any:
            kid = jwt.get_unverified_header(token).get("kid")
            key = jwks.get(str(kid))
            if key is None:
                raise AuthenticationError("bearer token is signed by an unknown key")
            return key

        return cls(issuer=issuer, audience=audience, key_resolver=resolve, **kwargs)

    def _resolve_key(self, token: str) -> Any:
        """Resolve the token's signing key, mapping *any* PyJWT failure onto an auth rejection.

        Key resolution parses the token header, so it can fail on a malformed token before a
        signature is ever checked. Letting that escape as a raw ``PyJWTError`` would surface as an
        ``UNKNOWN`` gRPC status instead of ``UNAUTHENTICATED`` — a caller (or a retry policy) could
        reasonably read that as a transient server fault rather than a refusal. It is a rejection,
        and it says so.
        """
        try:
            return self.key_resolver(token)
        except AuthenticationError:
            raise
        except jwt.PyJWTError as exc:
            raise AuthenticationError(f"bearer token is not valid: {type(exc).__name__}") from exc

    @classmethod
    def from_jwks_uri(
        cls, jwks_uri: str, *, issuer: str, audience: str, **kwargs: Any
    ) -> JwtVerifier:  # pragma: no cover - needs a live IdP; exercised in deployment, not in CI
        """Build a verifier that resolves signing keys from the IdP's JWKS endpoint (cached)."""
        client = PyJWKClient(jwks_uri)
        return cls(
            issuer=issuer,
            audience=audience,
            key_resolver=lambda token: client.get_signing_key_from_jwt(token).key,
            **kwargs,
        )

    def verify(self, token: str) -> Principal:
        """Verify ``token`` and return the :class:`Principal` it attests (raises on any defect)."""
        key = self._resolve_key(token)
        try:
            claims = jwt.decode(
                token,
                key=key,
                algorithms=list(self.algorithms),
                issuer=self.issuer,
                audience=self.audience,
                leeway=self.leeway_s,
                options={"require": ["exp", "iss", "aud", "sub"]},
            )
        except jwt.InvalidTokenError as exc:
            # A coarse message on purpose: do not tell an attacker which check failed.
            raise AuthenticationError(f"bearer token is not valid: {type(exc).__name__}") from exc
        return Principal(
            subject=str(claims["sub"]),
            scopes=frozenset(_scopes(claims)),
            capabilities=frozenset(_capabilities(claims.get(self.capabilities_claim, ()))),
        )


def _scopes(claims: Mapping[str, Any]) -> Iterable[str]:
    """The RBAC scopes of a token — the OAuth2 ``scope`` string, or the ``scp`` list."""
    raw = claims.get("scope")
    if isinstance(raw, str):
        return raw.split()
    return [str(s) for s in claims.get("scp", ())]


def _capabilities(raw: Any) -> Iterable[CapabilityTag]:
    """Map a token's capability claim onto Core's closed vocabulary (unknown value => rejection)."""
    values = raw.split() if isinstance(raw, str) else raw
    out: list[CapabilityTag] = []
    for value in values:
        try:
            out.append(CapabilityTag(value))
        except ValueError:
            raise AuthenticationError(
                f"token asserts an unrecognized capability {value!r}; the capability vocabulary is "
                "Core's, and a claim outside it is malformed"
            ) from None
    return out


@dataclass(frozen=True)
class ServerTls:
    """The service's TLS material: its certificate chain + key, and — for mTLS — a client CA.

    ``require_client_auth=True`` with a ``client_ca`` is the **mTLS** posture conventions.md §9
    calls for on service-to-service traffic: the server then rejects any peer that cannot present a
    certificate signed by that CA, *before* a token is even read. Material is passed as bytes
    (loaded from files by :meth:`from_files`; in a deployment, from External Secrets / Vault / KMS)
    — nothing is read from a default location, and no key material lives in this repo.
    """

    certificate_chain: bytes
    private_key: bytes
    client_ca: bytes | None = None
    require_client_auth: bool = False

    @classmethod
    def from_files(
        cls,
        certificate_chain: str | Path,
        private_key: str | Path,
        *,
        client_ca: str | Path | None = None,
        require_client_auth: bool = False,
    ) -> ServerTls:
        """Load PEM material from disk (paths supplied by the deployment's secret mount)."""
        return cls(
            certificate_chain=Path(certificate_chain).read_bytes(),
            private_key=Path(private_key).read_bytes(),
            client_ca=None if client_ca is None else Path(client_ca).read_bytes(),
            require_client_auth=require_client_auth,
        )

    def credentials(self) -> grpc.ServerCredentials:
        """The gRPC server credentials this TLS material implies."""
        if self.require_client_auth and self.client_ca is None:
            raise ValueError("require_client_auth=True needs a client_ca to verify peers against")
        return grpc.ssl_server_credentials(
            [(self.private_key, self.certificate_chain)],
            root_certificates=self.client_ca,
            require_client_auth=self.require_client_auth,
        )


@dataclass(frozen=True)
class ServiceAuth:
    """The **secure** posture: TLS transport + token authentication + per-method authorization.

    This is what :func:`~astro_mine.prospect.service.server.serve` expects. Both halves are
    mandatory: there is no "TLS but no tokens" and no "tokens but no TLS" configuration, because
    either alone leaves the ground-truth-adjacent write path exposed — a bearer token on a cleartext
    channel is a bearer token for whoever is listening.
    """

    tls: ServerTls
    verifier: TokenVerifier
    policy: Mapping[str, MethodPolicy] = field(default_factory=lambda: DEFAULT_POLICY)


@dataclass(frozen=True)
class InsecureDevAuth:
    """The **local-dev only** posture: no TLS, no tokens — every RPC served to whoever connects.

    It exists so that ``pip install astro-mine-platform[prospect-service]`` and a local
    round-trip still work
    without standing up an IdP (prospect.md §7's local tier; ``LUNAR-TR-004``). It is never the
    default and cannot be reached by omission: :func:`~astro_mine.prospect.service.server.serve`
    accepts it only when the caller passes it explicitly, the :data:`INSECURE_DEV_ENV_VAR`
    environment variable is set, **and** the bind address is loopback. Anything else raises.

    :attr:`principal` is the identity every caller is *granted* in this mode. It defaults to the
    full grant, so that local dev exercises the same authorization code path a deployment does;
    narrow it to prove that a caller lacking a grant is refused.
    """

    principal: Principal = Principal(
        subject="insecure-dev",
        scopes=frozenset({READ_SCOPE, WRITE_SCOPE}),
        capabilities=frozenset({CapabilityTag.GROUND_TRUTH_ACCESS}),
    )
    policy: Mapping[str, MethodPolicy] = field(default_factory=lambda: DEFAULT_POLICY)


def insecure_dev_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """Whether the environment permits the insecure local-dev mode at all (opt-in 2 of 3)."""
    env = os.environ if environ is None else environ
    return env.get(INSECURE_DEV_ENV_VAR, "").strip().lower() in {"1", "true", "yes", "on"}


def is_loopback(address: str) -> bool:
    """Whether ``address`` binds loopback only — the insecure mode's third and final constraint."""
    host = address.rsplit(":", 1)[0] if ":" in address else address
    return host in _LOOPBACK_HOSTS


class AuthInterceptor(grpc.ServerInterceptor):  # type: ignore[misc]
    """The gRPC server interceptor that authenticates and authorizes **every** RPC.

    Fail-closed by construction, in three independent ways:

    1. an RPC with **no policy entry** is denied, so adding a method and forgetting its policy
       cannot silently expose it;
    2. authentication runs before the servicer is reached, so no handler can be invoked by an
       unauthenticated caller, however that handler was registered;
    3. the original handler's *cardinality* is preserved when wrapping (unary and server-streaming
       alike), so a denial reaches the client as a proper gRPC status rather than as a protocol
       error it might mistake for a transient failure and retry into.
    """

    def __init__(
        self,
        *,
        policy: Mapping[str, MethodPolicy],
        verifier: TokenVerifier | None = None,
        static_principal: Principal | None = None,
    ) -> None:
        if verifier is None and static_principal is None:
            raise ValueError(
                "AuthInterceptor needs a token verifier (or, for local dev only, a static "
                "principal); it will not serve an RPC it cannot attribute to a caller"
            )
        self._policy = policy
        self._verifier = verifier
        self._static_principal = static_principal

    def intercept_service(
        self,
        continuation: Callable[[Any], Any],
        handler_call_details: Any,
    ) -> Any:
        handler = continuation(handler_call_details)
        if handler is None:  # an unknown method: gRPC's own UNIMPLEMENTED path
            return None
        method = str(handler_call_details.method)

        def guard(context: grpc.ServicerContext) -> None:
            """Authenticate + authorize, aborting the RPC with a gRPC status on any failure."""
            try:
                principal = self._authenticate(context.invocation_metadata())
                self._authorize(principal, method)
            except AuthenticationError as exc:
                context.abort(grpc.StatusCode.UNAUTHENTICATED, str(exc))
            except AuthorizationError as exc:
                context.abort(grpc.StatusCode.PERMISSION_DENIED, str(exc))

        if handler.unary_unary is not None:
            inner_unary = handler.unary_unary

            def unary(request: Any, context: grpc.ServicerContext) -> Any:
                guard(context)
                return inner_unary(request, context)

            return grpc.unary_unary_rpc_method_handler(
                unary,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )

        if handler.unary_stream is not None:
            inner_stream = handler.unary_stream

            def stream(request: Any, context: grpc.ServicerContext) -> Any:
                guard(context)
                yield from inner_stream(request, context)

            return grpc.unary_stream_rpc_method_handler(
                stream,
                request_deserializer=handler.request_deserializer,
                response_serializer=handler.response_serializer,
            )

        # The field service serves only unary and server-streaming RPCs. A client-streaming or
        # bidirectional handler would slip past both wrappers above, so it is refused outright
        # rather than served unguarded.
        raise AuthorizationError(
            f"{method} has a cardinality this interceptor cannot guard; refusing to serve it "
            "unguarded (fail-closed)"
        )

    def _authenticate(self, metadata: Sequence[tuple[str, str]]) -> Principal:
        if self._static_principal is not None:  # the insecure local-dev mode
            return self._static_principal
        assert self._verifier is not None  # guaranteed by __init__
        return self._verifier.verify(_bearer_token(metadata))

    def _authorize(self, principal: Principal, method: str) -> None:
        policy = self._policy.get(method)
        if policy is None:
            raise AuthorizationError(
                f"{method} has no authorization policy; refusing it (fail-closed — an RPC is "
                "denied until its policy is declared)"
            )
        policy.check(principal, method)


def _bearer_token(metadata: Sequence[tuple[str, str]]) -> str:
    """The bearer token in gRPC ``authorization`` metadata (raises if absent or malformed)."""
    for key, value in metadata:
        if key.lower() != _AUTHORIZATION:
            continue
        if value.lower().startswith(_BEARER_PREFIX):
            token = value[len(_BEARER_PREFIX) :].strip()
            if token:
                return token
        raise AuthenticationError("authorization metadata is not a bearer token")
    raise AuthenticationError("request carries no authorization metadata (fail-closed)")


def bearer_metadata(token: str) -> tuple[tuple[str, str], ...]:
    """The gRPC call metadata carrying ``token`` as a bearer credential.

    Send it only over a TLS channel (:func:`secure_channel`): a bearer token on a cleartext channel
    is a bearer token for anyone on the path (conventions.md §9).
    """
    return ((_AUTHORIZATION, f"Bearer {token}"),)


def secure_channel(
    address: str,
    *,
    root_certificates: bytes | None = None,
    private_key: bytes | None = None,
    certificate_chain: bytes | None = None,
    options: Sequence[tuple[str, Any]] | None = None,
) -> grpc.Channel:
    """A TLS — or, with a client key and chain, **mTLS** — channel to the field service.

    ``root_certificates`` is the CA the server's certificate is verified against (``None`` uses the
    host trust store); ``private_key`` + ``certificate_chain`` present a *client* certificate, which
    is what a server configured with :attr:`ServerTls.require_client_auth` demands. The bearer token
    rides as call metadata (:func:`bearer_metadata`) on top of this channel — never beside it.
    """
    credentials = grpc.ssl_channel_credentials(
        root_certificates=root_certificates,
        private_key=private_key,
        certificate_chain=certificate_chain,
    )
    return grpc.secure_channel(address, credentials, options=list(options or ()))
