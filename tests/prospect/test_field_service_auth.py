"""AuthN/AuthZ on the distributed field service (prospect.md §9; LUNAR-SR-001, LUNAR-SR-005).

Proves every acceptance criterion of the AuthN/Z gap, against a **real** server: a throwaway CA
mints the TLS material and a throwaway ES256 keypair stands in for the deployment's OIDC provider,
both generated at runtime — no key material is committed anywhere (conventions.md §9).

- **No unauthenticated default** — ``serve`` cannot be called without an ``auth`` posture at all,
  and the cleartext local-dev path is refused unless it is asked for explicitly *and* by environment
  variable *and* on loopback.
- **TLS + token auth** — a client with a valid bearer token over TLS round-trips the posterior; the
  transport is genuinely TLS (a cleartext client cannot talk to it), and mTLS is honoured when the
  server demands a client certificate.
- **Fail-closed** — a missing, malformed, unsigned, wrong-issuer, wrong-audience, or expired token
  is ``UNAUTHENTICATED``. Never a silent pass-through.
- **Ground-truth-adjacent RPCs are capability-gated** — a caller holding the write *scope* but not
  the Core ``GROUND_TRUTH_ACCESS`` grant is refused ``SubmitObservations`` with
  ``PERMISSION_DENIED``, while the read RPCs still work for it. This is the network-side mirror of
  the in-process seal (:mod:`astro_mine.prospect.isolation`).
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator

import grpc
import jwt
import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import NameOID

from astro_mine.core.sadf.enums import CapabilityTag
from astro_mine.prospect.belief import FieldObservation
from astro_mine.prospect.field import FieldGrid
from astro_mine.prospect.priors import load_prior
from astro_mine.prospect.service import (
    DEFAULT_POLICY,
    INSECURE_DEV_ENV_VAR,
    READ_SCOPE,
    WRITE_SCOPE,
    AuthenticationError,
    AuthInterceptor,
    AuthorizationError,
    FieldServiceClient,
    InsecureDevAuth,
    JwtVerifier,
    MethodPolicy,
    Principal,
    ServerTls,
    ServiceAuth,
    bearer_metadata,
    insecure_dev_enabled,
    secure_channel,
    serve,
)

_FIELD = "shackleton_water_ice"
_ISSUER = "https://idp.test.invalid/realms/astro-mine"
_AUDIENCE = "astro-mine-prospect"
_KID = "test-key-1"
_HOST = "localhost"
_GT = CapabilityTag.GROUND_TRUTH_ACCESS


# --- throwaway PKI + IdP (generated at runtime; nothing is committed) ---------------------------


def _self_signed_ca() -> tuple[ec.EllipticCurvePrivateKey, x509.Certificate]:
    key = ec.generate_private_key(ec.SECP256R1())
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "astro-mine-test-ca")])
    now = dt.datetime.now(dt.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(hours=1))
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )
    return key, cert


def _leaf(
    ca_key: ec.EllipticCurvePrivateKey, ca_cert: x509.Certificate, common_name: str
) -> tuple[bytes, bytes]:
    """A CA-signed leaf certificate + key (PEM), with a ``localhost`` SAN for gRPC's host check."""
    key = ec.generate_private_key(ec.SECP256R1())
    now = dt.datetime.now(dt.UTC)
    cert = (
        x509.CertificateBuilder()
        .subject_name(x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, common_name)]))
        .issuer_name(ca_cert.subject)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - dt.timedelta(minutes=5))
        .not_valid_after(now + dt.timedelta(hours=1))
        .add_extension(x509.SubjectAlternativeName([x509.DNSName(_HOST)]), critical=False)
        .sign(ca_key, hashes.SHA256())
    )
    key_pem = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    return cert.public_bytes(serialization.Encoding.PEM), key_pem


class _Idp:
    """A throwaway ES256 OIDC provider: it mints tokens, and publishes the JWKS to verify them."""

    def __init__(self, kid: str = _KID) -> None:
        self.kid = kid
        self._key = ec.generate_private_key(ec.SECP256R1())
        self.jwks = {kid: self._key.public_key()}

    def token(
        self,
        *,
        scopes: tuple[str, ...] = (READ_SCOPE, WRITE_SCOPE),
        capabilities: tuple[str, ...] = (_GT.value,),
        issuer: str = _ISSUER,
        audience: str = _AUDIENCE,
        expires_in_s: int = 300,
        subject: str = "sim-sensor-model",
    ) -> str:
        now = dt.datetime.now(dt.UTC)
        claims = {
            "sub": subject,
            "iss": issuer,
            "aud": audience,
            "iat": now,
            "exp": now + dt.timedelta(seconds=expires_in_s),
            "scope": " ".join(scopes),
            "capabilities": list(capabilities),
        }
        pem = self._key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        return jwt.encode(claims, pem, algorithm="ES256", headers={"kid": self.kid})


@pytest.fixture(scope="module")
def idp() -> _Idp:
    return _Idp()


@pytest.fixture(scope="module")
def pki() -> tuple[bytes, bytes, bytes, bytes, bytes]:
    """``(ca_pem, server_cert, server_key, client_cert, client_key)`` — all minted for this run."""
    ca_key, ca_cert = _self_signed_ca()
    server_cert, server_key = _leaf(ca_key, ca_cert, "field-service")
    client_cert, client_key = _leaf(ca_key, ca_cert, "sim-sensor-model")
    ca_pem = ca_cert.public_bytes(serialization.Encoding.PEM)
    return ca_pem, server_cert, server_key, client_cert, client_key


def _verifier(idp: _Idp) -> JwtVerifier:
    return JwtVerifier.from_jwks(idp.jwks, issuer=_ISSUER, audience=_AUDIENCE)


def _grid() -> FieldGrid:
    return FieldGrid(
        min_x_m=-1_000.0, min_y_m=-1_000.0, max_x_m=1_000.0, max_y_m=1_000.0, n_rows=8, n_cols=8
    )


def _obs() -> FieldObservation:
    return FieldObservation(x_m=0.0, y_m=0.0, value=0.05, noise_sigma=0.01, sensor="neutron")


def _serve_tls(
    idp: _Idp,
    pki: tuple[bytes, bytes, bytes, bytes, bytes],
    *,
    require_client_auth: bool = False,
) -> Iterator[tuple[str, bytes]]:
    ca_pem, server_cert, server_key, _, _ = pki
    tls = ServerTls(
        certificate_chain=server_cert,
        private_key=server_key,
        client_ca=ca_pem if require_client_auth else None,
        require_client_auth=require_client_auth,
    )
    auth = ServiceAuth(tls=tls, verifier=_verifier(idp))
    with serve({_FIELD: load_prior(grid=_grid())}, auth=auth, address=f"{_HOST}:0") as (
        _server,
        address,
    ):
        yield address, ca_pem


# --- AC1: there is no unauthenticated default ---------------------------------------------------


def test_serve_requires_an_auth_posture() -> None:
    # The signature itself forbids the old `serve(fields)` call: `auth` is keyword-only + required.
    with pytest.raises(TypeError, match="auth"):
        serve({_FIELD: load_prior(grid=_grid())})  # type: ignore[call-arg]


def test_insecure_dev_mode_needs_the_environment_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(INSECURE_DEV_ENV_VAR, raising=False)
    assert insecure_dev_enabled() is False
    with (
        pytest.raises(ValueError, match=INSECURE_DEV_ENV_VAR),
        serve({_FIELD: load_prior(grid=_grid())}, auth=InsecureDevAuth()),
    ):
        pass  # pragma: no cover - serve raises before yielding


def test_insecure_dev_mode_refuses_a_non_loopback_bind(monkeypatch: pytest.MonkeyPatch) -> None:
    # Even fully opted in, cleartext must never be reachable off-host (conventions.md §9).
    monkeypatch.setenv(INSECURE_DEV_ENV_VAR, "1")
    with (
        pytest.raises(ValueError, match="loopback only"),
        serve(
            {_FIELD: load_prior(grid=_grid())},
            auth=InsecureDevAuth(),
            address="0.0.0.0:0",
        ),
    ):
        pass  # pragma: no cover - serve raises before yielding


def test_insecure_dev_mode_serves_once_fully_opted_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(INSECURE_DEV_ENV_VAR, "true")
    with serve({_FIELD: load_prior(grid=_grid())}, auth=InsecureDevAuth()) as (_s, address):
        client = FieldServiceClient(grpc.insecure_channel(address))
        assert client.get_field(_FIELD).revision == 0


# --- AC2: TLS + token auth round-trips ----------------------------------------------------------


def test_tls_and_a_valid_token_round_trip_the_posterior(
    idp: _Idp, pki: tuple[bytes, bytes, bytes, bytes, bytes]
) -> None:
    for address, ca_pem in _serve_tls(idp, pki):
        client = FieldServiceClient(
            secure_channel(address, root_certificates=ca_pem), token=idp.token()
        )
        ack = client.submit_observations(_FIELD, [_obs()])
        state = client.get_field(_FIELD)
        assert ack.revision == 1
        assert state.content_hash == ack.content_hash
        assert state.belief.variance((0.0, 0.0, 0.0)) > 0.0  # still uncertainty-first


def test_the_transport_is_really_tls(
    idp: _Idp, pki: tuple[bytes, bytes, bytes, bytes, bytes]
) -> None:
    # A cleartext client cannot talk to the secure port at all — the token never even reaches it.
    for address, _ca in _serve_tls(idp, pki):
        client = FieldServiceClient(grpc.insecure_channel(address), token=idp.token())
        with pytest.raises(grpc.RpcError):
            client.get_field(_FIELD)


def test_mutual_tls_requires_a_client_certificate(
    idp: _Idp, pki: tuple[bytes, bytes, bytes, bytes, bytes]
) -> None:
    ca_pem, _sc, _sk, client_cert, client_key = pki
    for address, _ca in _serve_tls(idp, pki, require_client_auth=True):
        # With a client certificate signed by the CA the server trusts: served (conventions.md §9).
        ok = FieldServiceClient(
            secure_channel(
                address,
                root_certificates=ca_pem,
                private_key=client_key,
                certificate_chain=client_cert,
            ),
            token=idp.token(),
        )
        assert ok.get_field(_FIELD).revision == 0

        # Without one, the handshake itself fails — before any token is considered.
        anonymous = FieldServiceClient(
            secure_channel(address, root_certificates=ca_pem), token=idp.token()
        )
        with pytest.raises(grpc.RpcError):
            anonymous.get_field(_FIELD)


# --- AC3: a failed/missing token is fail-closed, never a silent pass-through ---------------------


@pytest.mark.parametrize(
    ("name", "token"),
    [
        ("none", None),
        ("garbage", "not-a-jwt"),
        (
            "unsigned",
            jwt.encode({"sub": "x", "iss": _ISSUER, "aud": _AUDIENCE}, key="", algorithm="none"),
        ),
    ],
)
def test_a_missing_or_unverifiable_token_is_unauthenticated(
    idp: _Idp, pki: tuple[bytes, bytes, bytes, bytes, bytes], name: str, token: str | None
) -> None:
    for address, ca_pem in _serve_tls(idp, pki):
        client = FieldServiceClient(secure_channel(address, root_certificates=ca_pem), token=token)
        with pytest.raises(grpc.RpcError) as excinfo:
            client.get_field(_FIELD)
        assert excinfo.value.code() is grpc.StatusCode.UNAUTHENTICATED, name


@pytest.mark.parametrize(
    ("name", "kwargs"),
    [
        ("wrong issuer", {"issuer": "https://evil.test.invalid"}),
        ("wrong audience", {"audience": "some-other-service"}),
        ("expired", {"expires_in_s": -60}),
    ],
)
def test_a_token_failing_any_oidc_check_is_unauthenticated(
    idp: _Idp,
    pki: tuple[bytes, bytes, bytes, bytes, bytes],
    name: str,
    kwargs: dict[str, object],
) -> None:
    for address, ca_pem in _serve_tls(idp, pki):
        client = FieldServiceClient(
            secure_channel(address, root_certificates=ca_pem),
            token=idp.token(**kwargs),  # type: ignore[arg-type]
        )
        with pytest.raises(grpc.RpcError) as excinfo:
            client.get_field(_FIELD)
        assert excinfo.value.code() is grpc.StatusCode.UNAUTHENTICATED, name


def test_a_token_signed_by_an_unknown_key_is_rejected(idp: _Idp) -> None:
    # The verifier resolves by `kid` against the IdP's JWKS; a token from another IdP has no entry.
    foreign = _Idp(kid="some-other-idp-key")
    with pytest.raises(AuthenticationError, match="unknown key"):
        _verifier(idp).verify(foreign.token())


def test_a_forged_signature_under_a_known_kid_is_rejected(idp: _Idp) -> None:
    # The nastier case: an attacker reuses a *known* `kid` but cannot sign for it. The key resolves,
    # and the signature check is what refuses them — so the kid is a hint, never a credential.
    impostor = _Idp(kid=idp.kid)
    with pytest.raises(AuthenticationError, match="not valid"):
        _verifier(idp).verify(impostor.token())


def test_a_token_asserting_an_unknown_capability_is_rejected(idp: _Idp) -> None:
    # An unrecognized capability is malformed, not "no capability" — quietly ignoring it would let a
    # typo'd or forged claim slide through unremarked.
    with pytest.raises(AuthenticationError, match="unrecognized capability"):
        _verifier(idp).verify(idp.token(capabilities=("prospecting.telepathy",)))


def test_malformed_authorization_metadata_is_rejected() -> None:
    interceptor = AuthInterceptor(policy=DEFAULT_POLICY, verifier=_Idp() and _verifier(_Idp()))
    with pytest.raises(AuthenticationError, match="not a bearer token"):
        interceptor._authenticate([("authorization", "Basic aGk6dGhlcmU=")])
    with pytest.raises(AuthenticationError, match="no authorization metadata"):
        interceptor._authenticate([("x-other", "value")])


def test_an_interceptor_without_any_way_to_attribute_a_caller_is_refused() -> None:
    with pytest.raises(ValueError, match="token verifier"):
        AuthInterceptor(policy=DEFAULT_POLICY)


# --- AC4 (and the crux): ground-truth-adjacent RPCs are capability-gated -------------------------


def test_submit_is_refused_without_the_ground_truth_capability_grant(
    idp: _Idp, pki: tuple[bytes, bytes, bytes, bytes, bytes]
) -> None:
    # This caller is fully authenticated and holds BOTH scopes — it simply does not hold the Core
    # GROUND_TRUTH_ACCESS grant. Since the only producer of synthetic observations is Sim's sensor
    # model reading the sealed field, a caller that could not have obtained the truth may not claim
    # to report it: the write path is refused, while the read path still works.
    token = idp.token(capabilities=())
    for address, ca_pem in _serve_tls(idp, pki):
        client = FieldServiceClient(secure_channel(address, root_certificates=ca_pem), token=token)

        assert client.get_field(_FIELD).revision == 0  # reads: allowed

        with pytest.raises(grpc.RpcError) as excinfo:
            client.submit_observations(_FIELD, [_obs()])
        assert excinfo.value.code() is grpc.StatusCode.PERMISSION_DENIED
        assert "ground_truth_access" in excinfo.value.details()


def test_reads_are_refused_without_the_read_scope(
    idp: _Idp, pki: tuple[bytes, bytes, bytes, bytes, bytes]
) -> None:
    token = idp.token(scopes=(WRITE_SCOPE,))
    for address, ca_pem in _serve_tls(idp, pki):
        client = FieldServiceClient(secure_channel(address, root_certificates=ca_pem), token=token)
        with pytest.raises(grpc.RpcError) as excinfo:
            client.get_field(_FIELD)
        assert excinfo.value.code() is grpc.StatusCode.PERMISSION_DENIED


def test_the_streaming_rpc_is_guarded_too(
    idp: _Idp, pki: tuple[bytes, bytes, bytes, bytes, bytes]
) -> None:
    # A server-streaming RPC must be guarded with its own cardinality preserved, or a denial would
    # reach the client as a protocol error rather than a status it can act on.
    token = idp.token(scopes=())
    for address, ca_pem in _serve_tls(idp, pki):
        client = FieldServiceClient(secure_channel(address, root_certificates=ca_pem), token=token)
        with pytest.raises(grpc.RpcError) as excinfo:
            next(client.stream_belief_updates(_FIELD))
        assert excinfo.value.code() is grpc.StatusCode.PERMISSION_DENIED


def test_the_default_policy_gates_exactly_the_write_path_on_the_capability() -> None:
    submit = DEFAULT_POLICY["/astro_mine.prospect.service.v1.FieldService/SubmitObservations"]
    get = DEFAULT_POLICY["/astro_mine.prospect.service.v1.FieldService/GetField"]
    stream = DEFAULT_POLICY["/astro_mine.prospect.service.v1.FieldService/StreamBeliefUpdates"]
    assert submit.capabilities == frozenset({_GT})
    assert submit.scopes == frozenset({WRITE_SCOPE})
    assert get.capabilities == frozenset() and get.scopes == frozenset({READ_SCOPE})
    assert stream.capabilities == frozenset() and stream.scopes == frozenset({READ_SCOPE})


def test_an_rpc_with_no_policy_entry_is_denied() -> None:
    # Fail-closed: adding an RPC and forgetting its policy must refuse it, never expose it.
    interceptor = AuthInterceptor(policy={}, static_principal=Principal(subject="anyone"))
    with pytest.raises(AuthorizationError, match="no authorization policy"):
        interceptor._authorize(Principal(subject="anyone"), "/svc/BrandNewRpc")


def test_method_policy_reports_every_missing_grant() -> None:
    policy = MethodPolicy(scopes=frozenset({READ_SCOPE}), capabilities=frozenset({_GT}))
    policy.check(Principal("ok", frozenset({READ_SCOPE}), frozenset({_GT})), "/svc/M")  # no raise
    with pytest.raises(AuthorizationError, match="missing scope"):
        policy.check(Principal("no-scope", frozenset(), frozenset({_GT})), "/svc/M")
    with pytest.raises(AuthorizationError, match="missing capability grant"):
        policy.check(Principal("no-cap", frozenset({READ_SCOPE}), frozenset()), "/svc/M")


# --- the verified principal speaks Core's own capability vocabulary ------------------------------


def test_a_verified_principal_carries_core_capability_tags(idp: _Idp) -> None:
    principal = _verifier(idp).verify(idp.token())
    assert principal.subject == "sim-sensor-model"
    assert principal.scopes == frozenset({READ_SCOPE, WRITE_SCOPE})
    # The network gate and the in-process seal speak the *same* vocabulary — Core's, not a parallel
    # string namespace that could drift from it.
    assert principal.capabilities == frozenset({CapabilityTag.GROUND_TRUTH_ACCESS})


def test_scopes_are_read_from_either_standard_claim(idp: _Idp) -> None:
    token = jwt.encode(
        {
            "sub": "s",
            "iss": _ISSUER,
            "aud": _AUDIENCE,
            "exp": dt.datetime.now(dt.UTC) + dt.timedelta(minutes=5),
            "scp": [READ_SCOPE],  # the list form, instead of the space-delimited `scope` string
        },
        idp._key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ),
        algorithm="ES256",
        headers={"kid": _KID},
    )
    assert _verifier(idp).verify(token).scopes == frozenset({READ_SCOPE})


def test_bearer_metadata_is_the_authorization_header() -> None:
    assert bearer_metadata("abc") == (("authorization", "Bearer abc"),)


def test_tls_config_refuses_mutual_tls_without_a_client_ca() -> None:
    with pytest.raises(ValueError, match="client_ca"):
        ServerTls(certificate_chain=b"x", private_key=b"y", require_client_auth=True).credentials()
