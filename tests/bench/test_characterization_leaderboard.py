"""Characterization tests for the leaderboard *library* (no FastAPI, no TestClient).

astro-mine-platform deliberately did not migrate the REST route module
(``astro_mine.bench.leaderboard._app``), which excluded the original TestClient-based suites
(``test_leaderboard*.py``). The library underneath — :class:`LeaderboardService` and the modules it
composes (``_auth``/``_authz``/``_objects``/``_supply_chain``/``_eval``/``_hub``/``_provenance``) —
is fully present; these tests pin its **current** behavior by driving the service and its building
blocks directly, exactly as the service docstring says it was designed to be driven.

Everything runs offline: in-memory stores, a throwaway per-run RSA IdP
(``tests.bench._factories.make_idp``), a local OCI-layout Hub registry under ``tmp_path``, an
httpx ``MockTransport`` for the JWKS/OPA HTTP seams, and the fast in-process sandbox double.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import jwt
import pytest

from astro_mine.bench.baseline import BaselinePolicy
from astro_mine.bench.leaderboard import (
    EMBARGO_ROOT,
    EMBARGO_ROOT_ENV,
    Action,
    AttestationPolicy,
    AuditDecision,
    AuthenticationError,
    AuthorizationRequest,
    FileObjectStore,
    HubResolutionError,
    HubSubmissionRequest,
    InMemoryAuditLog,
    InMemoryObjectStore,
    InMemoryRateLimiter,
    LeaderboardService,
    ManifestInterfaceError,
    ObjectIntegrityError,
    OidcTokenVerifier,
    OpaPolicyEngine,
    Principal,
    ProvenanceBundle,
    RbacPolicyEngine,
    Role,
    SubmissionRejected,
    SubmissionRequest,
    SubmissionStatus,
    SupplyChainRejected,
    attestation_policy_from_env,
    bearer_token,
    build_provenance_bundle,
    build_submission,
    evaluate,
    load_heldout_seeds,
    oidc_verifier_from_env,
    open_registry,
    policy_engine_from_env,
    rank,
    reference_policy_loader,
    resample_from_bundle,
    resolve_embargo_root,
    resolve_policy,
    resolve_submission,
    submission_policy_ref,
    validate_policy_ref,
    validate_submission_manifest,
    verify_submission_attestations,
)
from astro_mine.bench.leaderboard._auth import (
    AUDIENCE_ENV,
    ISSUER_ENV,
    JWKS_URL_ENV,
    ROLES_CLAIM_ENV,
)
from astro_mine.bench.leaderboard._authz import EMBARGOED_SCENARIOS_ENV, OPA_URL_ENV
from astro_mine.bench.leaderboard._eval import PolicyReferenceError, _sample_reproduces
from astro_mine.bench.leaderboard._objects import blob_digest
from astro_mine.bench.leaderboard._supply_chain import TRUSTED_KEY_ENV
from astro_mine.bench.metrics import Scorecard
from astro_mine.bench.sandbox import SandboxScorer
from astro_mine.bench.scenario import ScenarioSpec
from astro_mine.bench.zoo import ANCHOR_SCENARIO_ID, load_scenario
from astro_mine.core.registry import PluginKind
from astro_mine.core.registry.model import PluginManifest
from astro_mine.hub.registry import Blob, Registry
from astro_mine.hub.supply_chain import attest, generate_keypair
from tests.bench._factories import (
    BASELINE_REF,
    EXPLODING_REF,
    NONDETERMINISTIC_REF,
    InProcessSandbox,
    TestIdp,
    make_idp,
)

CLASS_REF = "astro_mine.bench.baseline:BaselinePolicy"
FACTORY_REF = "tests.bench._factories:idle_baseline"

#: The anchor scenario's scored metric set — pinned from the original suite.
ANCHOR_METRICS = frozenset(
    {
        "water_mass",
        "energy_per_kg",
        "information_gain",
        "psr_area_characterized",
        "nights_survived",
        "comms_robustness",
        "discovery_latency",
    }
)


@pytest.fixture(scope="module")
def anchor() -> ScenarioSpec:
    return load_scenario(ANCHOR_SCENARIO_ID)


@pytest.fixture(scope="module")
def idp() -> TestIdp:
    """A throwaway IdP: a real RSA key, its JWKS, and a minter for real RS256 bearer tokens."""
    return make_idp()


@pytest.fixture(scope="module")
def verifier(idp: TestIdp) -> OidcTokenVerifier:
    return OidcTokenVerifier(issuer=idp.issuer, audience=idp.audience, jwks=idp.jwks)


@pytest.fixture(scope="module")
def scorer() -> SandboxScorer:
    """The execution seam, over the fast in-process sandbox double (see tests/bench/_factories)."""
    return SandboxScorer(InProcessSandbox())


@pytest.fixture(scope="module")
def heldout() -> tuple[int, ...]:
    return load_heldout_seeds(ANCHOR_SCENARIO_ID)


@pytest.fixture(scope="module")
def heldout_card(
    anchor: ScenarioSpec, scorer: SandboxScorer, heldout: tuple[int, ...]
) -> Scorecard:
    """One baseline scoring run on the held-out seeds, shared by the read-only tests."""
    return scorer(anchor, BASELINE_REF, seeds=heldout)


@pytest.fixture
def audit() -> InMemoryAuditLog:
    return InMemoryAuditLog()


@pytest.fixture
def service(
    verifier: OidcTokenVerifier, audit: InMemoryAuditLog, scorer: SandboxScorer
) -> LeaderboardService:
    return LeaderboardService(authn=verifier, audit=audit, scorer=scorer)


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    return Registry(tmp_path / "hub-registry")


def principal(subject: str = "lab-1", *roles: str) -> Principal:
    return Principal(
        subject=subject, issuer="https://idp.test", roles=tuple(roles or ("submitter",))
    )


def _publish(
    registry: Registry,
    *,
    name: str = "acme/prospector",
    version: str = "1.0.0",
    interfaces: dict[str, str],
    entrypoint: str | None = BASELINE_REF,
    kind: PluginKind = PluginKind.POLICY,
    attested: bool = False,
) -> str:
    """Publish a policy artifact (a real Core plugin manifest + one ONNX payload layer)."""
    manifest = PluginManifest(
        name=name,
        version=version,
        kind=kind,
        core_interfaces=dict(interfaces),
        inputs=["Observation"],
        outputs=["ActionBatch"],
        attributes={} if entrypoint is None else {"entrypoint": entrypoint},
    )
    published = registry.publish(
        name=name,
        version=version,
        kind=PluginKind.POLICY.value if kind is PluginKind.POLICY else "world",
        # The bare manifest — the stored form (hub.md §2 principle 2, astro-mine-platform#14).
        config=manifest.model_dump(mode="json"),
        layers=[Blob("application/vnd.astro-mine.policy.onnx.v1", b"onnx-model-bytes")],
    )
    if attested:
        private_pem, _ = generate_keypair()
        attest(registry, published.digest, private_key_pem=private_pem, name=name, version=version)
    return str(published.digest)


# =================================================================================================
# _auth.py — bearer-token extraction + the OIDC verifier
# =================================================================================================


@pytest.mark.parametrize(
    "header", [None, "", "Bearer", "Bearer   ", "Basic abcdef", "abcdef", "Token abcdef"]
)
def test_bearer_token_extraction_fails_closed(header: str | None) -> None:
    """A missing/malformed/non-Bearer Authorization header is never an anonymous pass-through."""
    with pytest.raises(AuthenticationError):
        bearer_token(header)


def test_bearer_token_is_extracted(idp: TestIdp) -> None:
    token = idp.token()
    assert bearer_token(f"Bearer {token}") == token
    assert bearer_token(f"bearer {token}") == token  # the scheme match is case-insensitive


def test_expired_token_is_rejected(verifier: OidcTokenVerifier, idp: TestIdp) -> None:
    with pytest.raises(AuthenticationError, match="rejected"):
        verifier.verify(idp.token(expires_in=-3600))


def test_wrong_audience_is_rejected(verifier: OidcTokenVerifier, idp: TestIdp) -> None:
    with pytest.raises(AuthenticationError, match="rejected"):
        verifier.verify(idp.token(audience="some-other-service"))


def test_wrong_issuer_is_rejected(verifier: OidcTokenVerifier, idp: TestIdp) -> None:
    with pytest.raises(AuthenticationError, match="rejected"):
        verifier.verify(idp.token(issuer="https://evil.test"))


def test_token_signed_by_an_unknown_key_is_rejected(verifier: OidcTokenVerifier) -> None:
    """A valid-looking token minted by a *different* IdP key must not authenticate."""
    attacker = make_idp()
    with pytest.raises(AuthenticationError):
        verifier.verify(attacker.token(subject="attacker", roles=("admin",)))


def test_unsigned_alg_none_token_is_rejected(verifier: OidcTokenVerifier, idp: TestIdp) -> None:
    """The classic JWT forgery: ``alg: none``. Only asymmetric algorithms are admissible."""
    forged = jwt.encode(
        {"sub": "attacker", "iss": idp.issuer, "aud": idp.audience, "exp": 9_999_999_999},
        key="",
        algorithm="none",
    )
    with pytest.raises(AuthenticationError):
        verifier.verify(forged)


def test_garbage_token_is_rejected(verifier: OidcTokenVerifier) -> None:
    with pytest.raises(AuthenticationError, match="malformed"):
        verifier.verify("not-a-jwt-at-all")


def test_token_claims_become_the_principal(verifier: OidcTokenVerifier, idp: TestIdp) -> None:
    caller = verifier.verify(
        idp.token(subject="lab-7", roles=("maintainer", "submitter"), email="a@lab.test")
    )
    assert caller.subject == "lab-7"
    assert caller.issuer == idp.issuer
    assert set(caller.roles) == {"maintainer", "submitter"}
    assert caller.scopes == ("openid", "profile")
    assert caller.email == "a@lab.test"
    # The quota/audit key is issuer-qualified and comes from the token, not any request body.
    assert caller.identity == f"{idp.issuer}#lab-7"


def test_empty_subject_claim_is_rejected(verifier: OidcTokenVerifier, idp: TestIdp) -> None:
    """A token whose ``sub`` is present but empty fails the post-decode subject check."""
    with pytest.raises(AuthenticationError, match="no subject claim"):
        verifier.verify(idp.token(subject=""))


def test_roles_claim_string_form_is_split_on_whitespace(idp: TestIdp) -> None:
    """IdPs render roles as a list or a space-delimited string; both normalize to a tuple.

    Pinned by pointing ``roles_claim`` at the (string-valued) ``scope`` claim.
    """
    verifier = OidcTokenVerifier(
        issuer=idp.issuer, audience=idp.audience, jwks=idp.jwks, roles_claim="scope"
    )
    assert verifier.verify(idp.token()).roles == ("openid", "profile")


def test_verifier_requires_issuer_audience_and_a_key_source() -> None:
    with pytest.raises(ValueError, match="issuer and an audience"):
        OidcTokenVerifier(issuer="", audience="a", jwks={"keys": []})
    with pytest.raises(ValueError, match="jwks mapping or a jwks_url"):
        OidcTokenVerifier(issuer="https://i", audience="a")


def test_an_empty_static_jwks_matches_no_key(idp: TestIdp) -> None:
    """A static empty key set (no jwks_url to refetch from) fails closed on the kid lookup."""
    verifier = OidcTokenVerifier(issuer=idp.issuer, audience=idp.audience, jwks={"keys": []})
    with pytest.raises(AuthenticationError, match="no JWKS key matches"):
        verifier.verify(idp.token())


def test_jwks_is_fetched_over_http_and_cached(idp: TestIdp) -> None:
    """A deployment points the verifier at the issuer's JWKS URL; the fetched set is cached."""
    fetches: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        fetches.append(str(request.url))
        return httpx.Response(200, json=idp.jwks)

    http = httpx.Client(transport=httpx.MockTransport(handler))
    remote = OidcTokenVerifier(
        issuer=idp.issuer, audience=idp.audience, jwks_url="https://idp.test/certs", http=http
    )
    assert remote.verify(idp.token()).subject == "lab-1"
    assert remote.verify(idp.token()).subject == "lab-1"
    assert fetches == ["https://idp.test/certs"]  # cached: the second verify does not refetch


def test_unreachable_jwks_fails_closed(idp: TestIdp) -> None:
    http = httpx.Client(transport=httpx.MockTransport(lambda _r: httpx.Response(503)))
    remote = OidcTokenVerifier(
        issuer=idp.issuer, audience=idp.audience, jwks_url="https://idp.test/certs", http=http
    )
    with pytest.raises(AuthenticationError, match="could not fetch"):
        remote.verify(idp.token())


def test_unknown_kid_refetches_once_then_fails_closed(idp: TestIdp) -> None:
    """A rotated IdP key gets exactly one refetch; an unknown kid never becomes a fetch loop."""
    fetches: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        fetches.append(str(request.url))
        return httpx.Response(200, json={"keys": []})  # a key set that matches nothing

    http = httpx.Client(transport=httpx.MockTransport(handler))
    remote = OidcTokenVerifier(
        issuer=idp.issuer, audience=idp.audience, jwks_url="https://idp.test/certs", http=http
    )
    with pytest.raises(AuthenticationError, match="no JWKS key matches"):
        remote.verify(idp.token())
    assert len(fetches) == 2  # the initial fetch, then exactly one refresh — and no more


def test_oidc_verifier_from_env() -> None:
    """Unconfigured ⇒ None (the deployment refuses writes); configured ⇒ a verifier."""
    assert oidc_verifier_from_env({}) is None
    assert oidc_verifier_from_env({ISSUER_ENV: "https://idp.test/realms/x"}) is None  # no audience

    built = oidc_verifier_from_env(
        {
            ISSUER_ENV: "https://idp.test/realms/x",
            AUDIENCE_ENV: "astro-mine-bench",
            JWKS_URL_ENV: "https://idp.test/certs",
            ROLES_CLAIM_ENV: "groups",
        }
    )
    assert built is not None
    assert built.issuer == "https://idp.test/realms/x"


# =================================================================================================
# _authz.py — RBAC, quotas, embargo; the OPA sidecar engine
# =================================================================================================


def _request(action: Action, *, roles: tuple[str, ...] = ("submitter",), **context: Any) -> Any:
    return AuthorizationRequest(
        principal=principal("lab-1", *roles),
        action=action,
        resource=context.pop("resource", ANCHOR_SCENARIO_ID),
        context=context,
    )


def test_rbac_grants_by_role() -> None:
    engine = RbacPolicyEngine()
    assert engine.evaluate(_request(Action.SUBMISSION_CREATE)).allow
    assert engine.evaluate(_request(Action.SUBMISSION_CREATE_HUB)).allow
    # ...but a submitter may not mutate the board, author scenarios, or read the audit trail.
    for action in (Action.RANKING_MUTATE, Action.SCENARIO_AUTHOR, Action.AUDIT_READ):
        decision = engine.evaluate(_request(action))
        assert not decision.allow
        assert "do not grant" in decision.reason


def test_rbac_admin_grants_everything() -> None:
    engine = RbacPolicyEngine()
    for action in Action:
        assert engine.evaluate(_request(action, roles=("admin",))).allow


def test_maintainer_holds_the_authoring_rights() -> None:
    engine = RbacPolicyEngine()
    for action in (Action.SCENARIO_AUTHOR, Action.METRIC_AUTHOR, Action.EMBARGO_READ):
        assert engine.evaluate(_request(action, roles=("maintainer",))).allow
        assert not engine.evaluate(_request(action, roles=("submitter",))).allow


def test_no_known_role_denies_everything() -> None:
    engine = RbacPolicyEngine()
    decision = engine.evaluate(_request(Action.SUBMISSION_CREATE, roles=("wizard",)))
    assert not decision.allow
    assert "no known role" in decision.reason


def test_submission_quota_is_enforced_per_role() -> None:
    """The per-user cap: submitter 20, maintainer 200, admin uncapped; over-quota denies."""
    engine = RbacPolicyEngine()
    assert engine.evaluate(_request(Action.SUBMISSION_CREATE, submissions_in_window=20)).allow
    over = engine.evaluate(_request(Action.SUBMISSION_CREATE, submissions_in_window=21))
    assert not over.allow
    assert "quota exhausted" in over.reason
    assert engine.evaluate(
        _request(Action.SUBMISSION_CREATE, roles=("maintainer",), submissions_in_window=21)
    ).allow
    assert engine.evaluate(
        _request(Action.SUBMISSION_CREATE, roles=("admin",), submissions_in_window=100_000)
    ).allow


def test_quota_takes_the_most_generous_of_multiple_roles() -> None:
    """A submitter+maintainer principal gets the maintainer's larger cap."""
    engine = RbacPolicyEngine()
    assert engine.evaluate(
        _request(
            Action.SUBMISSION_CREATE,
            roles=("submitter", "maintainer"),
            submissions_in_window=21,
        )
    ).allow


def test_quota_does_not_apply_to_non_submission_actions() -> None:
    engine = RbacPolicyEngine()
    assert engine.evaluate(
        _request(Action.AUDIT_READ, roles=("admin",), submissions_in_window=999_999)
    ).allow


def test_embargoed_scenario_needs_the_embargo_right() -> None:
    engine = RbacPolicyEngine(embargoed_scenarios=frozenset({"secret-scenario-v1"}))
    denied = engine.evaluate(_request(Action.SUBMISSION_CREATE, scenario_id="secret-scenario-v1"))
    assert not denied.allow
    assert "under embargo" in denied.reason
    # A maintainer holds embargo:read, so the same submission is allowed.
    assert engine.evaluate(
        _request(Action.SUBMISSION_CREATE, roles=("maintainer",), scenario_id="secret-scenario-v1")
    ).allow


def test_allow_decision_names_the_granting_roles() -> None:
    decision = RbacPolicyEngine().evaluate(_request(Action.SUBMISSION_CREATE))
    assert decision.allow
    assert "granted by" in decision.reason and "submitter" in decision.reason


def test_authorization_request_renders_the_opa_input_document() -> None:
    request = _request(Action.SUBMISSION_CREATE_HUB, submissions_in_window=2)
    assert request.to_input() == {
        "principal": {
            "subject": "lab-1",
            "issuer": "https://idp.test",
            "roles": ["submitter"],
            "scopes": [],
        },
        "action": "submission:create_hub",
        "resource": ANCHOR_SCENARIO_ID,
        "context": {"submissions_in_window": 2},
    }


def test_opa_sidecar_decides_the_same_input_document() -> None:
    """The sidecar and the in-process engine exchange the *same* input/result documents."""
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content)["input"])
        return httpx.Response(200, json={"result": {"allow": True, "reason": "allowed by OPA"}})

    engine = OpaPolicyEngine(
        "http://opa:8181/v1/data/astromine/bench/decision",
        http=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    decision = engine.evaluate(_request(Action.SUBMISSION_CREATE_HUB, submissions_in_window=2))
    assert decision.allow and decision.reason == "allowed by OPA"
    assert seen[0]["action"] == "submission:create_hub"
    assert seen[0]["principal"]["roles"] == ["submitter"]
    assert seen[0]["context"]["submissions_in_window"] == 2


def test_opa_denial_carries_its_reason() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"result": {"allow": False, "reason": "quota exhausted"}})

    engine = OpaPolicyEngine(
        "http://opa:8181/x", http=httpx.Client(transport=httpx.MockTransport(handler))
    )
    decision = engine.evaluate(_request(Action.SUBMISSION_CREATE))
    assert not decision.allow and decision.reason == "quota exhausted"


@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (500, {"result": {"allow": True}}, "failed"),  # a 5xx never means yes
        (200, {}, "undefined policy rule"),  # OPA returns {} for an undefined rule
        (200, {"result": {}}, "denied by OPA"),  # a result with no explicit allow
        (200, {"result": {"allow": "yes"}}, "denied by OPA"),  # allow must be exactly True
    ],
)
def test_opa_fails_closed(status: int, body: dict[str, Any], expected: str) -> None:
    """An authorization service that cannot answer must never mean 'yes'."""
    transport = httpx.MockTransport(lambda _r: httpx.Response(status, json=body))
    engine = OpaPolicyEngine("http://opa:8181/x", http=httpx.Client(transport=transport))
    decision = engine.evaluate(_request(Action.SUBMISSION_CREATE))
    assert not decision.allow
    assert expected in decision.reason


def test_opa_unreachable_fails_closed() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    engine = OpaPolicyEngine(
        "http://opa:8181/x", http=httpx.Client(transport=httpx.MockTransport(handler))
    )
    assert not engine.evaluate(_request(Action.SUBMISSION_CREATE)).allow


def test_opa_engine_requires_a_url() -> None:
    with pytest.raises(ValueError, match="URL of an OPA"):
        OpaPolicyEngine("")


def test_policy_engine_from_env() -> None:
    engine = policy_engine_from_env({EMBARGOED_SCENARIOS_ENV: "a-v1, b-v2"})
    assert isinstance(engine, RbacPolicyEngine)
    assert not engine.evaluate(_request(Action.SUBMISSION_CREATE, scenario_id="a-v1")).allow
    assert not engine.evaluate(_request(Action.SUBMISSION_CREATE, scenario_id="b-v2")).allow
    assert engine.evaluate(_request(Action.SUBMISSION_CREATE, scenario_id="c-v3")).allow

    opa = policy_engine_from_env({OPA_URL_ENV: "http://opa:8181/v1/data/x"})
    assert isinstance(opa, OpaPolicyEngine)


def test_role_and_action_vocabularies_are_stable() -> None:
    """astro-mine-api's deploy/policy/bench.rego hard-codes these strings; a rename breaks here.

    The Rego moved with the tier that deploys it (RM-DIST-03); these vocabularies did not, because
    they are library types. This is the pin that keeps the two spellings in step across the repo
    boundary -- rename a Role or an Action here and this fails before the policy silently stops
    matching."""
    assert {str(role) for role in Role} == {"submitter", "maintainer", "admin"}
    assert {str(action) for action in Action} == {
        "submission:create",
        "submission:create_hub",
        "ranking:mutate",
        "scenario:author",
        "metric:author",
        "embargo:read",
        "audit:read",
    }


# =================================================================================================
# _objects.py — the content-addressed object store
# =================================================================================================


@pytest.fixture(params=["memory", "file"])
def object_store(request: pytest.FixtureRequest, tmp_path: Path) -> Any:
    return InMemoryObjectStore() if request.param == "memory" else FileObjectStore(tmp_path / "obj")


def test_blob_digest_is_a_sha256_content_address() -> None:
    digest = blob_digest(b"provenance-bundle-bytes")
    assert digest.startswith("sha256:") and len(digest) == len("sha256:") + 64
    assert digest == blob_digest(b"provenance-bundle-bytes")  # deterministic
    assert digest != blob_digest(b"other-bytes")


def test_object_store_roundtrip_and_absence(object_store: Any) -> None:
    data = b"provenance-bundle-bytes"
    digest = object_store.put(data)
    assert digest == blob_digest(data)
    assert object_store.get(digest) == data
    assert object_store.contains(digest)
    assert object_store.get("sha256:" + "0" * 64) is None
    assert not object_store.contains("sha256:" + "0" * 64)


def test_object_store_put_is_idempotent(object_store: Any) -> None:
    assert object_store.put(b"same") == object_store.put(b"same")


def test_file_object_store_verifies_on_read(tmp_path: Path) -> None:
    """A corrupted/swapped on-disk object must never be served as authentic."""
    store = FileObjectStore(tmp_path / "obj")
    digest = store.put(b"authentic")
    hexpart = digest.split(":", 1)[1]
    (tmp_path / "obj" / "sha256" / hexpart[:2] / hexpart).write_bytes(b"corrupted")
    with pytest.raises(ObjectIntegrityError, match="content-address mismatch"):
        store.get(digest)


def test_in_memory_object_store_verifies_on_read() -> None:
    """The in-memory backend applies the same verify-on-read as the on-disk one."""
    store = InMemoryObjectStore()
    digest = store.put(b"authentic")
    store._blobs[digest] = b"swapped"  # simulate backend corruption
    with pytest.raises(ObjectIntegrityError, match="content-address mismatch"):
        store.get(digest)


@pytest.mark.parametrize("bad", ["md5:" + "0" * 64, "sha256:short", "0" * 64, ""])
def test_file_object_store_rejects_non_sha256_digests(tmp_path: Path, bad: str) -> None:
    store = FileObjectStore(tmp_path / "obj")
    with pytest.raises(ValueError, match="not a sha256 object digest"):
        store.get(bad)


# =================================================================================================
# _supply_chain.py — cosign + SLSA + SBOM, fail-closed, before execution
# =================================================================================================


def test_unsigned_submission_is_rejected(registry: Registry, anchor: ScenarioSpec) -> None:
    """A content hash is not a signature: an artifact with no attestations must not run."""
    digest = _publish(registry, interfaces=anchor.core_interface)
    with pytest.raises(SupplyChainRejected, match="failed supply-chain verification"):
        verify_submission_attestations(registry, digest, AttestationPolicy())


def test_signed_and_attested_submission_verifies(registry: Registry, anchor: ScenarioSpec) -> None:
    """The happy path: cosign signature + SLSA provenance + SBOM, verified via Seal through Hub."""
    digest = _publish(registry, interfaces=anchor.core_interface)
    private_pem, public_pem = generate_keypair()
    attest(registry, digest, private_key_pem=private_pem, name="acme/prospector", version="1.0.0")

    verdict = verify_submission_attestations(
        registry, digest, AttestationPolicy(trusted_public_key_pem=public_pem)
    )
    assert verdict.verified
    assert verdict.signer_pinned
    assert verdict.subject == digest
    assert set(verdict.required) == {"signature", "slsa", "sbom"}
    assert verdict.detail is None


def test_unpinned_policy_verifies_but_is_not_signer_pinned(
    registry: Registry, anchor: ScenarioSpec
) -> None:
    """With no trusted key, any intact cosign signature is accepted — but recorded as unpinned."""
    digest = _publish(registry, interfaces=anchor.core_interface, attested=True)
    verdict = verify_submission_attestations(registry, digest, AttestationPolicy())
    assert verdict.verified
    assert not verdict.signer_pinned


def test_submission_signed_by_an_untrusted_key_is_rejected(
    registry: Registry, anchor: ScenarioSpec
) -> None:
    """A *valid* signature by the wrong signer is a rejection when the trust root is pinned."""
    digest = _publish(registry, interfaces=anchor.core_interface)
    attacker_private, _ = generate_keypair()
    _, our_public = generate_keypair()
    attest(
        registry, digest, private_key_pem=attacker_private, name="acme/prospector", version="1.0.0"
    )
    with pytest.raises(SupplyChainRejected):
        verify_submission_attestations(
            registry, digest, AttestationPolicy(trusted_public_key_pem=our_public)
        )


def test_a_broken_registry_is_wrapped_not_leaked() -> None:
    """A registry failure surfaces as a SupplyChainRejected, never a raw AttributeError.

    Hub's verify wraps registry-side failures into its own SupplyChainError, so this lands on the
    "failed supply-chain verification" message, not the generic fallback.
    """
    with pytest.raises(SupplyChainRejected, match="failed supply-chain verification"):
        verify_submission_attestations(object(), "sha256:" + "0" * 64, AttestationPolicy())


def test_a_non_supply_chain_failure_is_wrapped_with_its_type_named() -> None:
    """A failure Hub does not wrap (here a TypeError on a bad require token) still rejects.

    AttestationPolicy does not validate element types, so a non-string require token reaches
    Seal and explodes as a TypeError — pinned: the service converts it to SupplyChainRejected
    rather than leaking it.
    """
    policy = AttestationPolicy(required=("signature", 42))  # type: ignore[arg-type]
    with pytest.raises(SupplyChainRejected, match="could not be supply-chain verified"):
        verify_submission_attestations(object(), "sha256:" + "0" * 64, policy)


def test_verification_cannot_be_switched_off() -> None:
    """A deployment may pin *which* signer it trusts; it may not require *nothing*."""
    with pytest.raises(ValueError, match="must not be empty"):
        AttestationPolicy(required=())


def test_attestation_policy_from_env_loads_the_public_trust_root(tmp_path: Path) -> None:
    _, public_pem = generate_keypair()
    key_file = tmp_path / "cosign.pub"
    key_file.write_bytes(public_pem)

    assert attestation_policy_from_env({}).trusted_public_key_pem is None
    assert attestation_policy_from_env({TRUSTED_KEY_ENV: str(key_file)}).trusted_public_key_pem == (
        public_pem
    )


def test_an_unresolvable_trust_root_refuses_to_start() -> None:
    with pytest.raises(SupplyChainRejected, match="unresolvable trust root"):
        attestation_policy_from_env({TRUSTED_KEY_ENV: "/no/such/cosign.pub"})


# =================================================================================================
# _hub.py — resolve-by-digest, manifest validation, entrypoint reading
# =================================================================================================


def test_resolve_submission_by_digest_and_by_tag(registry: Registry, anchor: ScenarioSpec) -> None:
    digest = _publish(registry, interfaces=anchor.core_interface)
    resolved = resolve_submission(registry, digest)
    assert resolved.reference == digest
    assert resolved.manifest_digest == digest
    assert resolved.manifest.name == "acme/prospector"
    assert len(resolved.layer_digests) == 1
    assert resolved.layer_digests[0].startswith("sha256:")
    # A name:version tag resolves to the same immutable digest.
    tagged = resolve_submission(registry, "acme/prospector:1.0.0")
    assert tagged.manifest_digest == digest


def test_unresolvable_reference_is_a_hub_resolution_error(registry: Registry) -> None:
    with pytest.raises(HubResolutionError, match="cannot resolve Hub reference"):
        resolve_submission(registry, "sha256:" + "0" * 64)


def test_tampered_blob_fails_closed(
    registry: Registry, tmp_path: Path, anchor: ScenarioSpec
) -> None:
    """A stored blob that no longer hashes to its content address must not resolve."""
    digest = _publish(registry, interfaces=anchor.core_interface)
    image = registry.read_manifest(digest)
    layer_hex = image["layers"][0]["digest"].split(":", 1)[1]
    (tmp_path / "hub-registry" / "blobs" / "sha256" / layer_hex).write_bytes(b"tampered-bytes")
    with pytest.raises(HubResolutionError, match="integrity verification failed"):
        resolve_submission(registry, digest)


def test_an_unparseable_config_is_an_invalid_plugin_manifest() -> None:
    """A registry that serves junk config bytes fails the manifest parse, not the caller."""
    stub = SimpleNamespace(
        resolve=lambda reference: SimpleNamespace(digest="sha256:" + "a" * 64),
        verify=lambda digest: None,
        read_manifest=lambda digest: {"layers": []},
        read_config=lambda digest: b"not-a-manifest",
    )
    with pytest.raises(HubResolutionError, match="invalid plugin manifest"):
        resolve_submission(stub, "acme/junk:1.0.0")


def test_non_policy_artifact_is_rejected(registry: Registry, anchor: ScenarioSpec) -> None:
    digest = _publish(
        registry,
        name="acme/world",
        interfaces=anchor.core_interface,
        kind=PluginKind.WORLD_PROVIDER,
    )
    resolved = resolve_submission(registry, digest)
    with pytest.raises(ManifestInterfaceError, match="must be a policy"):
        validate_submission_manifest(resolved, anchor)


def test_interface_mismatch_is_rejected(registry: Registry, anchor: ScenarioSpec) -> None:
    bad = dict(anchor.core_interface)
    bad[next(iter(bad))] = "9.0.0"
    digest = _publish(registry, name="acme/future", interfaces=bad)
    resolved = resolve_submission(registry, digest)
    with pytest.raises(ManifestInterfaceError, match="do not satisfy"):
        validate_submission_manifest(resolved, anchor)


def test_missing_interface_is_rejected(registry: Registry, anchor: ScenarioSpec) -> None:
    """A manifest that declares none of the scenario's pinned interfaces cannot satisfy it."""
    digest = _publish(registry, name="acme/empty", interfaces={})
    resolved = resolve_submission(registry, digest)
    with pytest.raises(ManifestInterfaceError, match="do not satisfy"):
        validate_submission_manifest(resolved, anchor)


def test_compatible_manifest_validates(registry: Registry, anchor: ScenarioSpec) -> None:
    digest = _publish(registry, interfaces=anchor.core_interface)
    validate_submission_manifest(resolve_submission(registry, digest), anchor)  # does not raise


def test_submission_policy_ref_reads_without_importing(
    registry: Registry, anchor: ScenarioSpec
) -> None:
    """The service handles the reference *string*, never a live Policy object (bench#30)."""
    digest = _publish(registry, interfaces=anchor.core_interface, entrypoint=BASELINE_REF)
    assert submission_policy_ref(resolve_submission(registry, digest)) == BASELINE_REF


def test_a_manifest_without_an_entrypoint_is_rejected(
    registry: Registry, anchor: ScenarioSpec
) -> None:
    digest = _publish(registry, name="acme/no-entry", interfaces=anchor.core_interface,
                      entrypoint=None)
    resolved = resolve_submission(registry, digest)
    with pytest.raises(HubResolutionError, match="no 'entrypoint'"):
        submission_policy_ref(resolved)


def test_reference_policy_loader_materializes_the_entrypoint(
    registry: Registry, anchor: ScenarioSpec
) -> None:
    """The dependency-clean loader resolves the entrypoint to a live Policy (sandbox-side only)."""
    digest = _publish(registry, interfaces=anchor.core_interface, entrypoint=BASELINE_REF)
    policy = reference_policy_loader(resolve_submission(registry, digest), registry)
    assert isinstance(policy, BaselinePolicy)


def test_reference_policy_loader_wraps_a_bad_entrypoint(
    registry: Registry, anchor: ScenarioSpec
) -> None:
    digest = _publish(
        registry, name="acme/bad", interfaces=anchor.core_interface,
        entrypoint="no_such_module_xyz:thing",
    )
    with pytest.raises(HubResolutionError, match="did not load"):
        reference_policy_loader(resolve_submission(registry, digest), registry)


def test_open_registry_returns_the_hub_registry(tmp_path: Path) -> None:
    opened = open_registry(tmp_path / "r")
    assert isinstance(opened, Registry)


# =================================================================================================
# _eval.py — policy-ref resolution, held-out seeds, evaluate, build_submission, rank
# =================================================================================================


def test_resolve_policy_class_instance_and_factory() -> None:
    assert isinstance(resolve_policy(CLASS_REF), BaselinePolicy)  # a class
    assert isinstance(resolve_policy(BASELINE_REF), BaselinePolicy)  # an instance
    assert isinstance(resolve_policy(FACTORY_REF), BaselinePolicy)  # a zero-arg factory


@pytest.mark.parametrize(
    "ref",
    [
        "no-colon-here",
        "astro_mine.bench.baseline:Missing",
        "no_such_module_xyz:thing",
        "astro_mine.bench.leaderboard:InMemoryStore",  # resolves, but is not a Policy
    ],
)
def test_resolve_policy_rejects_bad_refs(ref: str) -> None:
    with pytest.raises(PolicyReferenceError):
        resolve_policy(ref)


def test_validate_policy_ref_checks_shape_without_importing() -> None:
    """The edge rejects a malformed ref without importing it (importing = executing)."""
    assert validate_policy_ref(BASELINE_REF) == BASELINE_REF
    # A well-shaped but unimportable reference passes the *shape* check; the sandboxed worker
    # discovers it does not import and hands that back as data.
    assert validate_policy_ref("no_such_module_xyz:thing") == "no_such_module_xyz:thing"
    for bad in ("no-colon-here", ":attr", "module:", "  :  "):
        with pytest.raises(PolicyReferenceError):
            validate_policy_ref(bad)


def test_load_heldout_seeds_for_the_anchor(heldout: tuple[int, ...]) -> None:
    assert len(heldout) == 12
    assert all(isinstance(seed, int) for seed in heldout)


def test_load_heldout_seeds_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no held-out seed set"):
        load_heldout_seeds("nope", embargo_root=tmp_path)


# --- the deployment override (#15) ---------------------------------------------------------------
#
# `EMBARGO_ROOT` is derived from this module's own location, which is right in a checkout and wrong
# on an installed wheel — there it points inside `site-packages`, while `embargo/` ships with
# astro-mine-api, the repository the hosted leaderboard runs from. Every submission to a served
# deployment answered 404 "no held-out seed set", and nothing saw it: the three places that score
# in-process each rebound the keyword default first, and a served process had no way to.


def test_the_embargo_root_comes_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A deployment can point the lookup at the checkout it runs from."""
    sealed = tmp_path / "moved" / "elsewhere-v1"
    sealed.mkdir(parents=True)
    (sealed / "heldout_seeds.json").write_text(json.dumps({"seeds": [11, 22, 33]}))

    monkeypatch.setenv(EMBARGO_ROOT_ENV, str(tmp_path / "moved"))
    assert resolve_embargo_root() == tmp_path / "moved"
    assert load_heldout_seeds("elsewhere-v1") == (11, 22, 33)


def test_the_embargo_root_is_read_per_call_not_bound_at_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """**The property the whole fix rests on.**

    ``load_heldout_seeds`` carried ``EMBARGO_ROOT`` as a *keyword default*, evaluated once when the
    module was imported. Setting the variable afterwards — which is every deployment, since the
    environment is read after the process has started — reached nothing, and rebinding the module
    attribute reached nothing either, because the default already held the old value.
    """
    sealed = tmp_path / "late" / "elsewhere-v1"
    sealed.mkdir(parents=True)
    (sealed / "heldout_seeds.json").write_text(json.dumps({"seeds": [7]}))

    # Import has already happened — this module imported `load_heldout_seeds` at the top.
    monkeypatch.setenv(EMBARGO_ROOT_ENV, str(tmp_path / "late"))
    assert load_heldout_seeds("elsewhere-v1") == (7,)


def test_an_explicit_root_still_wins_over_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The keyword survives, so every existing caller and test keeps working unchanged."""
    sealed = tmp_path / "explicit" / "elsewhere-v1"
    sealed.mkdir(parents=True)
    (sealed / "heldout_seeds.json").write_text(json.dumps({"seeds": [1, 2]}))

    monkeypatch.setenv(EMBARGO_ROOT_ENV, str(tmp_path / "ignored"))
    assert load_heldout_seeds("elsewhere-v1", embargo_root=tmp_path / "explicit") == (1, 2)


def test_unset_keeps_the_repo_relative_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """A checkout-based run needs no configuration, exactly as before."""
    monkeypatch.delenv(EMBARGO_ROOT_ENV, raising=False)
    assert resolve_embargo_root() == EMBARGO_ROOT


def test_evaluate_scores_heldout_and_verifies(
    anchor: ScenarioSpec, heldout: tuple[int, ...]
) -> None:
    """Baseline scoring on the held-out seeds verifies, dispatching every seed to the sandbox."""
    sandbox = InProcessSandbox()
    card, integrity = evaluate(
        anchor, BASELINE_REF, seeds=heldout, scorer=SandboxScorer(sandbox)
    )
    assert integrity == "verified"
    assert {m.metric for m in card.metrics} == ANCHOR_METRICS
    assert all(tuple(m.seeds) == tuple(heldout) for m in card.metrics)
    # 12 held-out rollouts + a re-executed sample of sample_size=1.
    assert len(sandbox.invocations) == 13
    assert {inv.policy_ref for inv in sandbox.invocations} == {BASELINE_REF}


def test_evaluate_takes_a_reference_not_a_policy_object(
    anchor: ScenarioSpec, scorer: SandboxScorer
) -> None:
    """The scoring API is typed on a *string*, never a live Policy (bench#30)."""
    with pytest.raises((AttributeError, TypeError, ValueError)):
        scorer(anchor, BaselinePolicy(), seeds=(1,))  # type: ignore[arg-type]


def test_evaluate_flags_a_nondeterministic_policy(
    anchor: ScenarioSpec, scorer: SandboxScorer, heldout: tuple[int, ...]
) -> None:
    _, integrity = evaluate(anchor, NONDETERMINISTIC_REF, seeds=heldout, scorer=scorer)
    assert integrity == "flagged"


def test_build_submission_is_content_addressed(heldout_card: Scorecard) -> None:
    request = SubmissionRequest(scenario_id=ANCHOR_SCENARIO_ID, policy_ref=BASELINE_REF)
    first = build_submission(request, heldout_card, "verified")
    assert first.submission_id.startswith("sha256:")
    assert first == build_submission(request, heldout_card, "verified")  # deterministic
    assert len(first.scores) == 7
    assert first.scorecard_hash == heldout_card.content_hash
    assert first.source is None and first.provenance_hash is None


def test_build_submission_folds_the_source_into_the_id(heldout_card: Scorecard) -> None:
    """Two artifacts with different Hub identities never collide on a submission id."""
    request = SubmissionRequest(scenario_id=ANCHOR_SCENARIO_ID, policy_ref=BASELINE_REF)
    plain = build_submission(request, heldout_card, "verified")
    sourced = build_submission(
        request, heldout_card, "verified", source="sha256:" + "a" * 64, provenance_hash="sha256:p"
    )
    assert sourced.submission_id != plain.submission_id
    assert sourced.source == "sha256:" + "a" * 64
    assert sourced.provenance_hash == "sha256:p"


def _entry(
    submission_id: str,
    *,
    value: float | None = 1.0,
    direction: str = "higher_better",
) -> Any:
    from astro_mine.bench.leaderboard import MetricScore, Submission

    return Submission(
        submission_id=submission_id,
        scenario_id="s",
        policy_ref="m:p",
        method=None,
        author=None,
        scorecard_hash="sha256:" + "0" * 64,
        runner="fixture/0.1.0",
        integrity="verified",
        scores=(
            MetricScore(
                metric="water_mass",
                unit="kg",
                direction=direction,
                aggregation="mean",
                value=value,
                dispersion=None,
                n=1,
            ),
        ),
    )


def test_rank_empty() -> None:
    assert rank([]) == []


def test_rank_orders_higher_better_descending() -> None:
    entries = rank(
        [_entry("sha256:a"), _entry("sha256:b", value=3.0), _entry("sha256:c", value=2.0)]
    )
    assert [e.submission_id for e in entries] == ["sha256:b", "sha256:c", "sha256:a"]
    assert [e.rank for e in entries] == [1, 2, 3]
    assert entries[0].primary_metric == "water_mass"
    assert entries[0].primary_value == 3.0 and entries[0].primary_unit == "kg"


def test_rank_orders_lower_better_ascending() -> None:
    entries = rank(
        [
            _entry("sha256:a", value=3.0, direction="lower_better"),
            _entry("sha256:b", value=1.0, direction="lower_better"),
        ]
    )
    assert [e.submission_id for e in entries] == ["sha256:b", "sha256:a"]


def test_rank_puts_na_last_and_breaks_ties_by_id() -> None:
    entries = rank(
        [
            _entry("sha256:z", value=None),
            _entry("sha256:b", value=5.0),
            _entry("sha256:a", value=5.0),
        ]
    )
    assert [e.submission_id for e in entries] == ["sha256:a", "sha256:b", "sha256:z"]


def test_sample_reproduces_rejects_an_unknown_metric() -> None:
    """A metric present in the re-execution but absent from the full run cannot be verified."""
    from astro_mine.bench.metrics import AggregateScore
    from astro_mine.core.objective import MetricAggregation, MetricDirection

    def _card(metric: str) -> Scorecard:
        return Scorecard(
            scenario_id="s",
            runner="fixture/0.1.0",
            metrics=(
                AggregateScore(
                    metric=metric,
                    version="0.1.0",
                    unit="u",
                    direction=MetricDirection.HIGHER_BETTER,
                    aggregation=MetricAggregation.MEAN,
                    value=1.0,
                    dispersion=None,
                    n=1,
                    seeds=(1,),
                    per_seed=(1.0,),
                ),
            ),
        )

    assert _sample_reproduces(_card("water_mass"), _card("other_metric")) is False


# =================================================================================================
# _provenance.py — the lineage bundle + re-execution audit
# =================================================================================================


@pytest.fixture(scope="module")
def small_card(anchor: ScenarioSpec, scorer: SandboxScorer) -> Scorecard:
    """A cheap three-seed baseline card for the provenance unit tests."""
    return scorer(anchor, BASELINE_REF, seeds=(900001, 900002, 900003))


def test_provenance_bundle_carries_full_lineage(
    anchor: ScenarioSpec, small_card: Scorecard
) -> None:
    bundle = build_provenance_bundle(
        anchor, small_card, source="sha256:" + "a" * 64, source_digest="sha256:" + "a" * 64,
        code_version="0.0.0",
    )
    assert bundle.scenario_id == ANCHOR_SCENARIO_ID
    assert bundle.scenario_spec_hash == anchor.spec_hash
    assert bundle.core_interface_version == dict(anchor.core_interface)
    assert bundle.content_hashes == {ref.id: ref.content_hash for ref in anchor.content_refs()}
    assert bundle.environment_lockfile.startswith("sha256:")
    assert bundle.scorecard_hash == small_card.content_hash
    assert bundle.seeds == (900001, 900002, 900003)
    assert len(bundle.per_seed) == 3
    assert set(bundle.per_seed[0].metrics) == ANCHOR_METRICS
    assert bundle.bundle_hash.startswith("sha256:")


def test_provenance_bundle_hash_is_reproducible(
    anchor: ScenarioSpec, scorer: SandboxScorer, small_card: Scorecard
) -> None:
    """Two separately-scored identical runs produce the identical bundle hash (env excluded)."""
    second_card = scorer(anchor, BASELINE_REF, seeds=(900001, 900002, 900003))
    first = build_provenance_bundle(anchor, small_card, source="ref:x", code_version="0.0.0")
    second = build_provenance_bundle(anchor, second_card, source="ref:x", code_version="0.0.0")
    assert first.bundle_hash == second.bundle_hash


def test_provenance_bundle_hash_covers_the_source(
    anchor: ScenarioSpec, small_card: Scorecard
) -> None:
    a = build_provenance_bundle(anchor, small_card, source="ref:x", code_version="0.0.0")
    b = build_provenance_bundle(anchor, small_card, source="ref:y", code_version="0.0.0")
    assert a.bundle_hash != b.bundle_hash


def test_resample_from_bundle_verifies_a_deterministic_policy(
    anchor: ScenarioSpec, small_card: Scorecard
) -> None:
    """fraction=0.25 of 3 seeds re-executes ceil(0.75)=1 seed — the lowest-id one — and matches."""
    bundle = build_provenance_bundle(anchor, small_card, source="ref:x", code_version="0.0.0")
    sandbox = InProcessSandbox()
    verdict = resample_from_bundle(
        bundle, anchor, BASELINE_REF, scorer=SandboxScorer(sandbox), fraction=0.25
    )
    assert verdict == "verified"
    assert [inv.seed for inv in sandbox.invocations] == [900001]


def test_resample_from_bundle_flags_a_tampered_bundle(
    anchor: ScenarioSpec, small_card: Scorecard, scorer: SandboxScorer
) -> None:
    """A recorded per-seed value that does not reproduce flags the bundle, not just a warning."""
    bundle = build_provenance_bundle(anchor, small_card, source="ref:x", code_version="0.0.0")
    doctored = bundle.model_copy(
        update={
            "per_seed": tuple(
                record.model_copy(update={"metrics": dict.fromkeys(record.metrics, -999.0)})
                for record in bundle.per_seed
            )
        }
    )
    assert resample_from_bundle(doctored, anchor, BASELINE_REF, scorer=scorer) == "flagged"


# =================================================================================================
# LeaderboardService — authenticate / authorize / rate-limit / audit
# =================================================================================================


def test_no_idp_configured_refuses_writes_rather_than_falling_open(
    scorer: SandboxScorer,
) -> None:
    """"No IdP" must never mean "everyone is trusted" — the service fails closed with 503."""
    open_service = LeaderboardService(authn=None, scorer=scorer)
    with pytest.raises(SubmissionRejected, match="not configured") as caught:
        open_service.authenticate("Bearer whatever")
    assert caught.value.status == 503
    denials = open_service.audit.query(action="authenticate", decision=AuditDecision.DENY)
    assert len(denials) == 1
    assert "no OIDC verifier" in denials[0].reason


def test_authenticate_rejects_a_bad_token_with_401_and_audits(
    service: LeaderboardService, audit: InMemoryAuditLog
) -> None:
    with pytest.raises(SubmissionRejected) as caught:
        service.authenticate(None)
    assert caught.value.status == 401
    denials = audit.query(action="authenticate", decision=AuditDecision.DENY)
    assert len(denials) == 1
    assert "bearer token" in denials[0].reason


def test_authenticate_builds_the_principal_and_audits_the_allow(
    service: LeaderboardService, audit: InMemoryAuditLog, idp: TestIdp
) -> None:
    caller = service.authenticate(f"Bearer {idp.token(subject='lab-3')}")
    assert caller.subject == "lab-3"
    allowed = audit.query(action="authenticate", decision=AuditDecision.ALLOW)
    assert len(allowed) == 1
    assert allowed[0].subject == "lab-3" and allowed[0].issuer == idp.issuer


def test_authorize_raises_403_with_the_policy_reason(
    service: LeaderboardService, audit: InMemoryAuditLog
) -> None:
    with pytest.raises(SubmissionRejected) as caught:
        service.authorize(principal("lab-1", "submitter"), Action.RANKING_MUTATE, "sha256:x")
    assert caught.value.status == 403
    assert "do not grant" in str(caught.value)
    denials = audit.query(action=str(Action.RANKING_MUTATE), decision=AuditDecision.DENY)
    assert len(denials) == 1 and denials[0].resource == "sha256:x"


def test_authorize_allows_and_audits(
    service: LeaderboardService, audit: InMemoryAuditLog
) -> None:
    service.authorize(principal(), Action.SUBMISSION_CREATE, BASELINE_REF)
    allowed = audit.query(action=str(Action.SUBMISSION_CREATE), decision=AuditDecision.ALLOW)
    assert len(allowed) == 1
    assert allowed[0].subject == "lab-1"


def test_ticket_is_keyed_on_the_authenticated_subject(scorer: SandboxScorer) -> None:
    service = LeaderboardService(scorer=scorer)
    request = HubSubmissionRequest(scenario_id=ANCHOR_SCENARIO_ID, hub_ref="a/b:1.0.0")
    mine = service.ticket(request, principal("lab-1"))
    theirs = service.ticket(request, principal("lab-2"))
    assert mine != theirs  # nobody can collide with, or guess, another lab's job ticket
    assert mine == service.ticket(request, principal("lab-1"))  # deterministic
    assert mine.startswith("sha256:")


def test_get_job_is_none_for_an_unknown_ticket(service: LeaderboardService) -> None:
    assert service.get_job("sha256:" + "0" * 64) is None


def test_the_wire_models_forbid_a_client_supplied_identity() -> None:
    """bench#29: the request models carry no identity field at all (extra='forbid')."""
    assert "identity" not in HubSubmissionRequest.model_fields
    assert "identity" not in SubmissionRequest.model_fields
    with pytest.raises(Exception):  # noqa: B017 - pydantic ValidationError on extra="forbid"
        HubSubmissionRequest(
            scenario_id="s", hub_ref="a/b:1.0.0", identity="someone-elses-quota"
        )


# =================================================================================================
# LeaderboardService — the local policy_ref intake path
# =================================================================================================


def test_submit_local_scores_verifies_and_audits(
    anchor: ScenarioSpec, verifier: OidcTokenVerifier, idp: TestIdp, heldout: tuple[int, ...]
) -> None:
    """The P0 intake: authorize → rate-limit → shape-check → sandboxed evaluate → catalog."""
    audit = InMemoryAuditLog()
    sandbox = InProcessSandbox()
    service = LeaderboardService(authn=verifier, audit=audit, scorer=SandboxScorer(sandbox))
    caller = service.authenticate(f"Bearer {idp.token()}")

    request = SubmissionRequest(
        scenario_id=ANCHOR_SCENARIO_ID, policy_ref=BASELINE_REF, method="baseline"
    )
    submission = service.submit_local(anchor, request, caller)

    assert submission.integrity == "verified"
    assert submission.submission_id.startswith("sha256:")
    assert len(submission.scores) == 7
    assert service.store.get_submission(submission.submission_id) == submission
    # Every held-out seed was dispatched through the sandbox seam, as the reference string.
    assert {inv.seed for inv in sandbox.invocations} >= set(heldout)
    assert {inv.policy_ref for inv in sandbox.invocations} == {BASELINE_REF}
    verified = audit.query(action=str(Action.SUBMISSION_CREATE), decision=AuditDecision.VERIFIED)
    assert len(verified) == 1
    assert verified[0].submission_id == submission.submission_id
    assert "integrity=verified" in verified[0].reason


def test_submit_local_is_idempotent(
    anchor: ScenarioSpec, service: LeaderboardService, idp: TestIdp
) -> None:
    """Same policy, same scores ⇒ the same content-addressed entry, catalogued once."""
    caller = service.authenticate(f"Bearer {idp.token()}")
    request = SubmissionRequest(scenario_id=ANCHOR_SCENARIO_ID, policy_ref=BASELINE_REF)
    first = service.submit_local(anchor, request, caller)
    again = service.submit_local(anchor, request, caller)
    assert first.submission_id == again.submission_id
    assert len(service.store.list_submissions(ANCHOR_SCENARIO_ID)) == 1


def test_submit_local_rejects_a_bad_policy_ref_with_400(
    anchor: ScenarioSpec, service: LeaderboardService, audit: InMemoryAuditLog, idp: TestIdp
) -> None:
    caller = service.authenticate(f"Bearer {idp.token()}")
    request = SubmissionRequest(scenario_id=ANCHOR_SCENARIO_ID, policy_ref="not-a-ref")
    with pytest.raises(SubmissionRejected, match="module:attribute") as caught:
        service.submit_local(anchor, request, caller)
    assert caught.value.status == 400
    rejected = audit.query(action=str(Action.SUBMISSION_CREATE), decision=AuditDecision.REJECTED)
    assert len(rejected) == 1 and rejected[0].resource == "not-a-ref"


def test_submit_local_rejects_a_policy_that_will_not_run_with_422(
    anchor: ScenarioSpec, service: LeaderboardService, audit: InMemoryAuditLog, idp: TestIdp
) -> None:
    """A submission that raises inside the sandbox is rejected as data — never scored."""
    caller = service.authenticate(f"Bearer {idp.token()}")
    request = SubmissionRequest(scenario_id=ANCHOR_SCENARIO_ID, policy_ref=EXPLODING_REF)
    with pytest.raises(SubmissionRejected, match="did not execute cleanly") as caught:
        service.submit_local(anchor, request, caller)
    assert caught.value.status == 422
    executions = audit.query(action="submission:execute", decision=AuditDecision.REJECTED)
    assert len(executions) == 1
    assert "sandbox_status" in executions[0].detail
    assert service.store.list_submissions(ANCHOR_SCENARIO_ID) == []


def test_submit_local_denies_a_role_that_cannot_submit(
    anchor: ScenarioSpec, service: LeaderboardService, idp: TestIdp
) -> None:
    caller = service.authenticate(f"Bearer {idp.token(roles=('wizard',))}")
    request = SubmissionRequest(scenario_id=ANCHOR_SCENARIO_ID, policy_ref=BASELINE_REF)
    with pytest.raises(SubmissionRejected, match="no known role") as caught:
        service.submit_local(anchor, request, caller)
    assert caught.value.status == 403


def test_submit_local_rate_limits_the_authenticated_subject(
    anchor: ScenarioSpec, verifier: OidcTokenVerifier, idp: TestIdp, scorer: SandboxScorer
) -> None:
    """The counter is bound to the token's issuer#subject — not anything the client can edit."""
    limited = LeaderboardService(
        authn=verifier, rate_limiter=InMemoryRateLimiter(limit=1), scorer=scorer
    )
    caller = limited.authenticate(f"Bearer {idp.token(subject='lab-9')}")
    request = SubmissionRequest(scenario_id=ANCHOR_SCENARIO_ID, policy_ref=BASELINE_REF)
    limited.submit_local(anchor, request, caller)
    with pytest.raises(SubmissionRejected, match="exceeded 1 submissions") as caught:
        limited.submit_local(anchor, request, caller)
    assert caught.value.status == 429
    # A genuinely different authenticated subject gets its own window.
    other = limited.authenticate(f"Bearer {idp.token(subject='lab-10')}")
    assert limited.submit_local(anchor, request, other).integrity == "verified"


def test_submit_local_missing_heldout_seeds_is_404(
    anchor: ScenarioSpec,
    service: LeaderboardService,
    audit: InMemoryAuditLog,
    idp: TestIdp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import astro_mine.bench.leaderboard._service as service_module

    def _no_seeds(scenario_id: str) -> tuple[int, ...]:
        raise FileNotFoundError("no sealed seed set")

    monkeypatch.setattr(service_module, "load_heldout_seeds", _no_seeds)
    caller = service.authenticate(f"Bearer {idp.token()}")
    request = SubmissionRequest(scenario_id=ANCHOR_SCENARIO_ID, policy_ref=BASELINE_REF)
    with pytest.raises(SubmissionRejected, match="no sealed seed set") as caught:
        service.submit_local(anchor, request, caller)
    assert caught.value.status == 404
    disclosures = audit.query(action="embargo:disclose", decision=AuditDecision.REJECTED)
    assert len(disclosures) == 1


# =================================================================================================
# LeaderboardService — the Hub-digest intake path
# =================================================================================================


def _hub_service(
    registry: Registry, verifier: OidcTokenVerifier, **kwargs: Any
) -> tuple[LeaderboardService, InProcessSandbox, InMemoryAuditLog]:
    sandbox = InProcessSandbox()
    audit = InMemoryAuditLog()
    service = LeaderboardService(
        registry=registry, authn=verifier, audit=audit, scorer=SandboxScorer(sandbox), **kwargs
    )
    return service, sandbox, audit


def test_submit_hub_without_a_registry_is_a_runtime_error(
    anchor: ScenarioSpec, service: LeaderboardService
) -> None:
    request = HubSubmissionRequest(scenario_id=ANCHOR_SCENARIO_ID, hub_ref="x:1.0.0")
    with pytest.raises(RuntimeError, match="no Hub registry"):
        service.submit_hub(anchor, request, principal())


def test_hub_digest_intake_scores_ranks_and_bundles(
    registry: Registry, verifier: OidcTokenVerifier, anchor: ScenarioSpec, idp: TestIdp
) -> None:
    """The full pipeline: resolve → verify → validate → queue → sandbox → bundle → re-run → rank."""
    service, sandbox, audit = _hub_service(registry, verifier)
    digest = _publish(registry, interfaces=anchor.core_interface, attested=True)
    caller = service.authenticate(f"Bearer {idp.token()}")

    request = HubSubmissionRequest(
        scenario_id=ANCHOR_SCENARIO_ID, hub_ref=digest, method="acme-v1"
    )
    job = service.submit_hub(anchor, request, caller)

    assert job.status is SubmissionStatus.RANKED
    assert job.job_id == service.ticket(request, caller)
    assert service.get_job(job.job_id) == job
    assert job.result_id is not None

    submission = service.store.get_submission(job.result_id)
    assert submission is not None
    assert submission.integrity == "verified"
    assert submission.source == digest  # provenance is the Hub digest, not an upload
    assert submission.policy_ref == BASELINE_REF  # the manifest's entrypoint, read as a string
    assert submission.method == "acme-v1"
    assert len(submission.scores) == 7

    bundle = service.get_provenance(job.result_id)
    assert isinstance(bundle, ProvenanceBundle)
    assert bundle.scenario_spec_hash == anchor.spec_hash
    assert bundle.source == digest and bundle.source_digest == digest
    assert len(bundle.per_seed) == 12  # scored on every held-out seed

    # 12 held-out seeds scored + ceil(0.25 * 12) = 3 re-executed = 15 sandboxed rollouts.
    assert len(sandbox.invocations) == 15
    assert service.job_queue.depth() == 0  # the envelope was consumed by the eval worker

    verified = audit.query(action="submission:verify", decision=AuditDecision.VERIFIED)
    assert len(verified) == 1
    assert verified[0].detail["required"] == ["signature", "slsa", "sbom"]
    assert verified[0].detail["verified"] is True
    ranked = audit.query(action=str(Action.SUBMISSION_CREATE_HUB), decision=AuditDecision.VERIFIED)
    assert len(ranked) == 1 and ranked[0].job_id == job.job_id


def test_hub_intake_by_name_version_tag(
    registry: Registry, verifier: OidcTokenVerifier, anchor: ScenarioSpec, idp: TestIdp
) -> None:
    service, _, _ = _hub_service(registry, verifier)
    _publish(registry, name="lab/policy", version="2.1.0", interfaces=anchor.core_interface,
             attested=True)
    caller = service.authenticate(f"Bearer {idp.token()}")
    request = HubSubmissionRequest(scenario_id=ANCHOR_SCENARIO_ID, hub_ref="lab/policy:2.1.0")
    assert service.submit_hub(anchor, request, caller).status is SubmissionStatus.RANKED


def test_hub_intake_rejects_an_unattested_submission_before_it_executes(
    registry: Registry, verifier: OidcTokenVerifier, anchor: ScenarioSpec, idp: TestIdp
) -> None:
    """Verification failure fails closed: rejected, audited, and the policy never ran."""
    service, sandbox, audit = _hub_service(registry, verifier)
    digest = _publish(registry, interfaces=anchor.core_interface)  # published, never attested
    caller = service.authenticate(f"Bearer {idp.token()}")
    request = HubSubmissionRequest(scenario_id=ANCHOR_SCENARIO_ID, hub_ref=digest)

    with pytest.raises(SubmissionRejected, match="supply-chain verification") as caught:
        service.submit_hub(anchor, request, caller)
    assert caught.value.status == 422
    assert sandbox.invocations == []  # nothing reached the sandbox
    assert service.store.list_submissions(ANCHOR_SCENARIO_ID) == []
    job = service.get_job(service.ticket(request, caller))
    assert job is not None and job.status is SubmissionStatus.REJECTED
    rejected = audit.query(action="submission:verify", decision=AuditDecision.REJECTED)
    assert len(rejected) == 1
    assert rejected[0].detail["required"] == ["signature", "slsa", "sbom"]


def test_hub_intake_rejects_an_unresolvable_digest_with_404(
    registry: Registry, verifier: OidcTokenVerifier, anchor: ScenarioSpec, idp: TestIdp
) -> None:
    service, _, audit = _hub_service(registry, verifier)
    caller = service.authenticate(f"Bearer {idp.token()}")
    request = HubSubmissionRequest(scenario_id=ANCHOR_SCENARIO_ID, hub_ref="sha256:" + "0" * 64)
    with pytest.raises(SubmissionRejected, match="cannot resolve") as caught:
        service.submit_hub(anchor, request, caller)
    assert caught.value.status == 404
    job = service.get_job(service.ticket(request, caller))
    assert job is not None and job.status is SubmissionStatus.REJECTED
    assert audit.query(action="submission:resolve", decision=AuditDecision.REJECTED)


def test_hub_intake_rejects_an_interface_mismatch_with_422(
    registry: Registry, verifier: OidcTokenVerifier, anchor: ScenarioSpec, idp: TestIdp
) -> None:
    service, _, audit = _hub_service(registry, verifier)
    bad = dict(anchor.core_interface)
    bad[next(iter(bad))] = "9.0.0"
    digest = _publish(registry, name="acme/future", interfaces=bad, attested=True)
    caller = service.authenticate(f"Bearer {idp.token()}")
    request = HubSubmissionRequest(scenario_id=ANCHOR_SCENARIO_ID, hub_ref=digest)
    with pytest.raises(SubmissionRejected, match="not satisfy") as caught:
        service.submit_hub(anchor, request, caller)
    assert caught.value.status == 422
    assert audit.query(action="submission:validate", decision=AuditDecision.REJECTED)


def test_hub_intake_rejects_a_manifest_without_an_entrypoint_with_422(
    registry: Registry, verifier: OidcTokenVerifier, anchor: ScenarioSpec, idp: TestIdp
) -> None:
    service, _, audit = _hub_service(registry, verifier)
    digest = _publish(
        registry, name="acme/no-entry", interfaces=anchor.core_interface,
        entrypoint=None, attested=True,
    )
    caller = service.authenticate(f"Bearer {idp.token()}")
    request = HubSubmissionRequest(scenario_id=ANCHOR_SCENARIO_ID, hub_ref=digest)
    with pytest.raises(SubmissionRejected, match="no 'entrypoint'") as caught:
        service.submit_hub(anchor, request, caller)
    assert caught.value.status == 422
    assert audit.query(action="submission:materialize", decision=AuditDecision.REJECTED)


def test_hub_intake_rate_limit_marks_the_job_rejected(
    registry: Registry, verifier: OidcTokenVerifier, anchor: ScenarioSpec, idp: TestIdp
) -> None:
    service, _, _ = _hub_service(registry, verifier, rate_limiter=InMemoryRateLimiter(limit=1))
    _publish(registry, name="acme/a", interfaces=anchor.core_interface, attested=True)
    _publish(registry, name="acme/b", interfaces=anchor.core_interface, attested=True)
    caller = service.authenticate(f"Bearer {idp.token(subject='lab-1')}")

    first = HubSubmissionRequest(scenario_id=ANCHOR_SCENARIO_ID, hub_ref="acme/a:1.0.0")
    assert service.submit_hub(anchor, first, caller).status is SubmissionStatus.RANKED

    second = HubSubmissionRequest(scenario_id=ANCHOR_SCENARIO_ID, hub_ref="acme/b:1.0.0")
    with pytest.raises(SubmissionRejected, match="exceeded 1 submissions") as caught:
        service.submit_hub(anchor, second, caller)
    assert caught.value.status == 429
    job = service.get_job(service.ticket(second, caller))
    assert job is not None and job.status is SubmissionStatus.REJECTED


def test_hub_intake_flags_a_nondeterministic_submission(
    registry: Registry, verifier: OidcTokenVerifier, anchor: ScenarioSpec, idp: TestIdp
) -> None:
    """A submission whose re-execution mismatches is catalogued but flagged, not ranked."""
    service, _, _ = _hub_service(registry, verifier)
    digest = _publish(
        registry, name="acme/flaky", interfaces=anchor.core_interface,
        entrypoint=NONDETERMINISTIC_REF, attested=True,
    )
    caller = service.authenticate(f"Bearer {idp.token()}")
    request = HubSubmissionRequest(scenario_id=ANCHOR_SCENARIO_ID, hub_ref=digest)
    job = service.submit_hub(anchor, request, caller)
    assert job.status is SubmissionStatus.FLAGGED
    assert job.detail == "provenance re-execution mismatch"
    submission = service.store.get_submission(job.result_id)
    assert submission is not None and submission.integrity == "flagged"


def test_hub_intake_rejects_a_submission_that_explodes_in_the_sandbox(
    registry: Registry, verifier: OidcTokenVerifier, anchor: ScenarioSpec, idp: TestIdp
) -> None:
    service, _, audit = _hub_service(registry, verifier)
    digest = _publish(
        registry, name="acme/boom", interfaces=anchor.core_interface,
        entrypoint=EXPLODING_REF, attested=True,
    )
    caller = service.authenticate(f"Bearer {idp.token()}")
    request = HubSubmissionRequest(scenario_id=ANCHOR_SCENARIO_ID, hub_ref=digest)
    with pytest.raises(SubmissionRejected, match="did not execute cleanly") as caught:
        service.submit_hub(anchor, request, caller)
    assert caught.value.status == 422
    job = service.get_job(service.ticket(request, caller))
    assert job is not None and job.status is SubmissionStatus.REJECTED
    assert audit.query(action="submission:execute", decision=AuditDecision.REJECTED)


# =================================================================================================
# LeaderboardService — retract, replay attachment, provenance reads
# =================================================================================================


def _catalogued(service: LeaderboardService, heldout_card: Scorecard) -> Any:
    """Insert a pre-scored submission into the service's store without re-scoring."""
    request = SubmissionRequest(scenario_id=ANCHOR_SCENARIO_ID, policy_ref=BASELINE_REF)
    submission = build_submission(request, heldout_card, "verified")
    service.store.add_submission(submission)
    return submission


def test_retract_is_admin_only_and_audited(
    service: LeaderboardService, audit: InMemoryAuditLog, heldout_card: Scorecard
) -> None:
    submission = _catalogued(service, heldout_card)

    with pytest.raises(SubmissionRejected) as caught:
        service.retract(submission.submission_id, principal("lab-1", "submitter"))
    assert caught.value.status == 403
    assert service.store.get_submission(submission.submission_id) is not None  # still on the board

    removed = service.retract(submission.submission_id, principal("boss", "admin"))
    assert removed == submission
    assert service.store.get_submission(submission.submission_id) is None
    retractions = audit.query(action="submission:retract", decision=AuditDecision.ALLOW)
    assert len(retractions) == 1
    assert retractions[0].submission_id == submission.submission_id


def test_retract_unknown_submission_is_a_key_error(service: LeaderboardService) -> None:
    with pytest.raises(KeyError, match="no submission"):
        service.retract("sha256:" + "0" * 64, principal("boss", "admin"))


def test_attach_replay_content_addresses_the_mcap(
    service: LeaderboardService, heldout_card: Scorecard
) -> None:
    """The replay digest lands on trace_hash without changing the submission id."""
    submission = _catalogued(service, heldout_card)
    assert service.get_replay(submission.submission_id) is None  # no replay attached yet

    mcap = b"mcap-episode-bytes"
    updated = service.attach_replay(submission.submission_id, mcap)
    assert updated.submission_id == submission.submission_id
    assert updated.trace_hash == blob_digest(mcap)
    assert service.store.get_submission(submission.submission_id) == updated
    assert service.get_replay(submission.submission_id) == mcap


def test_attach_replay_unknown_submission_is_a_key_error(service: LeaderboardService) -> None:
    with pytest.raises(KeyError, match="no submission"):
        service.attach_replay("sha256:" + "0" * 64, b"bytes")


def test_get_replay_and_get_provenance_are_none_for_unknown_ids(
    service: LeaderboardService, heldout_card: Scorecard
) -> None:
    assert service.get_replay("sha256:" + "0" * 64) is None
    assert service.get_provenance("sha256:" + "0" * 64) is None
    # A local-tier submission has no provenance bundle either.
    submission = _catalogued(service, heldout_card)
    assert service.get_provenance(submission.submission_id) is None
