"""The hosted leaderboard service — authenticated, authorized, verified, sandboxed, traced.

:class:`LeaderboardService` is the hosted tier's worker. It composes the dependency-clean building
blocks — the submission catalog (:mod:`._store`), the object store (:mod:`._objects`), the job
lifecycle and queue (:mod:`._jobs`), Hub-digest intake (:mod:`._hub`), the provenance bundle
(:mod:`._provenance`), authentication (:mod:`._auth`), policy (:mod:`._authz`), supply-chain
verification (:mod:`._supply_chain`), the audit trail (:mod:`._audit`), the execution sandbox
(:mod:`astro_mine.bench.sandbox`), and telemetry (:mod:`astro_mine.bench.telemetry`) — into the
pipeline the issue's exit criterion names: *an external lab publishes a policy to Hub and beats a
baseline on the public leaderboard, reproducibly*.

The pipeline for one submission (bench.md §3, §6, §9), on **both** intake paths:

1. **authenticate** the caller's OIDC bearer token — no token, no submission (bench#29);
2. **rate-limit + authorize** against the *authenticated subject*: RBAC, the per-role submission
   quota, and embargo control (bench#29);
3. **resolve** the artifact from Hub by digest, fail-closed on content-address integrity;
4. **verify its attestations** — cosign signature + SLSA provenance + SBOM — *before it executes*,
   reusing Seal's primitives through Hub (bench#29);
5. **validate** its Core plugin manifest against the scenario's interface;
6. **queue** it (the async hop; the trace context rides the envelope's headers — bench#32);
7. **disclose** the embargoed held-out seeds and run **submit-policy-we-run** — *in a sandbox*:
   out-of-process, no network egress, hard CPU/memory/time caps (bench#30). The evaluator never
   imports the submission; it hands the sandbox a reference and reads a result document back;
8. **bundle** the full lineage and store it, then **re-execute a sampled fraction from that bundle**
   — through the same sandbox — for the integrity verdict (``verified`` ⇒ ``ranked``; mismatch ⇒
   ``flagged``);
9. **catalog** the content-addressed entry, **audit** every decision, and record the Prometheus
   signals bench.md §10 puts on the dashboard.

Every backend defaults to a process-local implementation, so the whole pipeline runs offline in the
tests and the local tier; a deployment injects the Postgres / Redis / S3 / Hub / OIDC / OPA backends
behind the same protocols — the hosted service is a *deployment of this code*, not a second path
(bench.md §2.6).

Backlog: RM-P1-BENCH-10 — https://github.com/astro-mine/astro-mine-bench/issues/18;
bench#29, bench#30, bench#32
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NoReturn

from astro_mine.bench._version import __version__
from astro_mine.bench.leaderboard._audit import (
    AuditDecision,
    AuditLog,
    InMemoryAuditLog,
    audit_event,
)
from astro_mine.bench.leaderboard._auth import (
    AuthenticationError,
    Principal,
    TokenVerifier,
    bearer_token,
)
from astro_mine.bench.leaderboard._authz import (
    Action,
    AuthorizationError,
    AuthorizationRequest,
    PolicyEngine,
    RbacPolicyEngine,
)
from astro_mine.bench.leaderboard._eval import (
    build_submission,
    evaluate,
    load_heldout_seeds,
    validate_policy_ref,
)
from astro_mine.bench.leaderboard._hub import (
    HubRegistry,
    HubResolutionError,
    ManifestInterfaceError,
    ResolvedSubmission,
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
    Submission,
    SubmissionRequest,
)
from astro_mine.bench.leaderboard._objects import InMemoryObjectStore, ObjectStore
from astro_mine.bench.leaderboard._provenance import (
    ProvenanceBundle,
    build_provenance_bundle,
    resample_from_bundle,
)
from astro_mine.bench.leaderboard._store import InMemoryStore, LeaderboardStore
from astro_mine.bench.leaderboard._supply_chain import (
    AttestationPolicy,
    SupplyChainRejected,
    verify_submission_attestations,
)
from astro_mine.bench.sandbox import (
    PolicyScorer,
    Sandbox,
    SandboxError,
    SandboxScorer,
    SubprocessSandbox,
)
from astro_mine.bench.scenario import ScenarioSpec
from astro_mine.bench.scenario._hash import content_hash
from astro_mine.bench.telemetry import current_trace_id, extract_trace_context, span
from astro_mine.bench.telemetry import inject_trace_context as _inject
from astro_mine.bench.telemetry import metrics as _metrics

__all__ = ["LeaderboardService", "SubmissionRejected"]


class SubmissionRejected(Exception):
    """A submission that never entered the board — the terminal, audited refusal.

    ``status`` is the HTTP status the public edge maps it to: 401 unauthenticated, 403 denied by
    policy, 404 unresolved digest, 422 invalid manifest / failed attestation / failed sandboxed
    execution, 429 rate-limited.
    """

    def __init__(self, message: str, *, status: int) -> None:
        super().__init__(message)
        self.status = status


class LeaderboardService:
    """The hosted leaderboard worker — authenticated, authorized, verified, sandboxed intake.

    Every backend defaults to its process-local implementation; a deployment injects the durable
    ones (``SqlStore`` on Postgres, a Redis job store + rate limiter + queue, an S3 object store, a
    Hub ``Registry``, an OIDC verifier, an OPA sidecar) behind the same protocols.

    ``authn`` is the OIDC token verifier — **required for every write**: with none configured the
    service can authenticate nobody, so it refuses writes (503) rather than falling open (bench#29).
    ``scorer`` is the execution seam: it defaults to a
    :class:`~astro_mine.bench.sandbox.SandboxScorer`
    over a :class:`~astro_mine.bench.sandbox.SubprocessSandbox`, so submitted policies run
    out-of-process with no network egress and hard resource caps (bench#30). ``registry`` is needed
    for Hub-digest intake.
    """

    def __init__(
        self,
        *,
        store: LeaderboardStore | None = None,
        object_store: ObjectStore | None = None,
        job_store: JobStore | None = None,
        job_queue: JobQueue | None = None,
        rate_limiter: RateLimiter | None = None,
        registry: HubRegistry | None = None,
        authn: TokenVerifier | None = None,
        policy_engine: PolicyEngine | None = None,
        audit: AuditLog | None = None,
        attestation_policy: AttestationPolicy | None = None,
        sandbox: Sandbox | None = None,
        scorer: PolicyScorer | None = None,
        resample_fraction: float = 0.25,
    ) -> None:
        self.store: LeaderboardStore = store if store is not None else InMemoryStore()
        self.object_store: ObjectStore = (
            object_store if object_store is not None else InMemoryObjectStore()
        )
        self.job_store: JobStore = job_store if job_store is not None else InMemoryJobStore()
        self.job_queue: JobQueue = job_queue if job_queue is not None else InMemoryJobQueue()
        self.rate_limiter: RateLimiter = (
            rate_limiter if rate_limiter is not None else InMemoryRateLimiter()
        )
        self.registry = registry
        self.authn = authn
        self.policy_engine: PolicyEngine = (
            policy_engine if policy_engine is not None else RbacPolicyEngine()
        )
        self.audit: AuditLog = audit if audit is not None else InMemoryAuditLog()
        self.attestation_policy: AttestationPolicy = (
            attestation_policy if attestation_policy is not None else AttestationPolicy()
        )
        # The execution boundary. A caller may inject a whole scorer (a container-backed one, or a
        # fast trusted one for a private deployment); by default a submission is rolled seed-by-seed
        # in a capped, network-less subprocess.
        self.scorer: PolicyScorer = (
            scorer
            if scorer is not None
            else SandboxScorer(sandbox if sandbox is not None else SubprocessSandbox())
        )
        self.resample_fraction = resample_fraction

    # -- authentication + authorization (bench#29) -------------------------------------------------

    def authenticate(self, authorization: str | None) -> Principal:
        """Authenticate an ``Authorization: Bearer <jwt>`` header; raise 401/503 on failure.

        With no verifier configured the deployment cannot authenticate *anyone*, so it refuses the
        write outright (503) — it never treats "no IdP" as "everyone is trusted".
        """
        if self.authn is None:
            self._record(
                action="authenticate",
                decision=AuditDecision.DENY,
                reason="no OIDC verifier is configured on this deployment",
            )
            raise SubmissionRejected(
                "authentication is not configured on this deployment; submissions are refused. "
                "Set ASTRO_MINE_BENCH_OIDC_ISSUER and ASTRO_MINE_BENCH_OIDC_AUDIENCE.",
                status=503,
            )
        try:
            principal = self.authn.verify(bearer_token(authorization))
        except AuthenticationError as exc:
            self._record(action="authenticate", decision=AuditDecision.DENY, reason=str(exc))
            raise SubmissionRejected(str(exc), status=401) from exc
        self._record(
            action="authenticate",
            decision=AuditDecision.ALLOW,
            principal=principal,
            reason="bearer token verified",
        )
        return principal

    def authorize(
        self,
        principal: Principal,
        action: Action,
        resource: str,
        *,
        scenario_id: str | None = None,
    ) -> None:
        """Evaluate the policy engine for ``action`` and audit the decision; raise 403 on a denial.

        The quota context is the count of submissions this **authenticated subject** has made in the
        current window — not a client-supplied identity, which a submitter could rotate at will.
        """
        context = {
            "submissions_in_window": self.rate_limiter.observed(principal.identity),
            "scenario_id": scenario_id or resource,
        }
        request = AuthorizationRequest(
            principal=principal, action=action, resource=resource, context=context
        )
        decision = self.policy_engine.evaluate(request)
        _metrics().authz_decisions.labels(
            action=str(action), decision="allow" if decision.allow else "deny"
        ).inc()
        self._record(
            action=str(action),
            decision=AuditDecision.ALLOW if decision.allow else AuditDecision.DENY,
            principal=principal,
            resource=resource,
            reason=decision.reason,
        )
        if not decision.allow:
            raise SubmissionRejected(
                decision.reason or f"{action} denied", status=403
            ) from AuthorizationError(decision, action=action, resource=resource)

    def _rate_limit(self, principal: Principal, resource: str) -> None:
        """Count the submission against the authenticated subject's window; 429 over the limit."""
        try:
            self.rate_limiter.check(principal.identity)
        except RateLimitError as exc:
            self._record(
                action="rate_limit",
                decision=AuditDecision.DENY,
                principal=principal,
                resource=resource,
                reason=str(exc),
            )
            raise SubmissionRejected(str(exc), status=429) from exc

    def _record(
        self,
        *,
        action: str,
        decision: AuditDecision,
        principal: Principal | None = None,
        resource: str = "",
        reason: str = "",
        submission_id: str | None = None,
        job_id: str | None = None,
        detail: dict[str, object] | None = None,
    ) -> None:
        """Append one decision to the audit trail, stamped with the active OTel trace id."""
        self.audit.record(
            audit_event(
                action=action,
                decision=decision,
                subject=None if principal is None else principal.subject,
                issuer=None if principal is None else principal.issuer,
                resource=resource,
                reason=reason,
                submission_id=submission_id,
                job_id=job_id,
                trace_id=current_trace_id(),
                detail=dict(detail or {}),
            )
        )

    # -- lifecycle helpers ------------------------------------------------------------------------

    def ticket(self, request: HubSubmissionRequest, principal: Principal) -> str:
        """The deterministic job handle for ``request`` — the ticket the client polls.

        Keyed on the **authenticated** subject (bench#29): two different labs submitting the same
        artifact get two different tickets, and no one can guess or collide with another's job by
        replaying their ``identity``.
        """
        return content_hash(
            {
                "hub_ref": request.hub_ref,
                "scenario_id": request.scenario_id,
                "subject": principal.identity,
            }
        )

    def _transition(
        self,
        job_id: str,
        status: SubmissionStatus,
        *,
        detail: str | None = None,
        result_id: str | None = None,
    ) -> JobRecord:
        record = JobRecord(job_id=job_id, status=status, detail=detail, result_id=result_id)
        self.job_store.put_job(record)
        return record

    def get_job(self, job_id: str) -> JobRecord | None:
        """The current lifecycle record for ``job_id`` (the ticket), or ``None`` if unknown."""
        return self.job_store.get_job(job_id)

    def get_provenance(self, submission_id: str) -> ProvenanceBundle | None:
        """The stored :class:`ProvenanceBundle` behind ``submission_id`` (from the object store)."""
        submission = self.store.get_submission(submission_id)
        if submission is None or submission.provenance_hash is None:
            return None
        raw = self.object_store.get(submission.provenance_hash)
        return None if raw is None else ProvenanceBundle.model_validate_json(raw)

    # -- View replay handoff (RM-P1-BENCH-12) -----------------------------------------------------

    def attach_replay(self, submission_id: str, mcap: bytes) -> Submission:
        """Store an MCAP episode replay for a scored submission and record its digest (bench.md §6).

        The seam a producer (the Cloud eval path, or a local Sim run) calls to attach the episode
        recording behind a leaderboard entry, so View can fetch and render it. The bytes are
        content-addressed in the object store; the submission's ``trace_hash`` is set to that digest
        — which does **not** change its ``submission_id`` — and the entry is re-catalogued. Raises
        :class:`KeyError` if the submission is unknown.
        """
        submission = self.store.get_submission(submission_id)
        if submission is None:
            raise KeyError(f"no submission {submission_id!r}")
        trace_hash = self.object_store.put(mcap)
        updated = submission.model_copy(update={"trace_hash": trace_hash})
        self.store.add_submission(updated)
        return updated

    def get_replay(self, submission_id: str) -> bytes | None:
        """The stored MCAP replay bytes for ``submission_id`` (fail-closed verified), or ``None``.

        ``None`` when the submission is unknown or has no replay attached — View renders a replay
        only when Bench has one to serve (bench.md §6).
        """
        submission = self.store.get_submission(submission_id)
        if submission is None or submission.trace_hash is None:
            return None
        return self.object_store.get(submission.trace_hash)

    # -- board administration (bench#29) ----------------------------------------------------------

    def retract(self, submission_id: str, principal: Principal) -> Submission:
        """Remove an entry from the board — a ranking mutation, admin-only and audit-logged.

        The escape hatch for a submission found to be fraudulent, mis-attributed, or subject to a
        takedown. It is exactly the kind of privileged, board-altering act bench#29 requires to sit
        behind authN/Z rather than an open port.
        """
        submission = self.store.get_submission(submission_id)
        if submission is None:
            raise KeyError(f"no submission {submission_id!r}")
        self.authorize(
            principal,
            Action.RANKING_MUTATE,
            submission_id,
            scenario_id=submission.scenario_id,
        )
        self.store.remove_submission(submission_id)
        self._record(
            action="submission:retract",
            decision=AuditDecision.ALLOW,
            principal=principal,
            resource=submission_id,
            reason="entry retracted from the board",
            submission_id=submission_id,
        )
        return submission

    # -- the local policy_ref intake path (RM-P0-BENCH-06) ----------------------------------------

    def submit_local(
        self, spec: ScenarioSpec, request: SubmissionRequest, principal: Principal
    ) -> Submission:
        """Run an importable ``policy_ref`` on the held-out seeds — authenticated and sandboxed.

        The P0 intake path, brought up to the bench.md §9 posture: the reference is *shape-checked*
        here and **imported only inside the sandbox** (bench#30), and the caller must be an
        authenticated principal whose role and quota allow the submission (bench#29).
        """
        with span(
            "bench.submit",
            **{"bench.scenario_id": spec.scenario_id, "bench.intake": "policy_ref"},
        ):
            self.authorize(
                principal,
                Action.SUBMISSION_CREATE,
                request.policy_ref,
                scenario_id=spec.scenario_id,
            )
            self._rate_limit(principal, request.policy_ref)
            try:
                policy_ref = validate_policy_ref(request.policy_ref)
            except ValueError as exc:
                self._record(
                    action=str(Action.SUBMISSION_CREATE),
                    decision=AuditDecision.REJECTED,
                    principal=principal,
                    resource=request.policy_ref,
                    reason=str(exc),
                )
                raise SubmissionRejected(str(exc), status=400) from exc

            seeds = self._heldout(spec, principal, request.policy_ref)

            with (
                span("bench.evaluate", **{"bench.scenario_id": spec.scenario_id}),
                _metrics().time_stage(scenario=spec.scenario_id, stage="evaluate"),
            ):
                try:
                    card, integrity = evaluate(spec, policy_ref, seeds=seeds, scorer=self.scorer)
                except SandboxError as exc:
                    self._reject_execution(principal, request.policy_ref, exc)

            with span("bench.score", **{"bench.scenario_id": spec.scenario_id}):
                submission = build_submission(request, card, integrity)
                self.store.add_submission(submission)

            _metrics().submissions.labels(
                scenario=spec.scenario_id,
                outcome="ranked" if integrity == "verified" else "flagged",
            ).inc()
            self._record(
                action=str(Action.SUBMISSION_CREATE),
                decision=AuditDecision.VERIFIED
                if integrity == "verified"
                else AuditDecision.REJECTED,
                principal=principal,
                resource=request.policy_ref,
                reason=f"scored with integrity={integrity}",
                submission_id=submission.submission_id,
            )
            return submission

    # -- the Hub-digest intake path (RM-P1-BENCH-10) ----------------------------------------------

    def submit_hub(
        self, spec: ScenarioSpec, request: HubSubmissionRequest, principal: Principal
    ) -> JobRecord:
        """Run the full Hub-digest intake pipeline and return the terminal job record.

        Raises :class:`SubmissionRejected` for a submission that never entered the board (denied by
        policy, rate-refused, unresolved/failed-verify digest, **failed attestation**, an
        interface-incompatible manifest, or a submission that would not execute cleanly inside its
        sandbox); the job record is left in ``rejected`` with the reason, and every step is
        audit-logged. A submission that ran is ``ranked`` (integrity verified) or ``flagged``
        (provenance re-execution mismatch) — both catalogued.
        """
        if self.registry is None:
            raise RuntimeError("LeaderboardService has no Hub registry; digest intake unavailable")

        with span(
            "bench.submit",
            **{
                "bench.scenario_id": spec.scenario_id,
                "bench.intake": "hub_digest",
                "bench.hub_ref": request.hub_ref,
            },
        ):
            job_id = self.ticket(request, principal)
            self.authorize(
                principal,
                Action.SUBMISSION_CREATE_HUB,
                request.hub_ref,
                scenario_id=spec.scenario_id,
            )
            try:
                self._rate_limit(principal, request.hub_ref)
            except SubmissionRejected as exc:
                self._transition(job_id, SubmissionStatus.REJECTED, detail=str(exc))
                raise

            resolved = self._resolve(self.registry, job_id, principal, request)
            self._verify_attestations(job_id, principal, request, resolved.manifest_digest)
            self._validate_interface(job_id, principal, request, resolved, spec)
            policy_ref = self._policy_ref(job_id, principal, request, resolved)

            # The async hop: the submission is enqueued and picked up by the evaluation worker. The
            # trace context rides the envelope headers, so submit→evaluate→score→rank is one trace
            # even though a worker, not this handler, does the scoring (bench#32).
            envelope = SubmissionEnvelope(
                job_id=job_id,
                scenario_id=spec.scenario_id,
                reference=request.hub_ref,
                subject=principal.identity,
                headers=dict(_inject({})),
            )
            self.job_queue.publish(envelope)
            self._transition(job_id, SubmissionStatus.QUEUED)
            _metrics().queue_depth.set(self.job_queue.depth())

            return self._evaluate_queued(
                spec, request, principal, resolved.manifest_digest, policy_ref
            )

    def _evaluate_queued(
        self,
        spec: ScenarioSpec,
        request: HubSubmissionRequest,
        principal: Principal,
        source_digest: str,
        policy_ref: str,
    ) -> JobRecord:
        """The evaluation worker's half of the pipeline: dequeue, score, verify, rank."""
        envelope = self.job_queue.consume()
        _metrics().queue_depth.set(self.job_queue.depth())
        if envelope is None:  # pragma: no cover - the producer above always enqueues one
            raise RuntimeError("the submission queue lost an accepted submission")
        job_id = envelope.job_id

        # Re-attach the producer's trace context across the queue hop (bench#32).
        with extract_trace_context(envelope.headers):
            self._transition(job_id, SubmissionStatus.RUNNING)
            seeds = self._heldout(spec, principal, request.hub_ref, job_id=job_id)

            with (
                span(
                    "bench.evaluate",
                    **{"bench.scenario_id": spec.scenario_id, "bench.job_id": job_id},
                ),
                _metrics().time_stage(scenario=spec.scenario_id, stage="evaluate"),
            ):
                try:
                    # Submit-policy-we-run on the embargoed held-out seeds (disclosed only now) —
                    # executed in the sandbox, never in this process (bench#30).
                    card = self.scorer(spec, policy_ref, seeds=seeds)
                except SandboxError as exc:
                    self._transition(job_id, SubmissionStatus.REJECTED, detail=str(exc))
                    self._reject_execution(principal, request.hub_ref, exc, job_id=job_id)
            self._transition(job_id, SubmissionStatus.SCORED)

            with span("bench.score", **{"bench.job_id": job_id}):
                bundle = build_provenance_bundle(
                    spec,
                    card,
                    source=source_digest,
                    source_digest=source_digest,
                    code_version=__version__,
                )
                provenance_hash = self.object_store.put(bundle.model_dump_json().encode("utf-8"))

            # Determinism enforcement: re-execute a sampled fraction from the stored bundle, through
            # the same sandbox, because the audit re-runs the same untrusted code (bench.md §9).
            with (
                span("bench.reexecute", **{"bench.job_id": job_id}),
                _metrics().time_stage(scenario=spec.scenario_id, stage="reexecute"),
            ):
                try:
                    integrity = resample_from_bundle(
                        bundle,
                        spec,
                        policy_ref,
                        scorer=self.scorer,
                        fraction=self.resample_fraction,
                    )
                except SandboxError as exc:
                    self._transition(job_id, SubmissionStatus.REJECTED, detail=str(exc))
                    self._reject_execution(principal, request.hub_ref, exc, job_id=job_id)
            _metrics().reexecutions.labels(
                scenario=spec.scenario_id,
                verdict="verified" if integrity == "verified" else "mismatch",
            ).inc()

            with span("bench.rank", **{"bench.job_id": job_id}):
                submission = build_submission(
                    SubmissionRequest(
                        scenario_id=spec.scenario_id,
                        policy_ref=policy_ref,
                        method=request.method,
                        author=request.author,
                    ),
                    card,
                    integrity,
                    source=source_digest,
                    provenance_hash=provenance_hash,
                )
                self.store.add_submission(submission)

            _metrics().submissions.labels(
                scenario=spec.scenario_id,
                outcome="ranked" if integrity == "verified" else "flagged",
            ).inc()
            self._record(
                action=str(Action.SUBMISSION_CREATE_HUB),
                decision=AuditDecision.VERIFIED
                if integrity == "verified"
                else AuditDecision.REJECTED,
                principal=principal,
                resource=request.hub_ref,
                reason=f"scored with integrity={integrity}",
                submission_id=submission.submission_id,
                job_id=job_id,
            )

            if integrity == "verified":
                return self._transition(
                    job_id, SubmissionStatus.RANKED, result_id=submission.submission_id
                )
            return self._transition(
                job_id,
                SubmissionStatus.FLAGGED,
                detail="provenance re-execution mismatch",
                result_id=submission.submission_id,
            )

    # -- intake steps -----------------------------------------------------------------------------

    def _heldout(
        self,
        spec: ScenarioSpec,
        principal: Principal,
        resource: str,
        *,
        job_id: str | None = None,
    ) -> Sequence[int]:
        """Disclose the embargoed held-out seed set at evaluation time (bench.md §9)."""
        try:
            return load_heldout_seeds(spec.scenario_id)
        except FileNotFoundError as exc:
            if job_id is not None:
                self._transition(job_id, SubmissionStatus.REJECTED, detail=str(exc))
            self._record(
                action="embargo:disclose",
                decision=AuditDecision.REJECTED,
                principal=principal,
                resource=resource,
                reason=str(exc),
                job_id=job_id,
            )
            raise SubmissionRejected(str(exc), status=404) from exc

    def _resolve(
        self,
        registry: HubRegistry,
        job_id: str,
        principal: Principal,
        request: HubSubmissionRequest,
    ) -> ResolvedSubmission:
        """Resolve + content-address-verify the artifact from Hub, fail-closed."""
        with span("bench.resolve", **{"bench.hub_ref": request.hub_ref}):
            try:
                return resolve_submission(registry, request.hub_ref)
            except HubResolutionError as exc:
                self._transition(job_id, SubmissionStatus.REJECTED, detail=str(exc))
                self._record(
                    action="submission:resolve",
                    decision=AuditDecision.REJECTED,
                    principal=principal,
                    resource=request.hub_ref,
                    reason=str(exc),
                    job_id=job_id,
                )
                raise SubmissionRejected(str(exc), status=404) from exc

    def _verify_attestations(
        self, job_id: str, principal: Principal, request: HubSubmissionRequest, digest: str
    ) -> None:
        """Verify cosign signature + SLSA provenance + SBOM **before the submission executes**."""
        with span("bench.verify", **{"bench.hub_ref": request.hub_ref}):
            try:
                verdict = verify_submission_attestations(
                    self.registry, digest, self.attestation_policy
                )
            except SupplyChainRejected as exc:
                _metrics().verifications.labels(outcome="rejected").inc()
                self._transition(job_id, SubmissionStatus.REJECTED, detail=str(exc))
                self._record(
                    action="submission:verify",
                    decision=AuditDecision.REJECTED,
                    principal=principal,
                    resource=request.hub_ref,
                    reason=str(exc),
                    job_id=job_id,
                    detail={"required": list(self.attestation_policy.required)},
                )
                raise SubmissionRejected(str(exc), status=422) from exc
            _metrics().verifications.labels(outcome="verified").inc()
            self._record(
                action="submission:verify",
                decision=AuditDecision.VERIFIED,
                principal=principal,
                resource=request.hub_ref,
                reason="cosign signature, SLSA provenance, and SBOM verified",
                job_id=job_id,
                detail=verdict.model_dump(mode="json"),
            )

    def _validate_interface(
        self,
        job_id: str,
        principal: Principal,
        request: HubSubmissionRequest,
        resolved: ResolvedSubmission,
        spec: ScenarioSpec,
    ) -> None:
        """Assert the artifact is a policy whose Core interfaces satisfy the scenario's."""
        try:
            validate_submission_manifest(resolved, spec)
        except ManifestInterfaceError as exc:
            self._transition(job_id, SubmissionStatus.REJECTED, detail=str(exc))
            self._record(
                action="submission:validate",
                decision=AuditDecision.REJECTED,
                principal=principal,
                resource=request.hub_ref,
                reason=str(exc),
                job_id=job_id,
            )
            raise SubmissionRejected(str(exc), status=422) from exc

    def _policy_ref(
        self,
        job_id: str,
        principal: Principal,
        request: HubSubmissionRequest,
        resolved: ResolvedSubmission,
    ) -> str:
        """Read (never import) the manifest's entrypoint — the reference the sandbox will run."""
        try:
            return submission_policy_ref(resolved)
        except HubResolutionError as exc:
            self._transition(job_id, SubmissionStatus.REJECTED, detail=str(exc))
            self._record(
                action="submission:materialize",
                decision=AuditDecision.REJECTED,
                principal=principal,
                resource=request.hub_ref,
                reason=str(exc),
                job_id=job_id,
            )
            raise SubmissionRejected(str(exc), status=422) from exc

    def _reject_execution(
        self,
        principal: Principal,
        resource: str,
        exc: SandboxError,
        *,
        job_id: str | None = None,
    ) -> NoReturn:
        """A submission that did not execute cleanly under its envelope is rejected, not scored."""
        outcome = getattr(exc, "outcome", None)
        status = str(getattr(outcome, "status", "unavailable"))
        _metrics().sandbox_terminations.labels(status=status).inc()
        self._record(
            action="submission:execute",
            decision=AuditDecision.REJECTED,
            principal=principal,
            resource=resource,
            reason=str(exc),
            job_id=job_id,
            detail={"sandbox_status": status},
        )
        raise SubmissionRejected(str(exc), status=422) from exc
